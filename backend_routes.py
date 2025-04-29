from fastapi import FastAPI, Depends, HTTPException
from google.cloud import firestore
from google.api_core import exceptions, retry
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
import logging
import re
import requests
from pytz import UTC
import hashlib
from bs4 import BeautifulSoup
from textblob import TextBlob
from feedparser import parse
from services import (
    strip_html, extract_image, fetch_ebay_sales, identify_watch_model, 
    load_dynamic_criteria, load_brand_prefixes, archive_old_sales, 
    assess_sales_quality, calculate_confidence_score, summarize_text, 
    extract_published_date, refresh_ebay_token
)

# External references from main.py
from main import app, db, EBAY_API_TOKEN, EBAY_API_URL, EBAY_AUTH_ENCODED, PROJECT_ID, logger, get_current_user

@app.get("/fetch-missing-image-urls")
async def fetch_missing_image_urls(limit: int = 100, start_date: str = None):
    """Fetch image URLs from eBay API for watch models missing image_url."""
    global EBAY_API_TOKEN
    if not EBAY_API_TOKEN:
        EBAY_API_TOKEN = refresh_ebay_token(EBAY_AUTH_ENCODED, "https://api.ebay.com/identity/v1/oauth2/token")

    try:
        query = db.collection("watches")
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                query = query.where("created_at", ">=", start_dt.isoformat())
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")

        watches = query.get()
        watch_list = []
        for doc in watches:
            data = doc.to_dict()
            if "image_url" not in data or data["image_url"] is None or data["image_url"] == "":
                watch_list.append((doc.id, data))

        logger.info(f"Found {len(watch_list)} watch models missing image_url after manual filtering")

        if watch_list:
            sample_ids = [wid for wid, _ in watch_list[:5]]
            logger.debug(f"Sample watch IDs missing image_url: {sample_ids}")
        else:
            total_watches = len(list(db.collection("watches").limit(10).get()))
            logger.debug(f"Total watches in collection (sample): {total_watches}")
            raise HTTPException(
                status_code=404,
                detail="No watches found missing image_url. Verify collection data or filters."
            )

        batch = db.batch()
        batch_size = 0
        updated_count = 0
        processed_count = 0
        retries = 3

        for watch_id, watch_data in watch_list:
            source = watch_data.get("source", "")
            processed_count += 1

            match = re.match(r"eBay sale v1\|(\d+)\|\d+", source)
            if not match:
                logger.warning(f"Invalid source format for watch {watch_id}: {source}")
                continue

            item_id = match.group(1)
            detail_url = f"https://api.ebay.com/buy/browse/v1/item/v1|{item_id}|0"
            headers = {"Authorization": f"Bearer {EBAY_API_TOKEN}"}

            for attempt in range(retries):
                try:
                    response = requests.get(detail_url, headers=headers, timeout=5)
                    response.raise_for_status()
                    item_data = response.json()

                    image_url = item_data.get("image", {}).get("imageUrl")
                    if not image_url:
                        logger.debug(f"No image_url found for item {item_id}, watch {watch_id}")
                        break

                    watch_ref = db.collection("watches").document(watch_id)
                    batch.update(watch_ref, {
                        "image_url": image_url,
                        "updated_at": datetime.utcnow().isoformat()
                    })
                    batch_size += 1
                    updated_count += 1
                    logger.info(f"Queued image_url update for watch {watch_id}: {image_url}")
                    break

                except requests.HTTPError as e:
                    if e.response.status_code == 429 and attempt < retries - 1:
                        sleep_time = 2 ** attempt
                        logger.warning(f"Rate limited for item {item_id}, retrying in {sleep_time}s")
                        time.sleep(sleep_time)
                        continue
                    elif e.response.status_code == 404:
                        logger.debug(f"Item {item_id} not found for watch {watch_id}")
                        break
                    elif e.response.status_code == 401:
                        logger.info("Token expired, refreshing")
                        EBAY_API_TOKEN = refresh_ebay_token(EBAY_AUTH_ENCODED, "https://api.ebay.com/identity/v1/oauth2/token")
                        headers["Authorization"] = f"Bearer {EBAY_API_TOKEN}"
                        continue
                    logger.error(f"Failed to fetch item {item_id} for watch {watch_id}: {e}")
                    break
                except requests.RequestException as e:
                    logger.error(f"Network error for item {item_id}, watch {watch_id}: {e}")
                    break

            if batch_size >= 450:
                batch.commit()
                logger.info(f"Committed batch of {batch_size} updates")
                batch = db.batch()
                batch_size = 0

        if batch_size > 0:
            batch.commit()
            logger.info(f"Committed final batch of {batch_size} updates")

        logger.info(f"Processed {processed_count} watches, updated {updated_count} with image_urls")
        return {
            "status": "Completed",
            "processed_count": processed_count,
            "updated_count": updated_count
        }

    except Exception as e:
        logger.error(f"Error in fetch_missing_image_urls: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/consolidate-skus")
async def consolidate_skus(limit: int = None):
    """Consolidate redundant SKUs and aggregate historical sales."""
    try:
        watch_models = {doc.id: doc.to_dict() for doc in db.collection("watches").stream()}
        if not watch_models:
            logger.warning("No watch models found")
            return {"status": "No data", "merged_skus": {}}

        sku_groups = {}
        invalid_skus = []
        
        for sku, model in watch_models.items():
            try:
                brand = (model.get("brand", "") or "").strip().lower()
                model_name = (model.get("model_name", "") or "").strip().lower()
                ref_number = (model.get("reference_number", "") or "").strip().lower()

                if model_name == "watch":
                    invalid_skus.append(sku)
                    continue

                group_key = (brand, model_name, ref_number)
                if group_key not in sku_groups:
                    sku_groups[group_key] = []
                sku_groups[group_key].append(sku)
            except Exception as e:
                logger.error(f"Error processing SKU {sku}: {e}")
                continue

        batch = db.batch()
        clusters = {}
        cluster_id = 0
        consolidations = 0

        for sku in invalid_skus:
            model = watch_models[sku]
            model["archived_at"] = datetime.utcnow().isoformat()
            batch.set(db.collection("watches_archive").document(sku), model)
            batch.delete(db.collection("watches").document(sku))
            logger.info(f"Archived invalid SKU {sku} with model 'watch'")

        for group_key, skus in sku_groups.items():
            if len(skus) <= 1:
                continue

            if limit is not None and consolidations >= limit:
                logger.info(f"Reached consolidation limit of {limit}")
                break

            sku_scores = []
            for sku in skus:
                model = watch_models[sku]
                detail_count = sum(1 for key in ["canonical_name", "movement_type", "dial_color", 
                                               "case_material", "band", "complications"]
                                 if model.get(key) and str(model.get(key)).strip())
                sku_scores.append((sku, detail_count))

            primary_sku, _ = max(sku_scores, key=lambda x: x[1])
            redundant_skus = [s for s in skus if s != primary_sku]

            primary_model = watch_models[primary_sku]
            primary_hourly = primary_model.get("history", {}).get("hourly", [])
            primary_daily = primary_model.get("history", {}).get("daily", [])

            merged_hourly_dict = {
                entry["time"]: {
                    "time": entry["time"],
                    "avg_price": entry["avg_price"] * entry["sales_qty"],
                    "sales_qty": entry["sales_qty"],
                    "confidence": entry["confidence"],
                    "count": 1
                } for entry in primary_hourly
            }
            merged_daily_dict = {
                entry["date"]: {
                    "date": entry["date"],
                    "avg_price": entry["avg_price"] * entry["sales_qty"],
                    "sales_qty": entry["sales_qty"],
                    "confidence": entry["confidence"],
                    "count": 1
                } for entry in primary_daily
            }

            for sku in redundant_skus:
                model = watch_models[sku]
                for entry in model.get("history", {}).get("hourly", []):
                    time_key = entry["time"]
                    if time_key in merged_hourly_dict:
                        merged_hourly_dict[time_key]["avg_price"] += entry["avg_price"] * entry["sales_qty"]
                        merged_hourly_dict[time_key]["sales_qty"] += entry["sales_qty"]
                        merged_hourly_dict[time_key]["confidence"] = min(merged_hourly_dict[time_key]["confidence"], entry["confidence"])
                        merged_hourly_dict[time_key]["count"] += 1
                    else:
                        merged_hourly_dict[time_key] = {
                            "time": time_key,
                            "avg_price": entry["avg_price"] * entry["sales_qty"],
                            "sales_qty": entry["sales_qty"],
                            "confidence": entry["confidence"],
                            "count": 1
                        }
                for entry in model.get("history", {}).get("daily", []):
                    date_key = entry["date"]
                    if date_key in merged_daily_dict:
                        merged_daily_dict[date_key]["avg_price"] += entry["avg_price"] * entry["sales_qty"]
                        merged_daily_dict[date_key]["sales_qty"] += entry["sales_qty"]
                        merged_daily_dict[date_key]["confidence"] = min(merged_daily_dict[date_key]["confidence"], entry["confidence"])
                        merged_daily_dict[date_key]["count"] += 1
                    else:
                        merged_daily_dict[date_key] = {
                            "date": date_key,
                            "avg_price": entry["avg_price"] * entry["sales_qty"],
                            "sales_qty": entry["sales_qty"],
                            "confidence": entry["confidence"],
                            "count": 1
                        }

            merged_hourly = [
                {
                    "time": data["time"],
                    "avg_price": data["avg_price"] / data["sales_qty"],
                    "sales_qty": data["sales_qty"],
                    "confidence": data["confidence"]
                } for data in merged_hourly_dict.values()
            ]
            merged_daily = [
                {
                    "date": data["date"],
                    "avg_price": data["avg_price"] / data["sales_qty"],
                    "sales_qty": data["sales_qty"],
                    "confidence": data["confidence"]
                } for data in merged_daily_dict.values()
            ]

            batch.update(db.collection("watches").document(primary_sku), {
                "history.hourly": merged_hourly,
                "history.daily": merged_daily,
                "updated_at": datetime.utcnow().isoformat()
            })

            clusters[primary_sku] = redundant_skus
            for sku in redundant_skus:
                model = watch_models[sku]
                model["merged_into"] = primary_sku
                model["archived_at"] = datetime.utcnow().isoformat()
                batch.set(db.collection("watches_archive").document(sku), model)
                batch.delete(db.collection("watches").document(sku))
            consolidations += len(redundant_skus)

        batch.commit()
        logger.info(f"Consolidated {consolidations} SKUs into {len(clusters)} clusters")
        return {"status": "Consolidation complete", "merged_skus": clusters}

    except Exception as e:
        logger.error(f"Error in consolidate_skus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Pulls eBay data and processes sales
@app.get("/pull-ebay-data")
async def pull_ebay_data(max_sales_per_pull: int = 50, hours_window: int = 1):
    
    now = datetime.utcnow()
    start_time = (now - timedelta(hours=hours_window)).isoformat() + "Z"
    end_time = now.isoformat() + "Z"
    
    config_ref = db.collection("config").document("ebay_pull")
    config = config_ref.get().to_dict() or {"last_pull": "1970-01-01T00:00:00Z"}
    if config["last_pull"] > start_time:
        start_time = config["last_pull"]
    logger.info(f"Pulling sales from {start_time} to {end_time}")
    
    sales = fetch_ebay_sales(EBAY_API_TOKEN,EBAY_API_URL,start_time=start_time, end_time=end_time, max_sales=max_sales_per_pull)
    
    sales_count = 0
    batch = db.batch()
    timestamp = datetime.utcnow().isoformat()
    for sale in sales:
        # Query Firestore for existing sale with same itemId and lastSoldDate
        existing_sales = db.collection("sales") \
                           .where("itemId", "==", sale["itemId"]) \
                           .limit(1) \
                           .get()
        
        if existing_sales:
            logger.info(f"Skipping duplicate sale: itemId={sale['itemId']}, lastSoldDate={sale['lastSoldDate']}")
            continue
        
        sale_doc = {
            "itemId": sale["itemId"],
            "title": sale["title"],
            "lastSoldPrice": {"value": float(sale["lastSoldPrice"]["value"]), "currency": sale["lastSoldPrice"]["currency"]},
            "lastSoldDate": sale["lastSoldDate"],
            "condition": sale.get("condition", "Unknown"),
            "itemAspects": sale.get("itemAspects", {}),
            "description": strip_html(sale.get("description", "No description available")),
            "seller": sale.get("seller", 0),
            "sku": None,
            "image_url": sale["image_url"],
            "timestamp": timestamp,
            "processed": False
        }
        # Use itemId as doc ID for now, but deduplication prevents overwrites
        sale_ref = db.collection("sales").document(sale["itemId"])
        batch.set(sale_ref, sale_doc, merge=True)  # Merge to preserve sku/processed if present
        sales_count += 1
        if sales_count % 500 == 0:
            batch.commit()
            batch = db.batch()
    batch.commit()
    
    config_ref.set({"last_pull": end_time})
    logger.info(f"Pulled and stored {sales_count} new sales")
    return {"status": "Data pulled", "sales_processed": sales_count}

# Enhance in future to use AutoML to identify model and update sku into raw_sales directly. Inefficient to create new collection to store sales data again.
@app.get("/identify-raw-sales")
async def identify_raw_sales(date: Optional[str] = None):
    watch_models = {doc.id: doc.to_dict() for doc in db.collection("watches").get()}
    term_sets, regex_patterns, weights = load_dynamic_criteria(db)
    brand_prefixes = load_brand_prefixes(term_sets, db)
    
    sales_ref = db.collection("sales")
    unprocessed_sales = sales_ref.where(filter=firestore.FieldFilter("sku", "==", None)).get()
    
    sales_count = 0
    batch = db.batch()
    for sale_doc in unprocessed_sales:
        sale = sale_doc.to_dict()
        sku = identify_watch_model(sale, watch_models, term_sets, brand_prefixes, regex_patterns, weights, db)
        batch.update(sale_doc.reference, {"sku": sku})
        
        sales_count += 1
        if sales_count % 500 == 0:
            batch.commit()
            batch = db.batch()
    
    batch.commit()
    logger.info(f"Processed {sales_count} sales")
    return {"status": "Sales processed", "sales_count": sales_count}

# Runs periodic aggregation for all watches
@app.get("/aggregate")
async def run_aggregation():
    retry_policy = retry.Retry(
        predicate=retry.if_exception_type(exceptions.ServiceUnavailable, exceptions.DeadlineExceeded),
        initial=1.0, maximum=10.0, multiplier=2.0, deadline=60.0
    )

    try:
        watch_models = {doc.id: doc.to_dict() for doc in retry_policy(lambda: db.collection("watches").get())()}
    except exceptions.GoogleAPIError as e:
        logger.error(f"Failed to fetch watch models: {str(e)}")
        return {"status": "Failed", "error": "Unable to fetch watch models"}

    term_sets, regex_patterns, weights = load_dynamic_criteria(db)

    processed_sales_ids = set()
    aggregated_sales_count = 0
    batch = db.batch()
    batch_size = 0

    # Fetch all unprocessed sales once
    try:
        all_sales = retry_policy(lambda: db.collection("sales") \
                         .where("sku", "!=", None) \
                         .where("processed", "==", False) \
                         .stream())()
    except exceptions.GoogleAPIError as e:
        logger.error(f"Failed to fetch unprocessed sales: {str(e)}")
        return {"status": "Failed", "error": "Unable to fetch sales"}

    if not all_sales:
        logger.info("No unprocessed sales with SKUs found")
        return {"status": "No data to aggregate"}

    # Determine time range for 4-hour buckets and filter out Unidentified SKUs
    sales_list = [(sale.id, sale.to_dict()) for sale in all_sales]
    valid_sales_list = [(sale_id, sale_dict) for sale_id, sale_dict in sales_list 
                        if sale_dict.get("sku") != "Unidentified"]
    if not valid_sales_list:
        logger.info("All sales were Unidentified; no valid data to aggregate")
        # Still mark all sales as processed
        processed_sales_ids.update(sale_id for sale_id, _ in sales_list)
    else:
        timestamps = [datetime.fromisoformat(s[1].get("timestamp", "1970-01-01T00:00:00Z").rstrip("Z")) 
                      for s in valid_sales_list]
        oldest_time = min(timestamps)
        newest_time = max(timestamps)

        start_time = oldest_time.replace(minute=0, second=0, microsecond=0)
        while start_time.hour % 4 != 0:
            start_time -= timedelta(hours=1)
        end_time = newest_time.replace(minute=59, second=59, microsecond=999999)
        time_buckets = []
        current = start_time
        while current < end_time:
            bucket_end = current + timedelta(hours=4)
            if bucket_end > end_time:
                bucket_end = end_time
            time_buckets.append((current, bucket_end))
            current = bucket_end

        # Group sales by SKU, date, and 4-hour bucket
        sales_by_sku = {}
        for sale_id, sale_dict in valid_sales_list:
            timestamp = datetime.fromisoformat(sale_dict.get("timestamp", "1970-01-01T00:00:00Z").rstrip("Z"))
            date_key = timestamp.strftime("%Y-%m-%d")
            hour_key = next((start.strftime("%Y-%m-%dT%H:00:00Z") for start, end in time_buckets 
                            if start <= timestamp < end), None)
            if hour_key:  # Only process if sale fits a bucket
                sku = sale_dict["sku"]
                sales_by_sku.setdefault(sku, {}).setdefault("daily", {}).setdefault(date_key, []).append((sale_id, sale_dict))
                sales_by_sku[sku].setdefault("hourly", {}).setdefault(hour_key, []).append((sale_id, sale_dict))
                processed_sales_ids.add(sale_id)

        if not sales_by_sku:
            logger.info("No valid sales matched time buckets")
        else:
            # Process both daily and hourly aggregations
            for sku, aggregations in sales_by_sku.items():
                try:
                    watch_ref = db.collection("watches").document(sku)
                    watch_doc = retry_policy(lambda: watch_ref.get())()
                    hist_avg = None
                    daily_history = []
                    hourly_history = []
                    if watch_doc.exists:
                        watch_model = watch_doc.to_dict()
                        daily_history = watch_model.get("history", {}).get("daily", [])
                        hourly_history = watch_model.get("history", {}).get("hourly", [])
                        if daily_history:
                            hist_avg = float(daily_history[-1]["avg_price"])
                    else:
                        logger.warning(f"Watch model {sku} not found in `watches` collection")
                        continue
                except exceptions.GoogleAPIError as e:
                    logger.error(f"Failed to fetch watch {sku}: {str(e)}")
                    continue

                # Daily aggregation
                for date_key, daily_sales in aggregations.get("daily", {}).items():
                    sale_data = [s[1] for s in daily_sales]
                    authentic_prices = assess_sales_quality(sale_data, sku, term_sets, weights, watch_models)
                    if authentic_prices:
                        aggregated_sales_count += len(sale_data)
                        existing_daily = next((entry for entry in daily_history if entry["date"] == date_key), None)
                        if existing_daily:
                            total_sales = existing_daily["sales_qty"] + len(authentic_prices)
                            total_value = (existing_daily["avg_price"] * existing_daily["sales_qty"]) + sum(authentic_prices)
                            avg_price = total_value / total_sales
                            confidence = calculate_confidence_score(
                                authentic_prices + [existing_daily["avg_price"]] * existing_daily["sales_qty"], 
                                hist_avg, total_sales
                            )
                            updated_entry = {"date": date_key, "avg_price": avg_price, "sales_qty": total_sales, "confidence": confidence}
                            batch.update(watch_ref, {"history.daily": firestore.ArrayRemove([existing_daily])})
                            batch.update(watch_ref, {"history.daily": firestore.ArrayUnion([updated_entry])})
                        else:
                            avg_price = sum(authentic_prices) / len(authentic_prices)
                            confidence = calculate_confidence_score(authentic_prices, hist_avg, len(sale_data))
                            daily_entry = {"date": date_key, "avg_price": avg_price, "sales_qty": len(authentic_prices), "confidence": confidence}
                            batch.update(watch_ref, {"history.daily": firestore.ArrayUnion([daily_entry])})
                        batch_size += 2 if existing_daily else 1

                # Hourly aggregation
                for hour_key, hourly_sales in aggregations.get("hourly", {}).items():
                    sale_data = [s[1] for s in hourly_sales]
                    authentic_prices = assess_sales_quality(sale_data, sku, term_sets, weights, watch_models)
                    if authentic_prices:
                        aggregated_sales_count += len(sale_data)
                        existing_hourly = next((entry for entry in hourly_history if entry["time"] == hour_key), None)
                        if existing_hourly:
                            total_sales = existing_hourly["sales_qty"] + len(authentic_prices)
                            total_value = (existing_hourly["avg_price"] * existing_hourly["sales_qty"]) + sum(authentic_prices)
                            avg_price = total_value / total_sales
                            confidence = calculate_confidence_score(
                                authentic_prices + [existing_hourly["avg_price"]] * existing_hourly["sales_qty"], 
                                hist_avg, total_sales
                            )
                            updated_entry = {"time": hour_key, "avg_price": avg_price, "sales_qty": total_sales, "confidence": confidence}
                            batch.update(watch_ref, {"history.hourly": firestore.ArrayRemove([existing_hourly])})
                            batch.update(watch_ref, {"history.hourly": firestore.ArrayUnion([updated_entry])})
                        else:
                            avg_price = sum(authentic_prices) / len(authentic_prices)
                            confidence = calculate_confidence_score(authentic_prices, hist_avg, len(sale_data))
                            hourly_entry = {"time": hour_key, "avg_price": avg_price, "sales_qty": len(authentic_prices), "confidence": confidence}
                            batch.update(watch_ref, {"history.hourly": firestore.ArrayUnion([hourly_entry])})
                        batch_size += 2 if existing_hourly else 1

                # Update timestamp for watch
                batch.update(watch_ref, {"updated_at": datetime.utcnow().isoformat()})
                batch_size += 1

                if batch_size >= 450:
                    retry_policy(lambda: batch.commit())()
                    batch = db.batch()
                    batch_size = 0

    # Mark all sales as processed (including Unidentified ones)
    try:
        for sale_id in processed_sales_ids:
            batch.update(db.collection("sales").document(sale_id), {"processed": True})
            batch_size += 1
            if batch_size >= 450:
                retry_policy(lambda: batch.commit())()
                logger.info(f"Committed batch of {batch_size} sales marked as processed")
                batch = db.batch()
                batch_size = 0
        if batch_size > 0:
            retry_policy(lambda: batch.commit())()
            logger.info(f"Committed final batch of {batch_size} sales marked as processed")
    except exceptions.GoogleAPIError as e:
        logger.error(f"Failed to mark sales as processed: {str(e)}")
        return {"status": "Partial success", "error": "Aggregation completed but failed to mark sales as processed", 
                "aggregated_sales_count": aggregated_sales_count}

    logger.info(f"Aggregation completed (daily and 4-hour) with {aggregated_sales_count} sales aggregated")
    return {"status": "Aggregated", "aggregated_sales_count": aggregated_sales_count}

# Clean HTML from raw sales
@app.get("/clean-html")
async def clean_html_endpoint():
    now = datetime.utcnow()
    start_time = (now - timedelta(days=3)).isoformat() + "Z"
    # Use collection_group to query all 'sales' subcollections
    sales_ref = db.collection_group("sales").where(
        filter=firestore.FieldFilter("timestamp", ">=", start_time)
    ).get()
    sales = [(s.id, s.to_dict(), s.reference) for s in sales_ref]
    cleaned_count = 0
    total_sales = len(sales)
    logger.info(f"Pulled {total_sales} raw sales from Firestore")    
    batch = db.batch()
    updates_per_batch = 500
    
    for i, (sale_id, sale, doc_ref) in enumerate(sales):
        raw_desc = sale.get("description", "")
        if "<" in raw_desc:
            cleaned_count += 1
            clean_desc = strip_html(raw_desc)
            batch.update(doc_ref, {"description": clean_desc})
            logger.info(f"Queued clean for {sale_id}")
        
        if (i + 1) % updates_per_batch == 0 or i == len(sales) - 1:
            batch.commit()
            logger.info(f"Committed batch of updates")
            batch = db.batch()
    
    return {"message": "Cleaning completed", "cleaned_sales_count": cleaned_count}

# GCP required health check response
@app.get("/")
async def root():
    return {"status": "OK"}

# Reset processed flag for sales older than x
@app.get("/reset-processed")
async def reset_processed(cutoff: str = "2025-03-01T00:00:00Z"):
    cutoff_datetime = datetime.fromisoformat(cutoff.rstrip("Z"))
    # Calculate cutoff datetime (UTC now minus days_back)
    cutoff_str = cutoff_datetime.isoformat() + "Z"
    
    # Query sales older than cutoff
    sales_ref = db.collection("sales") \
                  .where("processed", "==", True) \
                  .stream()  # Stream for large datasets
    
    updated_count = 0
    batch = db.batch()
    batch_size = 0
    
    for sale in sales_ref:
        sale_id = sale.id
        batch.update(db.collection("sales").document(sale_id), {"processed": False, "sku": None})
        batch_size += 1
        updated_count += 1
        # logger.info(f"Resetting processed flag for sale {sale_id}")
    
        # Commit batch at 500 updates (Firestore limit)
        if batch_size >= 500:
            batch.commit()
            logger.info(f"Committed batch of {batch_size} updates")
            batch = db.batch()
            batch_size = 0
    
    # Commit any remaining updates
    if batch_size > 0:
        batch.commit()
        logger.info(f"Committed final batch of {batch_size} updates")
    
    logger.info(f"Reset processed flag for {updated_count} sales older than {cutoff_str}")
    return {"status": "Processed flags reset", "updated_count": updated_count}

@app.get("/precompute-trending")
async def precompute_trending(max_items: int = 10):
    """Precompute trending watches, brands, and models by sales volume and growth."""
    try:
        now = datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")
        periods = {
            "sales": ["last_7_days", "last_30_days", "last_90_days"],
            "growth": ["last_30_days", "last_90_days"]
        }
        sales_data = {}
        watch_models = {doc.id: doc.to_dict() for doc in db.collection("watches").stream()}

        for period in periods["sales"]:
            days = {"last_7_days": 7, "last_30_days": 30, "last_90_days": 90}[period]
            start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
            sales_totals = {}
            for watch_id, watch in watch_models.items():
                daily_history = sorted(watch.get("history", {}).get("daily", []), key=lambda x: x["date"])
                filtered = [entry for entry in daily_history if entry["date"] >= start_date]
                total_sales = sum(entry["sales_qty"] for entry in filtered)
                if total_sales > 0:
                    latest_entry = max(filtered, key=lambda x: x["date"], default=None)
                    sales_totals[watch_id] = {
                        "sales_count": total_sales,
                        "avg_price": float(latest_entry["avg_price"]) if latest_entry else 0.0,
                        "brand": watch.get("brand", ""),
                        "model_name": watch.get("model_name", ""),
                        "canonical_name": watch.get("canonical_name", ""),
                        "image_url": watch.get("image_url", "")
                    }
        
            trending = sorted(sales_totals.items(), key=lambda x: x[1]["sales_count"], reverse=True)[:max_items]
            sales_data[period] = [
                {
                    "watch_id": watch_id,
                    "brand": data["brand"],
                    "model_name": data["model_name"],
                    "sales_count": data["sales_count"],
                    "avg_price": data["avg_price"],
                    "canonical_name": data["canonical_name"],
                    "image_url": data["image_url"]
                } for watch_id, data in trending
            ]
    
        growth_data = {}
        for period in periods["growth"]:
            days = int(period.split("_")[1].replace("days", ""))
            start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
            growth_totals = {}
        
            for watch_id, watch in watch_models.items():
                daily_history = sorted(watch.get("history", {}).get("daily", []), key=lambda x: x["date"])
                filtered = [entry for entry in daily_history if entry["date"] >= start_date]
                if len(filtered) >= 2:
                    start_price = float(filtered[0]["avg_price"])
                    end_price = float(filtered[-1]["avg_price"])
                    if start_price > 0:
                        growth_percentage = ((end_price - start_price) / start_price) * 100
                        growth_totals[watch_id] = {
                            "growth_percentage": round(growth_percentage, 2),
                            "current_avg_price": end_price,
                            "brand": watch.get("brand", ""),
                            "model_name": watch.get("model_name", ""),
                            "canonical_name": watch.get("canonical_name", ""),
                            "image_url": watch.get("image_url", "")
                        }
        
            trending = sorted(growth_totals.items(), key=lambda x: x[1]["growth_percentage"], reverse=True)[:max_items]
            growth_data[period] = [
                {
                    "watch_id": watch_id,
                    "brand": data["brand"],
                    "model_name": data["model_name"],
                    "growth_percentage": data["growth_percentage"],
                    "current_avg_price": data["current_avg_price"],
                    "canonical_name": data["canonical_name"],
                    "image_url": data["image_url"]
                } for watch_id, data in trending
            ]

        # Precompute brand sales trends
        brand_sales_data = {}
        for period in periods["sales"]:
            days = {"last_7_days": 7, "last_30_days": 30, "last_90_days": 90}[period]
            start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
            brand_totals = {}
            for watch_id, watch in watch_models.items():
                brand = watch.get("brand", "").title()
                if not brand:
                    continue
                daily_history = sorted(watch.get("history", {}).get("daily", []), key=lambda x: x["date"])
                filtered = [entry for entry in daily_history if entry["date"] >= start_date]
                total_sales = sum(entry["sales_qty"] for entry in filtered)
                if total_sales > 0:
                    latest_entry = max(filtered, key=lambda x: x["date"], default=None)
                    if brand not in brand_totals:
                        brand_totals[brand] = {
                            "sales_count": 0,
                            "total_value": 0.0,
                            "watch_ids": set(),
                            "image_url": watch.get("image_url", "")
                        }
                    brand_totals[brand]["sales_count"] += total_sales
                    brand_totals[brand]["total_value"] += float(latest_entry["avg_price"]) * total_sales if latest_entry else 0.0
                    brand_totals[brand]["watch_ids"].add(watch_id)
            
            trending = sorted(brand_totals.items(), key=lambda x: x[1]["sales_count"], reverse=True)[:max_items]
            brand_sales_data[period] = [
                {
                    "brand": brand,
                    "sales_count": data["sales_count"],
                    "avg_price": data["total_value"] / data["sales_count"] if data["sales_count"] > 0 else 0.0,
                    "watch_count": len(data["watch_ids"]),
                    "image_url": data["image_url"]
                } for brand, data in trending
            ]

        # Precompute model sales trends
        model_sales_data = {}
        for period in periods["sales"]:
            days = {"last_7_days": 7, "last_30_days": 30, "last_90_days": 90}[period]
            start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
            model_totals = {}
            for watch_id, watch in watch_models.items():
                model_name = watch.get("model_name", "").title()
                if not model_name:
                    continue
                daily_history = sorted(watch.get("history", {}).get("daily", []), key=lambda x: x["date"])
                filtered = [entry for entry in daily_history if entry["date"] >= start_date]
                total_sales = sum(entry["sales_qty"] for entry in filtered)
                if total_sales > 0:
                    latest_entry = max(filtered, key=lambda x: x["date"], default=None)
                    if model_name not in model_totals:
                        model_totals[model_name] = {
                            "sales_count": 0,
                            "total_value": 0.0,
                            "watch_ids": set(),
                            "image_url": watch.get("image_url", ""),
                            "brand": watch.get("brand", "")
                        }
                    model_totals[model_name]["sales_count"] += total_sales
                    model_totals[model_name]["total_value"] += float(latest_entry["avg_price"]) * total_sales if latest_entry else 0.0
                    model_totals[model_name]["watch_ids"].add(watch_id)
            
            trending = sorted(model_totals.items(), key=lambda x: x[1]["sales_count"], reverse=True)[:max_items]
            model_sales_data[period] = [
                {
                    "model_name": model_name,
                    "sales_count": data["sales_count"],
                    "avg_price": data["total_value"] / data["sales_count"] if data["sales_count"] > 0 else 0.0,
                    "watch_count": len(data["watch_ids"]),
                    "image_url": data["image_url"],
                    "brand": data["brand"]
                } for model_name, data in trending
            ]
    
        batch = db.batch()
        trending_doc = {
            "date": date_str,
            "sales": sales_data,
            "growth": growth_data,
            "brands": brand_sales_data,
            "models": model_sales_data,
            "updated_at": now.isoformat()
        }
        batch.set(db.collection("trending").document(date_str), trending_doc)
        batch.commit()
    
        logger.info(f"Precomputed trending lists for watches, brands, and models for {date_str}")
        return {"status": "Trending lists precomputed", "date": date_str}

    except Exception as e:
        logger.error(f"Error in precompute_trending: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/pulse/sync")
async def sync_pulse(token: Optional[str] = None):
    """Trigger scraping, sentiment analysis, and trend aggregation for watch content."""
    try:
        logger.info("Starting pulse sync")
        sources = [
            {
                "url": "https://www.hodinkee.com/articles",
                "type": "article",
                "selector": ".content-card",
                "content_selector": ".article-body, .post-content",
            },
            {
                "url": "https://www.ablogtowatch.com/feed/",
                "type": "rss",
            },
            {
                "url": "https://www.watchuseek.com/forums/whats-new/",
                "type": "article",
                "selector": ".threadinfo",
                "content_selector": ".postcontent",
            },
            {
                "url": "https://www.fratellowatches.com/feed/",
                "type": "rss",
            },
        ]
        content_items = []
        brand_mentions = {}
        model_mentions = {}
        topic_trends = {}

        term_sets, regex_patterns, _ = load_dynamic_criteria(db)
        
        for source in sources:
            headers = {"User-Agent": "TickTrend/1.0 (+https://ticktrend.app)"}
            try:
                response = requests.get(source["url"], headers=headers, timeout=10)
                response.raise_for_status()
                logger.debug(f"Fetched {source['url']} successfully")
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch {source['url']}: {str(e)}")
                continue

            if source["type"] == "article":
                soup = BeautifulSoup(response.text, "html.parser")
                articles = soup.select(source["selector"])
                for article in articles:
                    title_elem = article.select_one("h2, h3, .title")
                    title = strip_html(title_elem.text.strip()) if title_elem else "Untitled"
                    link_elem = article.select_one("a")
                    link = link_elem["href"] if link_elem else "#"
                    if not link.startswith("http"):
                        link = urljoin(source["url"], link)
                    try:
                        article_response = requests.get(link, headers=headers, timeout=10)
                        article_soup = BeautifulSoup(article_response.text, "html.parser")
                        content_elem = article_soup.select_one(source.get("content_selector", "p"))
                        content = strip_html(content_elem.text.strip()) if content_elem else ""
                        summary = summarize_text(content)
                        image_url = extract_image(article_soup, {}, "article", link)
                    except Exception as e:
                        logger.warning(f"Failed to fetch article content for {link}: {str(e)}")
                        summary_elem = article.select_one("p, .summary")
                        summary = strip_html(summary_elem.text.strip())[:200] if summary_elem else ""
                        image_url = None
                    published = extract_published_date(article_soup if 'article_soup' in locals() else soup, {}, "article")
                    content_items.append((title, link, summary, published, source["url"], image_url))
                    logger.debug(f"Processed article {link} with image: {image_url}")
            elif source["type"] == "rss":
                feed = parse(source["url"])
                for entry in feed.entries:
                    title = strip_html(entry.get("title", "Untitled"))
                    link = entry.get("link", "#")
                    content = strip_html(entry.get("content", [{}])[0].get("value", ""))
                    summary = summarize_text(content) if content else strip_html(entry.get("summary", ""))[:200]
                    published = extract_published_date(None, entry, "rss")
                    image_url = extract_image(None, entry, "rss", link)
                    content_items.append((title, link, summary, published, source["url"], image_url))
                    logger.debug(f"Processed RSS entry {link} with image: {image_url}")

        batch = db.batch()
        for title, link, summary, published, source_url, image_url in content_items:
            item_id = hashlib.sha1(link.encode()).hexdigest()
            text = f"{title} {summary}"
            blob = TextBlob(text)
            subjectivity_score = blob.sentiment.subjectivity
            sentiment_score = blob.sentiment.polarity
            sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
            tags = []
            for word in title.split() + summary.split():
                word = word.lower()
                if word in term_sets.get("VALID_TAGS", []) or word in ["vintage", "luxury", "review", "news", "limited"]:
                    tags.append(word)
            tags = list(set(tags))

            brand_match = re.search(regex_patterns["brand"], text, re.IGNORECASE)
            brand = brand_match.group(0).title() if brand_match and brand_match.group(0).lower() in term_sets["VALID_BRANDS"] else None

            model_match = re.search(regex_patterns["model_name"], text, re.IGNORECASE)
            model = model_match.group(0).title() if model_match else None

            topics = []
            for topic, pattern in regex_patterns.get("topics", {}).items():
                deleteme = regex_patterns.get("topics")
                if re.search(pattern, text, re.IGNORECASE):
                    topics.append(topic)

            content_data = {
                "id": item_id,
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "subjectivity_score": subjectivity_score,
                "tags": tags,
                "brand": brand,
                "model": model,
                "topics": topics,
                "source": source_url,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
            }
            if image_url:
                content_data["image_url"] = image_url

            batch.set(db.collection("content").document(item_id), content_data)
            logger.debug(f"Stored content item {item_id} with image: {image_url}")

            if brand:
                brand_mentions[brand] = brand_mentions.get(brand, {"count": 0, "sentiment": 0.0, "links": []})
                brand_mentions[brand]["count"] += 1
                brand_mentions[brand]["sentiment"] += sentiment_score
                brand_mentions[brand]["links"].append(link)
            if model:
                model_key = model
                model_mentions[model_key] = model_mentions.get(model_key, {"count": 0, "sentiment": 0.0, "links": [], "brands": set()})
                model_mentions[model_key]["count"] += 1
                model_mentions[model_key]["sentiment"] += sentiment_score
                model_mentions[model_key]["links"].append(link)
                if brand:
                    model_mentions[model_key]["brands"].add(brand)
            for topic in topics:
                topic_trends[topic] = topic_trends.get(topic, {"count": 0, "sentiment": 0.0})
                topic_trends[topic]["count"] += 1
                topic_trends[topic]["sentiment"] += sentiment_score

        batch.commit()

        now = datetime.utcnow()
        start_date = now - timedelta(days=7)
        start_date_str = start_date.strftime("%Y-%m-%d")

        watches = {doc.id: doc.to_dict() for doc in db.collection("watches").get()}
        
        brand_sales = {}
        for brand in brand_mentions:
            brand_sales[brand] = {"sales_count": 0, "prices": [], "watch_ids": set()}
            for watch_id, watch in watches.items():
                if watch.get("brand", "").lower() == brand.lower():
                    daily_history = watch.get("history", {}).get("daily", [])
                    for entry in daily_history:
                        if entry["date"] >= start_date_str:
                            brand_sales[brand]["sales_count"] += entry["sales_qty"]
                            brand_sales[brand]["prices"].extend([entry["avg_price"]] * entry["sales_qty"])
                            brand_sales[brand]["watch_ids"].add(watch_id)

        model_sales = {}
        for model_key in model_mentions:
            model_sales[model_key] = {"sales_count": 0, "prices": [], "watch_ids": set(), "brands": model_mentions[model_key]["brands"]}
            for watch_id, watch in watches.items():
                model_name = watch.get("model_name", "").lower()
                if model_key.lower() in model_name:
                    daily_history = watch.get("history", {}).get("daily", [])
                    for entry in daily_history:
                        if entry["date"] >= start_date_str:
                            model_sales[model_key]["sales_count"] += entry["sales_qty"]
                            model_sales[model_key]["prices"].extend([entry["avg_price"]] * entry["sales_qty"])
                            model_sales[model_key]["watch_ids"].add(watch_id)

        batch = db.batch()
        for brand, data in brand_mentions.items():
            avg_sentiment = data["sentiment"] / data["count"] if data["count"] > 0 else 0
            sales_data = brand_sales.get(brand, {"sales_count": 0, "prices": []})
            price_change = 0.0
            if sales_data["prices"]:
                current_avg = sum(sales_data["prices"]) / len(sales_data["prices"])
                historical_prices = []
                for watch_id in sales_data["watch_ids"]:
                    watch = watches.get(watch_id, {})
                    daily_history = watch.get("history", {}).get("daily", [])
                    historical_prices.extend(
                        entry["avg_price"] for entry in daily_history if entry["date"] < start_date_str
                    )
                historical_avg = sum(historical_prices) / len(historical_prices) if historical_prices else current_avg
                price_change = ((current_avg - historical_avg) / historical_avg * 100) if historical_avg != 0 else 0.0

            trend_ref = db.collection("trending_brands").document(hashlib.sha1(brand.encode()).hexdigest())
            batch.set(trend_ref, {
                "name": brand,
                "type": "brand",
                "sentiment_score": avg_sentiment,
                "mentions": data["count"],
                "top_links": data["links"][:3],
                "sales_count": sales_data["sales_count"],
                "price_change_percent": round(price_change, 2),
                "updated_at": datetime.utcnow().isoformat(),
                "logo_url": None
            })

        for model_key, data in model_mentions.items():
            avg_sentiment = data["sentiment"] / data["count"] if data["count"] > 0 else 0
            sales_data = model_sales.get(model_key, {"sales_count": 0, "prices": [], "brands": set()})
            price_change = 0.0
            if sales_data["prices"]:
                current_avg = sum(sales_data["prices"]) / len(sales_data["prices"])
                historical_prices = []
                for watch_id in sales_data["watch_ids"]:
                    watch = watches.get(watch_id, {})
                    daily_history = watch.get("history", {}).get("daily", [])
                    historical_prices.extend(
                        entry["avg_price"] for entry in daily_history if entry["date"] < start_date_str
                    )
                historical_avg = sum(historical_prices) / len(historical_prices) if historical_prices else current_avg
                price_change = ((current_avg - historical_avg) / historical_avg * 100) if historical_avg != 0 else 0.0

            trend_ref = db.collection("trending_models").document(hashlib.sha1(model_key.encode()).hexdigest())
            batch.set(trend_ref, {
                "name": model_key,
                "type": "model",
                "sentiment_score": avg_sentiment,
                "mentions": data["count"],
                "top_links": data["links"][:3],
                "sales_count": sales_data["sales_count"],
                "price_change_percent": round(price_change, 2),
                "top_brands": list(sales_data["brands"])[:3],
                "updated_at": datetime.utcnow().isoformat()
            })

        for topic, data in topic_trends.items():
            avg_sentiment = data["sentiment"] / data["count"] if data["count"] > 0 else 0
            trend_ref = db.collection("trends").document(hashlib.sha1(topic.encode()).hexdigest())
            batch.set(trend_ref, {
                "name": topic,
                "type": "topic",
                "sentiment_score": avg_sentiment,
                "mentions": data["count"],
                "sentiment_history": firestore.ArrayUnion([{
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "score": avg_sentiment,
                    "mentions": data["count"]
                }]),
                "updated_at": datetime.utcnow().isoformat()
            }, merge=True)

        batch.commit()

        logger.info(f"Synced {len(content_items)} content items, {len(brand_mentions)} brands, {len(model_mentions)} models, {len(topic_trends)} topics")
        return {"status": "Sync complete", "content_items": len(content_items)}
    except Exception as e:
        logger.error(f"Error syncing pulse: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")