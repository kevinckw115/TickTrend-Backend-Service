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

class AddWatchRequest(BaseModel):
    watch_id: str
    purchase_price: float | None = None

class PulseContent(BaseModel):
    id: str
    title: str
    link: str
    summary: str
    published: str
    sentiment: str
    sentiment_score: float
    tags: List[str]
    brand: Optional[str]
    model: Optional[str]
    source: str
    image_url: Optional[str] = None

class PulseTrend(BaseModel):
    name: str
    type: str
    sentiment_score: float
    mentions: int
    sentiment_history: List[dict]

class BrandTrend(BaseModel):
    name: str
    type: str
    sentiment_score: float
    mentions: int
    top_links: List[str]
    updated_at: str
    sales_count: int
    price_change_percent: float
    logo_url: Optional[str] = None
    trend_score: float
    trend_summary: str

class ModelTrend(BaseModel):
    name: str
    type: str
    sentiment_score: float
    mentions: int
    top_links: List[str]
    updated_at: str
    sales_count: int
    price_change_percent: float
    top_brands: List[str]
    trend_score: float
    trend_summary: str

@app.get("/watches/{watch_id}", dependencies=[Depends(get_current_user)])
async def get_watch_details(watch_id: str):
    """Retrieve detailed information about a specific watch model."""
    watch_ref = db.collection("watches").document(watch_id).get()
    if not watch_ref.exists:
        raise HTTPException(status_code=404, detail="Watch not found")
    
    watch = watch_ref.to_dict()
    daily_history = sorted(watch.get("history", {}).get("daily", []), key=lambda x: x["date"], reverse=True)
    last_known_price = {
        "date": daily_history[0]["date"],
        "avg_price": float(daily_history[0]["avg_price"])
    } if daily_history else None
    
    total_sales = sum(entry["sales_qty"] for entry in daily_history)
    
    response = {
        "watch_id": watch_id,
        "canonical_name": watch.get("canonical_name", ""),
        "brand": watch.get("brand", ""),
        "model_name": watch.get("model_name", ""),
        "reference_number": watch.get("reference_number", ""),
        "specifications": {
            "case_material": watch.get("case_material"),
            "movement": watch.get("movement_type"),
            "dial_color": watch.get("dial_color"),
            "case_size": watch.get("case_size"),
            "water_resistance": watch.get("water_resistance")
        },
        "last_known_price": last_known_price,
        "total_sales": total_sales,
        "image_url": watch.get("image_url", "")
    }
    return response

@app.get("/watches/{watch_id}/sales", dependencies=[Depends(get_current_user)])
async def get_watch_sales(watch_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = 20, offset: int = 0):
    """Fetch individual sales for a watch model with optional date filtering."""
    logger.info(f"Fetching sales for watch_id: {watch_id}, limit: {limit}, offset: {offset}, start_date: {start_date}, end_date: {end_date}")
    
    try:
        watch_ref = db.collection("watches").document(watch_id).get()
        if not watch_ref.exists:
            logger.warning(f"Watch not found: {watch_id}")
            raise HTTPException(status_code=404, detail="Watch not found")

        sales_query = db.collection("sales").where("sku", "==", watch_id)
        
        if start_date:
            try:
                datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                sales_query = sales_query.where("lastSoldDate", ">=", start_date)
            except ValueError as e:
                logger.error(f"Invalid start_date format: {start_date}, error: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        
        if end_date:
            try:
                datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                sales_query = sales_query.where("lastSoldDate", "<=", end_date)
            except ValueError as e:
                logger.error(f"Invalid end_date format: {end_date}, error: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid end_date format")

        sales_query = sales_query.order_by("lastSoldDate", direction=firestore.Query.DESCENDING).offset(offset).limit(limit)
        
        sales = sales_query.get()
        sales_data = []
        
        for sale in sales:
            sale_dict = sale.to_dict()
            try:
                last_sold_date = sale_dict.get("lastSoldDate")
                if not last_sold_date:
                    logger.warning(f"Sale {sale.id} missing lastSoldDate, skipping")
                    continue
                
                last_sold_price = sale_dict.get("lastSoldPrice", {})
                price_value = last_sold_price.get("value")
                if price_value is None:
                    logger.warning(f"Sale {sale.id} missing lastSoldPrice.value, skipping")
                    continue
                
                sales_data.append({
                    "sale_id": sale.id,
                    "sold_date": last_sold_date,
                    "price": float(price_value),
                    "currency": last_sold_price.get("currency", "USD"),
                    "condition": sale_dict.get("condition", "Unknown"),
                    "title": sale_dict.get("title", ""),
                    "seller_feedback_score": sale_dict.get("seller", 0),
                    "item_aspects": sale_dict.get("itemAspects", {})
                })
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing sale {sale.id}: {str(e)}")
                continue
        
        total = len(sales_data)
        logger.info(f"Returning {total} sales for watch_id: {watch_id}")
        
        return {
            "watch_id": watch_id,
            "sales": sales_data,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    
    except Exception as e:
        logger.error(f"Error in get_watch_sales for watch_id {watch_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/watches/{watch_id}/price-history", dependencies=[Depends(get_current_user)])
async def get_price_history(watch_id: str, start_date: str, end_date: str, aggregation: str = "daily"):
    """Fetch historical sold price data for a watch with time-based filtering and aggregation."""
    watch_ref = db.collection("watches").document(watch_id).get()
    if not watch_ref.exists:
        raise HTTPException(status_code=404, detail="Watch not found")
    
    try:
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        if start > end:
            raise ValueError("start_date must be before end_date")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format or range: {str(e)}")
    
    watch = watch_ref.to_dict()
    daily_history = watch.get("history", {}).get("daily", [])
    filtered_history = [entry for entry in daily_history if start_date <= entry["date"] <= end_date]
    
    if not filtered_history:
        return {"watch_id": watch_id, "aggregation": aggregation, "data": []}
    
    if aggregation == "daily":
        data = [{"date": entry["date"], "avg_price": float(entry["avg_price"]), "sales_qty": entry["sales_qty"]} 
                for entry in filtered_history]
    elif aggregation == "weekly":
        weekly_data = {}
        for entry in filtered_history:
            date = datetime.strptime(entry["date"], "%Y-%m-%d")
            week_start = (date - timedelta(days=date.weekday())).strftime("%Y-%m-%d")
            if week_start not in weekly_data:
                weekly_data[week_start] = {"prices": [], "sales": 0}
            weekly_data[week_start]["prices"].append(float(entry["avg_price"]) * entry["sales_qty"])
            weekly_data[week_start]["sales"] += entry["sales_qty"]
        data = [
            {
                "date": week,
                "avg_price": sum(d["prices"]) / d["sales"],
                "sales_qty": d["sales"]
            } for week, d in weekly_data.items()
        ]
        data.sort(key=lambda x: x["date"])
    elif aggregation == "monthly":
        monthly_data = {}
        for entry in filtered_history:
            month = entry["date"][:7]
            if month not in monthly_data:
                monthly_data[month] = {"prices": [], "sales": 0}
            monthly_data[month]["prices"].append(float(entry["avg_price"]) * entry["sales_qty"])
            monthly_data[month]["sales"] += entry["sales_qty"]
        data = [
            {
                "date": f"{month}-01",
                "avg_price": sum(d["prices"]) / d["sales"],
                "sales_qty": d["sales"]
            } for month, d in monthly_data.items()
        ]
        data.sort(key=lambda x: x["date"])
    else:
        raise HTTPException(status_code=400, detail="Invalid aggregation type. Use 'daily', 'weekly', or 'monthly'.")
    
    return {"watch_id": watch_id, "aggregation": aggregation, "data": data}

@app.get("/watches/{watch_id}/price-trend", dependencies=[Depends(get_current_user)])
async def get_price_trend(watch_id: str, period: str, end_date: str = None) -> Dict[str, List[Dict]]:
    """Fetch price trend data for a watch model."""
    valid_periods = {"1d": 1, "30d": 30, "90d": 90, "1y": 365}
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail="Invalid period. Use: 1d, 30d, 90d, 1y")
    
    if end_date:
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00")).replace(tzinfo=UTC)
    else:
        end = datetime.utcnow().replace(tzinfo=UTC)
    start = (end - timedelta(days=valid_periods[period])).replace(tzinfo=UTC)
    
    watch_ref = db.collection("watches").document(watch_id).get()
    if not watch_ref.exists:
        raise HTTPException(status_code=404, detail="Watch not found")
    watch = watch_ref.to_dict()
    
    is_hourly = period == "1d"
    history_key = "hourly" if is_hourly else "daily"
    history_data = watch.get("history", {}).get(history_key, [])
    
    if is_hourly and end.hour < 12:
        start = (start - timedelta(days=1)).replace(tzinfo=UTC)
    
    logger.info(f"Raw {history_key} data for {watch_id}: {history_data}")
    
    filtered_data = []
    for entry in history_data:
        key = "time" if is_hourly else "date"
        value = entry[key]
        if is_hourly:
            entry_dt = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=UTC)
            if start <= entry_dt <= end:
                filtered_data.append(entry)
        else:
            entry_dt = datetime.fromisoformat(value + "T00:00:00+00:00").replace(tzinfo=UTC)
            if start <= entry_dt <= end:
                filtered_data.append(entry)
    
    def get_timestamp(entry):
        key = "time" if is_hourly else "date"
        value = entry[key]
        return datetime.fromisoformat(value.replace("Z", "+00:00") if is_hourly else value + "T00:00:00+00:00").replace(tzinfo=UTC)
    
    filtered_data.sort(key=get_timestamp)
    
    result = [
        {
            "timestamp": entry["time" if is_hourly else "date"] if is_hourly else f"{entry['date']}T00:00:00Z",
            "price": float(entry["avg_price"]),
            "sales_count": entry["sales_qty"]
        }
        for entry in filtered_data
    ]
    
    logger.info(f"Filtered and sorted data: {result}")
    return {"data": result}

@app.get("/trending/sales", dependencies=[Depends(get_current_user)])
async def get_trending_sales(period: str, limit: int = 10):
    """Return precomputed trending watches by sales volume."""
    if period not in ["last_7_days", "last_30_days", "last_90_days"]:
        raise HTTPException(status_code=400, detail="Invalid period. Use 'last_7_days', 'last_30_days', or 'last_90_days'.")
    
    today = datetime.utcnow().strftime("%Y-%m-%d")
    trending_ref = db.collection("trending").document(today).get()
    if not trending_ref.exists:
        raise HTTPException(status_code=404, detail="Trending data not available for today. Please run /precompute-trending.")
    
    trending_data = trending_ref.to_dict()
    sales_list = trending_data["sales"].get(period, [])
    limited_list = sales_list[:min(limit, len(sales_list))]
    
    return {
        "trending_type": "sales",
        "period": period,
        "watches": limited_list
    }

@app.get("/trending/growth", dependencies=[Depends(get_current_user)])
async def get_trending_growth(period: str, limit: int = 10):
    """Return precomputed trending watches by price growth."""
    if period not in ["last_30_days", "last_90_days"]:
        raise HTTPException(status_code=400, detail="Invalid period. Use 'last_30_days' or 'last_90_days'.")
    
    today = datetime.utcnow().strftime("%Y-%m-%d")
    trending_ref = db.collection("trending").document(today).get()
    if not trending_ref.exists:
        raise HTTPException(status_code=404, detail="Trending data not available for today. Please run /precompute-trending.")
    
    trending_data = trending_ref.to_dict()
    growth_list = trending_data["growth"].get(period, [])
    limited_list = growth_list[:min(limit, len(growth_list))]
    
    return {
        "trending_type": "growth",
        "period": period,
        "watches": limited_list
    }

@app.get("/trending-all", dependencies=[Depends(get_current_user)])
async def get_trending(
    trend_types: Optional[str] = "sales,growth,models,brands",  # Comma-separated list
    periods: Optional[str] = "last_7_days,last_30_days,last_90_days",  # Comma-separated list
    limit: int = 10
):
    """
    Return precomputed trending data for sales, growth, models, and brands across specified periods.
    Use query params to filter trend types and periods.
    """
    # Validate trend types
    valid_trend_types = {"sales", "growth", "models", "brands"}
    requested_trend_types = set(trend_types.split(",")) if trend_types else valid_trend_types
    if not requested_trend_types.issubset(valid_trend_types):
        raise HTTPException(status_code=400, detail=f"Invalid trend_types. Use {', '.join(valid_trend_types)}.")

    # Validate periods
    valid_periods = {"last_7_days", "last_30_days", "last_90_days"}
    requested_periods = set(periods.split(",")) if periods else valid_periods
    if not requested_periods.issubset(valid_periods):
        raise HTTPException(status_code=400, detail=f"Invalid periods. Use {', '.join(valid_periods)}.")
    
    # Additional validation for growth (no 7-day period)
    if "growth" in requested_trend_types and "last_7_days" in requested_periods:
        requested_periods.discard("last_7_days")  # Silently ignore invalid period for growth

    # Fetch trending data
    today = datetime.utcnow().strftime("%Y-%m-%d")
    trending_ref = db.collection("trending").document(today).get()
    if not trending_ref.exists:
        raise HTTPException(status_code=404, detail="Trending data not available for today. Please run /precompute-trending.")
    
    trending_data = trending_ref.to_dict()
    
    # Build response
    response = {}
    for trend_type in requested_trend_types:
        response[trend_type] = {}
        for period in requested_periods:
            # Skip invalid combinations (e.g., growth for last_7_days)
            if trend_type == "growth" and period == "last_7_days":
                continue
            # Fetch data, assuming models and brands are stored similarly in the database
            trend_list = trending_data.get(trend_type, {}).get(period, [])
            limited_list = trend_list[:min(limit, len(trend_list))]
            response[trend_type][period] = limited_list
    
    return {
        "trending": response,
        "trend_types": list(requested_trend_types),
        "periods": list(requested_periods)
    }

@app.get("/watches", dependencies=[Depends(get_current_user)])
async def search_watches(brand: Optional[str] = None, model: Optional[str] = None, 
                        sort_by: str = "name", limit: int = 20, offset: int = 0):
    """Search watches by brand and/or model with sorting and pagination."""
    try:
        if brand:
            brand = brand.capitalize()
        if model:
            model = model.capitalize()

        query = db.collection("watches")
        watches = set()
        total = 0

        if brand:
            logger.info(f"Querying brand: {brand}")
            brand_query = query.where(filter=firestore.FieldFilter("brand", "==", brand))
            brand_results = brand_query.get()
            brand_watches = [
                {
                    "watch_id": doc.id,
                    "brand": doc.to_dict().get("brand", ""),
                    "model_name": doc.to_dict().get("model_name", ""),
                    "reference_number": doc.to_dict().get("reference_number", "")
                } for doc in brand_results
            ]
            logger.info(f"Brand query found {len(brand_watches)} watches")
            watches.update((w["watch_id"],) for w in brand_watches)

        if model:
            logger.info(f"Querying model: {model}")
            model_query = query.where(filter=firestore.FieldFilter("model_name", ">=", model)) \
                              .where(filter=firestore.FieldFilter("model_name", "<=", model + "\uf8ff"))
            model_results = model_query.get()
            model_watches = [
                {
                    "watch_id": doc.id,
                    "brand": doc.to_dict().get("brand", ""),
                    "model_name": doc.to_dict().get("model_name", ""),
                    "reference_number": doc.to_dict().get("reference_number", "")
                } for doc in model_results
            ]
            logger.info(f"Model query found {len(model_watches)} watches")
            watches.update((w["watch_id"],) for w in model_watches)

        final_watches = []
        if watches:
            for watch_id in watches:
                doc = db.collection("watches").document(watch_id[0]).get()
                if doc.exists:
                    final_watches.append({
                        "watch_id": doc.id,
                        "brand": doc.to_dict().get("brand", ""),
                        "model_name": doc.to_dict().get("model_name", ""),
                        "reference_number": doc.to_dict().get("reference_number", ""),
                        "canonical_name": doc.to_dict().get("canonical_name", ""),
                        "image_url": doc.to_dict().get("image_url", "")
                    })

        if sort_by == "name":
            final_watches.sort(key=lambda x: x["model_name"] or "")
        elif sort_by == "sales":
            final_watches.sort(key=lambda x: db.collection("watches").document(x["watch_id"]).get().to_dict().get("total_sales", 0), reverse=True)
        elif sort_by == "price":
            final_watches.sort(key=lambda x: db.collection("watches").document(x["watch_id"]).get().to_dict().get("last_avg_price", 0), reverse=True)
        else:
            raise HTTPException(status_code=400, detail="Invalid sort_by. Use 'name', 'sales', or 'price'.")

        total = len(final_watches)
        paginated_watches = final_watches[offset:offset + limit]

        logger.info(f"Final search for brand: {brand}, model: {model}, found {total} results, returning {len(paginated_watches)}")
        return {
            "watches": paginated_watches,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error in search_watches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/users/{user_id}/collection", response_model=dict)
async def add_to_collection(user_id: str, request: AddWatchRequest, current_user: str = Depends(get_current_user)):
    """Add a watch to a user's collection."""
    if user_id != current_user:
        logger.warning(f"Unauthorized attempt: user_id={user_id}, current_user={current_user}")
        raise HTTPException(status_code=403, detail="You can only modify your own collection")

    watch_id = request.watch_id
    watch_ref = db.collection("watches").document(watch_id)
    if not watch_ref.get().exists:
        logger.info(f"Watch not found: {watch_id}")
        raise HTTPException(status_code=404, detail="Watch not found")

    collection_ref = db.collection("users").document(user_id).collection("collection").document(watch_id)
    if collection_ref.get().exists:
        logger.info(f"Watch already in collection: {watch_id} for user {user_id}")
        raise HTTPException(status_code=409, detail="Watch already in collection")

    try:
        collection_data = {
            "watch_id": watch_id,
            "added_at": datetime.utcnow().isoformat(),
        }
        if request.purchase_price is not None:
            collection_data["purchase_price"] = request.purchase_price
        collection_ref.set(collection_data)
        logger.info(f"Added watch {watch_id} to user {user_id}'s collection")
        return {"status": "Watch added", "watch_id": watch_id}
    except Exception as e:
        logger.error(f"Failed to add watch {watch_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/users/{user_id}/collection/{watch_id}", response_model=dict)
async def remove_from_collection(user_id: str, watch_id: str, current_user: str = Depends(get_current_user)):
    """Remove a watch from a user's collection."""
    if user_id != current_user:
        logger.warning(f"Unauthorized attempt: user_id={user_id}, current_user={current_user}")
        raise HTTPException(status_code=403, detail="You can only modify your own collection")

    collection_ref = db.collection("users").document(user_id).collection("collection").document(watch_id)
    if not collection_ref.get().exists:
        logger.info(f"Watch not in collection: {watch_id} for user {user_id}")
        raise HTTPException(status_code=404, detail="Watch not in collection")

    try:
        collection_ref.delete()
        logger.info(f"Removed watch {watch_id} from user {user_id}'s collection")
        return {"status": "Watch removed", "watch_id": watch_id}
    except Exception as e:
        logger.error(f"Failed to remove watch {watch_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/users/{user_id}/collection", dependencies=[Depends(get_current_user)])
async def get_user_collection(user_id: str):
    """Get a user's watch collection with current values."""
    collection_ref = db.collection("users").document(user_id).collection("collection").stream()
    watches = []
    total_value = 0.0
    
    for doc in collection_ref:
        watch_id = doc.to_dict()["watch_id"]
        watch_data = db.collection("watches").document(watch_id).get().to_dict()
        if watch_data:
            daily_history = sorted(watch_data.get("history", {}).get("daily", []), key=lambda x: x["date"], reverse=True)
            last_price = float(daily_history[0]["avg_price"]) if daily_history else 0.0
            total_value += last_price
            watches.append({
                "watch_id": watch_id,
                "brand": watch_data.get("brand", ""),
                "model_name": watch_data.get("model_name", ""),
                "canonical_name": watch_data.get("canonical_name", ""),
                "reference_number": watch_data.get("reference_number", ""),
                "last_known_price": last_price,
                "image_url": watch_data.get("image_url", ""),
                "added_at": doc.to_dict()["added_at"],
            })
    
    return {
        "watches": watches,
        "total_value": total_value,
    }

@app.get("/users/{user_id}/collection/history", dependencies=[Depends(get_current_user)])
async def get_collection_history(user_id: str, period: str = "30d"):
    """Get historical value of a user's watch collection."""
    valid_periods = {"1d": 1, "30d": 30, "90d": 90, "1y": 365}
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail="Invalid period. Use: 1d, 30d, 90d, 1y")
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=valid_periods[period])
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    collection_ref = db.collection("users").document(user_id).collection("collection").stream()
    watch_ids = [doc.to_dict()["watch_id"] for doc in collection_ref]
    if not watch_ids:
        return {"history": []}
    
    history_dict = {}
    for watch_id in watch_ids:
        watch_data = db.collection("watches").document(watch_id).get().to_dict()
        if not watch_data or "history" not in watch_data:
            continue
        daily_history = watch_data["history"].get("daily", [])
        for entry in daily_history:
            date = entry["date"]
            if start_str <= date <= end_str:
                if date not in history_dict:
                    history_dict[date] = {"total_value": 0.0}
                history_dict[date]["total_value"] += float(entry["avg_price"])
    
    history = [
        {"date": date, "total_value": data["total_value"]}
        for date, data in sorted(history_dict.items())
    ]
    
    return {"history": history}

@app.get("/pulse/content", response_model=dict)
async def get_pulse_content(filter: Optional[str] = "all", limit: int = 20, offset: int = 0):
    """Fetch curated watch-related articles and posts with sentiment."""
    try:
        query = db.collection("content").order_by("published", direction=firestore.Query.DESCENDING)
        if filter != "all":
            query = query.where("tags", "array_contains", filter.lower())
        items = query.offset(offset).limit(limit).get()
        content = [item.to_dict() for item in items]
        logger.info(f"Fetched {len(content)} pulse content items with filter: {filter}")
        return {"articles": content}
    except Exception as e:
        logger.error(f"Error fetching pulse content: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/pulse/trends", response_model=dict)
async def get_pulse_trends(limit: int = 10):
    """Fetch trending brands and models with sentiment and mentions."""
    try:
        trends = db.collection("trends").order_by("mentions", direction=firestore.Query.DESCENDING).limit(limit).get()
        trend_data = [trend.to_dict() for trend in trends]
        logger.info(f"Fetched {len(trend_data)} pulse trends")
        return {"trends": trend_data}
    except Exception as e:
        logger.error(f"Error fetching pulse trends: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/trending", response_model=List[Union[BrandTrend, ModelTrend]])
async def get_trending(
     limit: int = 10,
     sort_by: str = "trend_score",  # Options: mentions, sales_count, price_change_percent, trend_score
     trend_type: Optional[str] = None,  # Options: brand, model, None (both)
     min_sentiment: Optional[float] = None,  # Filter by minimum sentiment score
     _=Depends(get_current_user)
):
    """Fetch enriched trending brands and models with balanced representation and enhanced insights."""
    try:
        # Ensure limit is even for balanced output
        limit_per_type = limit // 2 if trend_type is None else limit

        # Initialize result lists
        enriched_trends = []

        # Define sorting field mapping
        sort_field_map = {
            "mentions": "mentions",
            "sales_count": "sales_count",
            "price_change_percent": "price_change_percent"
        }
        sort_field = sort_field_map.get(sort_by, "mentions")

        # Helper function to calculate trend score and summary
        def compute_trend_metrics(trend_data: dict) -> dict:
            mentions = trend_data.get("mentions", 0)
            sales_count = trend_data.get("sales_count", 0)
            price_change = abs(trend_data.get("price_change_percent", 0))
            # Weighted score: 40% mentions, 40% sales, 20% price change
            trend_score = (0.4 * mentions / max(1, mentions)) + (0.4 * sales_count / max(1, sales_count)) + (0.2 * price_change / 100)
            # Generate summary based on dominant factor
            if mentions > sales_count and mentions > price_change:
                trend_summary = f"Surging with {mentions} mentions in recent articles"
            elif sales_count > mentions and sales_count > price_change:
                trend_summary = f"High demand with {sales_count} sales this week"
            else:
                trend_summary = f"Price { 'up' if price_change > 0 else 'down' } {price_change:.1f}% in recent markets"
            trend_data["trend_score"] = trend_score
            trend_data["trend_summary"] = trend_summary
            return trend_data

        # Fetch brands
        if trend_type is None or trend_type == "brand":
            brand_query = db.collection("trending_brands").order_by(sort_field, direction=firestore.Query.DESCENDING).limit(limit_per_type)
            if min_sentiment is not None:
                brand_query = brand_query.where("sentiment_score", ">=", min_sentiment)
            brand_docs = brand_query.get()
            for doc in brand_docs:
                trend_data = doc.to_dict()
                trend_data["type"] = "brand"
                trend_data = compute_trend_metrics(trend_data)
                enriched_trends.append(BrandTrend(**trend_data))

        # Fetch models
        if trend_type is None or trend_type == "model":
            model_query = db.collection("trending_models").order_by(sort_field, direction=firestore.Query.DESCENDING).limit(limit_per_type)
            if min_sentiment is not None:
                model_query = model_query.where("sentiment_score", ">=", min_sentiment)
            model_docs = model_query.get()
            for doc in model_docs:
                trend_data = doc.to_dict()
                trend_data["type"] = "model"
                trend_data = compute_trend_metrics(trend_data)
                enriched_trends.append(ModelTrend(**trend_data))

        # Sort combined trends by sort_field and trim to limit
        enriched_trends.sort(key=lambda x: getattr(x, sort_field, 0), reverse=True)
        enriched_trends = enriched_trends[:limit]

        logger.info(f"Fetched {len(enriched_trends)} enriched trends (brands: {sum(1 for t in enriched_trends if t.type == 'brand')}, models: {sum(1 for t in enriched_trends if t.type == 'model')})")
        return enriched_trends
    except Exception as e:
        logger.error(f"Error fetching trending data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/brands/{brand}")
async def get_brand_details(brand: str, _=Depends(get_current_user)):
    """Fetch detailed information for a brand, including articles, watches, and sales."""
    try:
        articles = db.collection("content").where("brand", "==", brand).order_by("published", direction=firestore.Query.DESCENDING).limit(20).get()
        articles_data = [article.to_dict() for article in articles]
        
        watches = db.collection("watches").where("brand", "==", brand).limit(10).get()
        watches_data = [watch.to_dict() for watch in watches]
        
        sales_data = []
        start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
        for watch in watches_data:
            sales = db.collection("sales").where("sku", "==", watch["sku"]).where("lastSoldDate", ">=", start_date).limit(5).get()
            for sale in sales:
                sale_dict = sale.to_dict()
                sales_data.append({
                    "sale_id": sale.id,
                    "sold_date": sale_dict.get("lastSoldDate"),
                    "price": float(sale_dict.get("lastSoldPrice", {}).get("value", 0)),
                    "currency": sale_dict.get("lastSoldPrice", {}).get("currency", "USD"),
                    "condition": sale_dict.get("condition", "Unknown")
                })

        logger.info(f"Fetched details for brand {brand}: {len(articles_data)} articles, {len(watches_data)} watches, {len(sales_data)} sales")
        return {
            "brand": brand,
            "articles": articles_data,
            "watches": watches_data,
            "sales": sales_data,
            "logo_url": None
        }
    except Exception as e:
        logger.error(f"Error fetching brand details for {brand}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/models/{model}")
async def get_model_details(model: str, _=Depends(get_current_user)):
    """Fetch detailed information for a model, including articles, watches, and sales."""
    try:
        articles = db.collection("content").where("model", "==", model).order_by("published", direction=firestore.Query.DESCENDING).limit(20).get()
        articles_data = [article.to_dict() for article in articles]
        
        watches_data = []
        watches_ref = db.collection("watches").stream()
        for watch in watches_ref:
            watch_dict = watch.to_dict()
            if model.lower() in watch_dict.get("model_name", "").lower():
                watches_data.append(watch_dict)
        watches_data = watches_data[:10]
        
        sales_data = []
        start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
        for watch in watches_data:
            sales = db.collection("sales").where("sku", "==", watch["sku"]).where("lastSoldDate", ">=", start_date).limit(5).get()
            for sale in sales:
                sale_dict = sale.to_dict()
                sales_data.append({
                    "sale_id": sale.id,
                    "sold_date": sale_dict.get("lastSoldDate"),
                    "price": float(sale_dict.get("lastSoldPrice", {}).get("value", 0)),
                    "currency": sale_dict.get("lastSoldPrice", {}).get("currency", "USD"),
                    "condition": sale_dict.get("condition", "Unknown")
                })

        top_brands = list(set(watch.get("brand", "") for watch in watches_data if watch.get("brand")))[:3]
        
        logger.info(f"Fetched details for model {model}: {len(articles_data)} articles, {len(watches_data)} watches, {len(sales_data)} sales")
        return {
            "model": model,
            "articles": articles_data,
            "watches": watches_data,
            "sales": sales_data,
            "top_brands": top_brands
        }
    except Exception as e:
        logger.error(f"Error fetching model details for {model}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")