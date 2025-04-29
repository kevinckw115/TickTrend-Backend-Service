from bs4 import BeautifulSoup
import requests
import re
from rapidfuzz import fuzz, process
from datetime import datetime, timedelta
import statistics
import logging
from requests.exceptions import HTTPError
from typing import List, Dict, Optional, Set, Tuple
from statistics import mean, median
from pytz import UTC
from textblob import TextBlob
from feedparser import parse
import hashlib
from urllib.parse import urljoin
import html
from google.cloud import secretmanager
from google.cloud import firestore
from google.api_core import exceptions, retry
import time
from fastapi import HTTPException

# Logging setup
logger = logging.getLogger(__name__)

def get_secret(secret_name: str, project_id: str) -> str:
    """Retrieve a secret from Google Cloud Secret Manager."""
    try:
        secret_client = secretmanager.SecretManagerServiceClient()
        secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = secret_client.access_secret_version(request={"name": secret_path})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Failed to retrieve secret {secret_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve secret: {secret_name}")

def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities from text, preserving content."""
    if not text:
        return ""
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ").strip()
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text.strip()

def extract_image(soup: Optional[BeautifulSoup], entry: dict, source_type: str, article_url: str) -> Optional[str]:
    """Extract the header or featured image URL from an article or RSS entry."""
    try:
        if source_type == "article":
            logger.debug(f"Extracting image for article: {article_url}")
            og_image = soup.select_one('meta[property="og:image"]')
            if og_image and og_image.get("content"):
                logger.debug(f"Found og:image: {og_image['content']}")
                return og_image["content"]
            img_selectors = [
                'img.wp-post-image',
                'img.featured-image',
                'img.attachment-post-thumbnail',
                'img.hero-image',
                '.article-header img',
                '.post-thumbnail img',
                '.entry-content img:first-of-type',
                'img[alt*="featured"]',
                '.featured-media img',
                '.post-image img'
            ]
            for selector in img_selectors:
                img = soup.select_one(selector)
                if img and img.get("src"):
                    img_url = img["src"]
                    if not img_url.startswith("http"):
                        img_url = urljoin(article_url, img_url)
                    logger.debug(f"Found image with selector {selector}: {img_url}")
                    return img_url
            content_img = soup.select_one('.article-body img, .post-content img, .entry-content img')
            if content_img and content_img.get("src"):
                img_url = content_img["src"]
                if not img_url.startswith("http"):
                    img_url = urljoin(article_url, img_url)
                logger.debug(f"Found fallback content image: {img_url}")
                return img_url
            logger.debug("No image found in article")
        elif source_type == "rss":
            logger.debug(f"Extracting image for RSS entry: {article_url}")
            if "enclosures" in entry:
                for enc in entry["enclosures"]:
                    if enc.get("type", "").startswith("image/"):
                        logger.debug(f"Found enclosure image: {enc.get('url')}")
                        return enc.get("url")
            if "media_content" in entry:
                for media in entry["media_content"]:
                    if media.get("medium") == "image":
                        logger.debug(f"Found media:content image: {media.get('url')}")
                        return media.get("url")
            content = entry.get("content", [{}])[0].get("value", "") or entry.get("summary", "")
            if content:
                soup = BeautifulSoup(content, "html.parser")
                img = soup.select_one('img.wp-post-image, img.attachment-post-thumbnail, img.featured-image')
                if img and img.get("src"):
                    img_url = img["src"]
                    if not img_url.startswith("http"):
                        img_url = urljoin(article_url, img_url)
                    logger.debug(f"Found embedded image with class: {img_url}")
                    return img_url
                img = soup.find("img")
                if img and img.get("src"):
                    img_url = img["src"]
                    if not img_url.startswith("http"):
                        img_url = urljoin(article_url, img_url)
                    logger.debug(f"Found first embedded image: {img_url}")
                    return img_url
            logger.debug("No image found in RSS content")
            try:
                headers = {"User-Agent": "TickTrend/1.0 (+https://ticktrend.app)"}
                response = requests.get(article_url, headers=headers, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                og_image = soup.select_one('meta[property="og:image"]')
                if og_image and og_image.get("content"):
                    logger.debug(f"Found og:image in article fetch: {og_image['content']}")
                    return og_image["content"]
                img = soup.select_one('img.wp-post-image, img.attachment-post-thumbnail, .entry-content img')
                if img and img.get("src"):
                    img_url = img["src"]
                    if not img_url.startswith("http"):
                        img_url = urljoin(article_url, img_url)
                    logger.debug(f"Found image in article fetch: {img_url}")
                    return img_url
            except requests.RequestException as e:
                logger.debug(f"Failed to fetch article for image: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Failed to extract image for {article_url}: {str(e)}")
        return None

def refresh_ebay_token(ebay_auth_encoded: str, ebay_auth_url: str) -> str:
    """Refreshes eBay API token for authenticated requests."""
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {ebay_auth_encoded}"}
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    try:
        resp = requests.post(ebay_auth_url, headers=headers, data=data)
        resp.raise_for_status()
        token = resp.json()["access_token"]
        logger.info("Successfully refreshed eBay API token")
        return token
    except requests.RequestException as e:
        logger.error(f"Failed to refresh eBay token: {e}")
        raise HTTPException(status_code=500, detail="Unable to refresh eBay token")

def load_dynamic_criteria(db: firestore.Client) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, float]]:
    """Loads dynamic terms, regex patterns, and weights from Firestore."""
    terms_ref = db.collection("search_criteria").document("term_sets").get()
    if not terms_ref.exists:
        default_terms = {
            "VALID_BRANDS": {"rolex", "patek philippe", "audemars piguet", "omega", "cartier", "citizen", "tag heuer", 
                             "casio", "seiko", "timex", "swatch", "invicta", "fossil", "vacheron constantin", 
                             "jaeger-lecoultre", "iwc", "blancpain", "breguet", "hublot", "zenith", "longines", 
                             "breitling", "chopard", "ulysse nardin", "glashutte original", "a. lange & sohne", 
                             "piaget", "montblanc", "baume & mercier", "h. moser & cie", "roger dubuis", 
                             "parmigiani fleurier", "bovet", "raymond weil", "nomos", "junghans", "sinn", 
                             "stowa", "laco", "muhle-glashutte", "damasko", "grand seiko", "orient", "minolta", 
                             "q&q", "bell & ross", "lip", "yema", "michel herbelin", "panerai", "bulgari", 
                             "salvatore ferragamo", "bremont", "arnold & son", "christopher ward", "smiths", 
                             "hamilton", "bulova", "shinola", "weiss", "rgm", "ball watch co.", "anordain", 
                             "baltic", "farer", "halios", "lorier", "monta", "oak & oscar", "steinhart", 
                             "straton", "vaer", "zelos", "doxa", "squale", "gucci", "louis vuitton", 
                             "michael kors", "hugo boss", "armani", "versace", "chanel", "dior", "hermes", 
                             "tiffany & co.", "van cleef & arpels", "apple", "garmin", "samsung", "fitbit", 
                             "suunto", "tissot", "rado", "movado", "ebel", "frederique constant", "alpina", 
                             "glycine", "oris", "maurice lacroix", "perrelet", "corum"},
            "WATCH_TERMS": {"watch", "timepiece", "chronograph", "wristwatch"},
            "ACCESSORY_TERMS": {"watchband", "strap", "bezel", "case only"},
            "FAKE_TERMS": {"replica", "fake", "copy"},
            "BAD_CONDITIONS": {"parts only", "not working", "broken"},
            "FEATURE_TERMS": {"movement", "dial", "water resistant"},
            "MODEL_TERMS": {"submariner", "datejust", "daytona"},
            "AUTHORITY_TERMS": {"sotheby", "christie", "auction"},
            "LUXURY_BRANDS": {"rolex", "patek philippe", "audemars piguet"}
        }
        db.collection("search_criteria").document("term_sets").set(default_terms)
        logger.info("Initialized default term sets in Firestore")
        term_sets = {k: set(v) for k, v in default_terms.items()}
    else:
        term_sets = {k: set(v) for k, v in terms_ref.to_dict().items()}

    regex_ref = db.collection("search_criteria").document("regex_patterns").get()
    if not regex_ref.exists:
        default_regex = {
            "brand": r"|".join(f"\\b{re.escape(b)}\\b" for b in term_sets["VALID_BRANDS"]),
            "reference_number": r"(?:\bref\.|\breference\b|\bmodel\b)?\s*([a-z]{0,2}-?[0-9]{3,6}[a-z]{0,2}[-/]?[0-9a-z]*)",
            "model_name": r"(?i)\b(submariner|daytona|datejust|gmt-master|explorer|sea-dweller|yacht-master|milgauss|air-king|cellini|oyster perpetual|sky-dweller|nautilus|aquanaut|calatrava|grand complications|twenty~4|perpetual calendar|world time|royal oak|royal oak offshore|code 11\.59|millenary|jules audemars|speedmaster|seamaster|constellation|de ville|planet ocean|aqua terra|moonwatch|tank|santos|ballon bleu|pasha|drive|clé|ronde|overseas|patrimony|traditionnelle|historiques|malte|quai de l’île|reverso|master|polaris|rendez-vous|duometre|geophysic|portugieser|pilot|aquatimer|ingenieur|da vinci|big pilot|portofino|fifty fathoms|villeret|le brassus|l-evolution|classique|marine|heritage|type xx|reine de naples|big bang|classic fusion|spirit of big bang|king power|el primero|chronomaster|defy|pilot|elite|lange 1|saxonia|zeitwerk|datograph|odysseus|1815|polo|altiplano|limelight|possession|1858|star|heritage|summit|timewalker|clifton|riviera|hampton|capeland|endeavour|pioneer|streamliner|heritage|excalibur|velvet|hommage|kingsquare|tonda|kalpa|bugatti|toric|fleurier|récital|amadeo|virtuoso|luminor|radiomir|submersible|mare nostrum|octo|serpenti|lvcea|diagono|admiral|bubble|golden bridge|heritage|carrera|monaco|aquaracer|formula 1|link|autavia|master collection|hydroconquest|conquest|heritage|spirit|legend diver|navitimer|superocean|chronomat|avenger|premier|colt|alpine eagle|mille miglia|happy sport|l\.u\.c|marine|diver|freak|blast|executive|seaq|sixties|seventies|panomatic|senator|freelancer|maestro|tango|parsifal|le locle|prx|seastar|gentleman|visodate|t-touch|captain cook|hyperchrome|true|diamaster|museum|bold|series 800|edge|1911|sport classic|discovery|wave|classics|manufacture|highlife|slimline|alpiner|seastrong|startimer|comtesse|combat|airman|golden eye|incursore|aquis|big crown|divers sixty-five|propilot|artix|aikon|eliros|masterpiece|pontos|turbine|first class|skeleton|chronograph|eco-drive|promaster|chandler|brycen|tsuyosa|g-shock|edifice|pro trek|oceanus|casiotone|prospex|presage|astron|grand seiko|turtle|monster|samurai|alpinist|cocktail|weekender|waterbury|marlin|q timex|expedition|sistem51|big bold|skin|irony|moonswatch|pro diver|speedway|angel|bolt|townsman|neutra|sport|defender|g-timeless|grip|dive|eryx|tambour|escale|voyager|lexington|bradshaw|runway|portia|hero|pilot edition|ocean edition|palazzo|greca|v-race|j12|première|boy\.friend|code coco|grand bal|chiffre rouge|la d de dior|arceau|cape cod|h08|slim d’hermès|atlas|east west|tiffany 1837|pierre arpels|poetic complications|tangente|metro|club|orion|ludwig|max bill|meister|form|chronoscope|556|104|356|ezm|u1|flieger|marine|antea|partitio|aachen|augsburg|erbstück|squad|sea-timer|terrasport|29er|da36|dc56|dk30|snowflake|white birch|spring drive|bambino|mako|kamasu|star|br 03|br 05|br v2|br-x1|himalaya|nautic-ski|churchill|superman|rallygraf|speedgraf|newport|antarès|cap camarat|mb|alt1|supermarine|solo|hm|nebula|perpetual moon|c60|c65|trident|malvern|everest|commando|prs-29|khaki field|jazzmaster|ventura|broadway|lunar pilot|precisionist|computron|accutron|runwell|canfield|detrola|monster|standard issue|field watch|801|caliber 20|pennsylvania|engineer|fireman|roadmaster|model 1|model 2|world timer|chrono|aqua|seaforth|puck|fairwind|neptune|falcon|hydra|noble|atlas|triumph|olmsted|sandford|humboldt|ocean|nav b-uhr|triton|syncro|curve|daily driver|d5|c5|a5|swordfish|mako|hammerhead|sub 300|sub 600|sub 1500t|1521|20 atmos|tiger|watch|series|ultra|se|fenix|forerunner|vivoactive|instinct|galaxy watch|gear|versa|sense|charge|9 peak|spartan|traverse|classic|evo2|smilesolar)\b",
            "year": r"(?:\byear\b|\bmanufactured\b|\bfrom\b)?\s*(20[0-2][0-9]|19[0-9]{2})",
            "dial_color": r"\b(black|white|blue|green|silver|gold|red|grey|brown|purple)\b\s*(dial)?",
            "case_material": r"\b(stainless steel|gold|rose gold|yellow gold|white gold|titanium|ceramic|platinum|bronze)\b",
            "band_type": r"\b(oyster|jubilee|president|leather|rubber|silicone|nylon|mesh|bracelet|strap)\s*(band|strap)?\b",
            "bezel_type": r"\b(ceramic|steel|gold|tachymeter|rotating|fixed|engraved)\b\s*(bezel)?",
            "crystal": r"\b(sapphire|mineral|acrylic|crystal)\b",
            "movement": r"\b(automatic|quartz|manual|eco-drive|kinetic|solar)\b",
            "case_size": r"\b([2-5][0-9]mm)\b",
            "water_resistance": r"\b([0-9]{1,4}m|water resistant)\b",
            "condition": r"\b(new|used|pre-owned|vintage|mint|excellent|good|fair|poor)\b",
            "complications": r"\b(date|chronograph|gmt|moonphase|day-date|power reserve)\b",
            "gender": r"\b(men\'s|women\'s|unisex)\b"
        }
        db.collection("search_criteria").document("regex_patterns").set(default_regex)
        logger.info("Initialized default regex patterns in Firestore")
        regex_patterns = default_regex
    else:
        regex_patterns = regex_ref.to_dict()

    weights_ref = db.collection("search_criteria").document("scoring_weights").get()
    if not weights_ref.exists:
        default_weights = {
            "match_brand": 0.30, "match_reference_number": 0.40, "match_model_name": 0.20, "match_year": 0.10,
            "match_spec": 0.05, "match_alias_boost": 15.0, "match_threshold": 65.0,
            "quality_new": 50.0, "quality_new_defects": 40.0, "quality_used_good": 40.0, "quality_used_fair": 30.0,
            "quality_poor": 10.0, "quality_unspecified": 20.0, "quality_no_fake": 30.0, "quality_features": 20.0,
            "quality_desc_rich": 30.0, "quality_desc_moderate": 15.0, "quality_authority": 20.0,
            "quality_price_in_range": 50.0, "quality_price_penalty_max": 30.0, "quality_seller_high": 10.0,
            "quality_seller_low": -10.0, "quality_min_luxury_new": 100.0, "quality_min_standard": 80.0,
            "quality_vintage_adjust": -20.0, "watch_title_terms": 40.0, "watch_desc_terms": 30.0,
            "watch_valid_brand": 20.0, "watch_features": 20.0, "watch_ref_number": 30.0,
            "watch_accessory_penalty": -20.0, "watch_min_score": 20.0
        }
        db.collection("search_criteria").document("scoring_weights").set(default_weights)
        logger.info("Initialized default scoring weights in Firestore")
        weights = default_weights
    else:
        weights = weights_ref.to_dict()

    return term_sets, regex_patterns, weights

def load_brand_prefixes(term_sets: Dict[str, Set[str]], db: firestore.Client) -> Dict[str, str]:
    """Loads brand prefixes, syncing with VALID_BRANDS."""
    prefixes_ref = db.collection("search_criteria").document("brand_prefixes").get()
    valid_brands = term_sets["VALID_BRANDS"]
    
    if not prefixes_ref.exists:
        prefixes = {}
        used_prefixes = set()
        for brand in sorted(valid_brands):
            base = brand[:3].upper()
            prefix = base
            counter = 1
            while prefix in used_prefixes:
                words = brand.split()
                if len(words) > 1 and counter == 1:
                    prefix = (words[0][0] + words[-1][:2]).upper()
                else:
                    prefix = f"{base[:2]}{counter:01d}"
                counter += 1
            prefixes[brand] = prefix
            used_prefixes.add(prefix)
        db.collection("search_criteria").document("brand_prefixes").set({
            "prefixes": prefixes,
            "last_updated": datetime.utcnow().isoformat()
        })
        return prefixes
    else:
        data = prefixes_ref.to_dict()
        prefixes = data["prefixes"]
        missing_brands = valid_brands - set(prefixes.keys())
        if missing_brands:
            used_prefixes = set(prefixes.values())
            for brand in missing_brands:
                base = brand[:3].upper()
                prefix = base
                counter = 1
                while prefix in used_prefixes:
                    words = brand.split()
                    if len(words) > 1 and counter == 1:
                        prefix = (words[0][0] + words[-1][:2]).upper()
                    else:
                        prefix = f"{base[:2]}{counter:01d}"
                    counter += 1
                prefixes[brand] = prefix
                used_prefixes.add(prefix)
            db.collection("search_criteria").document("brand_prefixes").set({
                "prefixes": prefixes,
                "last_updated": datetime.utcnow().isoformat()
            })
        return prefixes

def fetch_ebay_sales(ebay_api_token: str, ebay_api_url: str, category_id: str = "281", limit: int = 100, 
                    start_time: Optional[str] = None, end_time: Optional[str] = None, 
                    max_sales: Optional[int] = 50) -> List[Dict]:
    """Fetches recent eBay sales with retry logic for rate limits."""
    headers = {"Authorization": f"Bearer {ebay_api_token}"}
    all_sales = []
    offset = 0
    retries = 3
    while True:
        if len(all_sales) >= max_sales:
            break
        params = {
            "q": "watch",
            "category_ids": category_id,
            "limit": str(limit),
            "offset": str(offset),
            "filter": f"soldItemsOnly:true,lastSoldDate:[{start_time or '1970-01-01T00:00:00Z'}..{end_time or ''}]"
        }
        for attempt in range(retries):
            try:
                response = requests.get(ebay_api_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                items = data.get("itemSummaries", [])
                if not items:
                    break
                batch = []
                for item in items[:min(5, max_sales - len(all_sales))]:
                    detail_url = f"https://api.ebay.com/buy/browse/v1/item/{item['itemId']}"
                    detail_resp = requests.get(detail_url, headers=headers)
                    detail_resp.raise_for_status()
                    detail = detail_resp.json()
                    image_url = detail.get("image", {}).get("imageUrl", None)
                    if not image_url and "image" in item:
                        image_url = item["image"].get("imageUrl", None)
                    batch.append({
                        "itemId": item["itemId"],
                        "title": item["title"],
                        "lastSoldPrice": {"value": item["price"]["value"], "currency": item["price"]["currency"]},
                        "lastSoldDate": item.get("itemEndDate", datetime.utcnow().isoformat()),
                        "condition": detail.get("condition", "Unknown"),
                        "itemAspects": detail.get("itemAspects", {}),
                        "description": detail.get("description", "No description available"),
                        "seller": detail.get("seller", {}).get("feedbackScore", 0),
                        "image_url": image_url
                    })
                all_sales.extend(batch)
                offset += limit
                total = int(data.get("total", 0))
                if offset >= total or len(all_sales) >= max_sales:
                    break
                time.sleep(0.1)
                break
            except HTTPError as e:
                if e.response.status_code == 429 and attempt < retries - 1:
                    sleep_time = 2 ** attempt
                    logger.warning(f"Rate limited, retrying in {sleep_time}s")
                    time.sleep(sleep_time)
                    continue
                logger.error(f"eBay API error: {e}")
                return all_sales
    return all_sales

def generate_sku(brand: str, model_code: str, existing_skus: Set[str], brand_prefixes: Dict[str, str]) -> str:
    """Generates a unique SKU using brand prefix and model code."""
    brand_lower = brand.lower()
    prefix = brand_prefixes.get(brand_lower, brand[:3].upper())
    base_sku = f"{prefix}-{model_code.replace('/', '-').replace(' ', '').upper()}"
    variant = 1
    sku = f"{base_sku}-{variant:02d}"
    while sku in existing_skus:
        variant += 1
        sku = f"{base_sku}-{variant:02d}"
    return sku

def identify_watch_model(sale: Dict, watch_models: Dict[str, Dict], term_sets: Dict[str, Set[str]], 
                        brand_prefixes: Dict[str, str], regex_patterns: Dict[str, str], weights: Dict[str, float], 
                        db: firestore.Client) -> str:
    """Identifies or creates a watch model based on sale data."""
    title = sale.get("title", "").lower()
    aspects = sale.get("itemAspects", {})
    description = sale.get("description", "").lower()
    full_text = f"{title} {description}".strip()

    def extract_feature(pattern: str, text: str, aspect_key: str, default: str = None) -> tuple[str, float]:
        aspect_value = aspects.get(aspect_key, "").strip().lower()
        if aspect_value:
            return aspect_value, 0.95
        matches = re.findall(pattern, text, re.I)
        if matches:
            value = matches[0][0] if isinstance(matches[0], tuple) else matches[0]
            return value, 0.60
        return default, 0.20

    brand, brand_conf = extract_feature(regex_patterns["brand"], full_text, "Brand", "unknown")
    brand = brand.title()
    reference_number, ref_conf = extract_feature(regex_patterns["reference_number"], full_text, "Reference Number", "")
    model_name, model_conf = extract_feature(regex_patterns["model_name"], title, "Model", "")
    if not model_name and not reference_number:
        tokens = title.split()
        model_name = tokens[1] if len(tokens) > 1 else "unknown"
    model_name = model_name.title()
    year, year_conf = extract_feature(regex_patterns["year"], full_text, "Year of Manufacture", "")

    if not brand or brand.lower() == "unknown" or \
       not model_name or model_name.lower() == "unknown" or \
       not reference_number:
        logger.debug(f"Sale {sale.get('itemId', 'unknown')} missing valid brand, model, or reference. Returning 'Unidentified'")
        return "Unidentified"

    attributes = {}
    model_fields = [
        "dial_color", "case_material", "movement", "band_type", "bezel_type", "crystal", 
        "case_size", "water_resistance", "complications"
    ]
    for field in model_fields:
        if field in regex_patterns:
            value, conf = extract_feature(regex_patterns[field], full_text, field.capitalize())
            if value:
                attributes[field] = (value, conf)

    if "movement" in attributes:
        attributes["movement_type"] = attributes.pop("movement")
    if "band_type" in attributes:
        attributes["band"] = attributes.pop("band_type")

    canonical_parts = [brand, model_name]
    if reference_number:
        canonical_parts.append(reference_number)
    
    feature_priority = [
        "movement_type", "complications", "dial_color", "band", 
        "case_size", "case_material", "bezel_type"
    ]
    feature = None
    for field in feature_priority:
        if field in attributes:
            feature = attributes[field][0].capitalize()
            break
    if not feature:
        feature = "Watch"
    canonical_parts.append(feature)
    
    canonical_name = " ".join(p.strip() for p in canonical_parts if p).strip()
    canonical_conf = 0.80

    candidate = {
        "brand": (brand, brand_conf),
        "reference_number": (reference_number, ref_conf),
        "model_name": (model_name, model_conf),
        "year": (year, year_conf),
        "canonical_name": (canonical_name, canonical_conf),
        **attributes
    }

    def score_match(existing: Dict, candidate: Dict) -> float:
        score = 0.0
        max_score = 100.0

        core_weights = {
            "brand": weights["match_brand"],
            "reference_number": weights["match_reference_number"],
            "model_name": weights["match_model_name"],
            "year": weights["match_year"],
            "canonical_name": weights.get("match_canonical_name", 0.8)
        }
        total_weight = sum(core_weights.values())

        for field, weight in core_weights.items():
            cand_value, cand_conf = candidate.get(field, ("", 0.20))
            existing_value = existing.get(field, "").lower() if existing.get(field) else ""
            if not cand_value and not existing_value:
                score += (weight / total_weight) * 100 * cand_conf
            elif cand_value and existing_value:
                if field in {"brand", "reference_number", "year", "canonical_name"}:
                    if cand_value.lower() == existing_value:
                        score += (weight / total_weight) * 100 * cand_conf
                elif field == "model_name":
                    ratio = fuzz.ratio(cand_value.lower(), existing_value)
                    score += (weight / total_weight) * (100 if ratio > 90 else 60 if ratio > 75 else 0) * cand_conf

        attr_weight_per = weights["match_spec"]
        total_attr_weight = len(attributes) * attr_weight_per
        for field in model_fields:
            cand_value, cand_conf = candidate.get(field, ("", 0.20))
            existing_value = existing.get(field, "").lower() if existing.get(field) else ""
            if cand_value and existing_value and cand_value.lower() == existing_value:
                score += (attr_weight_per / (total_weight + total_attr_weight)) * 100 * cand_conf

        return min(round(score, 2), max_score)

    best_match = None
    best_score = 0.0
    brand_lower = brand.lower()
    ref_lower = reference_number.lower()
    potential_matches = {sku: data for sku, data in watch_models.items() 
                        if data["brand"].lower() == brand_lower or 
                           (ref_lower and data.get("reference_number", "").lower() == ref_lower)}
    for sku, model_data in potential_matches.items():
        score = score_match(model_data, candidate)
        if score > best_score:
            best_score = score
            best_match = sku

    if best_match and best_score >= weights["match_threshold"]:
        existing_model = watch_models[best_match]
        updates = {"updated_at": datetime.utcnow().isoformat()}
        for field, (value, conf) in candidate.items():
            if field not in {"brand", "reference_number"} and value:
                existing_value = existing_model.get(field, "")
                existing_conf = existing_model.get("confidence_score", 0.0) if field == "canonical_name" else 0.20
                if not existing_value or conf > existing_conf:
                    updates[field] = value
        if sale.get("image_url") and not existing_model.get("image_url"):
            updates["image_url"] = sale["image_url"]
        if updates:
            db.collection("watches").document(best_match).update(updates)
            watch_models[best_match].update(updates)
            logger.info(f"Enhanced {best_match} with: {updates}")
        return best_match

    existing_skus = set(watch_models.keys())
    model_code = reference_number if reference_number else re.sub(r"[^a-z0-9]", "", model_name.lower())
    sku = generate_sku(brand, model_code + "-" + model_name, existing_skus, brand_prefixes)

    all_confidences = [c[1] for _, c in candidate.items() if c[0]]
    confidence = round(mean(all_confidences), 2) if all_confidences else 0.0

    new_model = {
        "sku": sku,
        "canonical_name": canonical_name,
        "brand": brand,
        "reference_number": reference_number,
        "model_name": model_name,
        "year": year if year else None,
        "dial_color": attributes.get("dial_color", ("", 0.0))[0] or None,
        "case_material": attributes.get("case_material", ("", 0.0))[0] or None,
        "movement_type": attributes.get("movement", ("", 0.0))[0] or None,
        "band": attributes.get("band_type", ("", 0.0))[0] or None,
        "bezel_type": attributes.get("bezel_type", ("", 0.0))[0] or None,
        "crystal": attributes.get("crystal", ("", 0.0))[0] or None,
        "case_size": attributes.get("case_size", ("", 0.0))[0] or None,
        "water_resistance": attributes.get("water_resistance", ("", 0.0))[0] or None,
        "complications": attributes.get("complications", ("", 0.0))[0] or None,
        "confidence_score": confidence,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "source": "eBay sale " + sale.get("itemId", "unknown"),
        "history": {
            "hourly": [],
            "daily": []
        },
        "image_url": sale.get("image_url")
    }

    new_model = {k: v for k, v in new_model.items() if v is not None}

    db.collection("watches").document(sku).set(new_model)
    watch_models[sku] = new_model
    logger.info(f"Created new SKU {sku} with confidence {confidence}")
    return sku

def archive_old_sales(watch_id: str, cutoff_date: str, db: firestore.Client) -> None:
    """Archives sales older than cutoff date."""
    sales_ref = db.collection("sold_watches").document(watch_id).collection("sales")
    old_sales = sales_ref.where(filter=firestore.FieldFilter("soldDate", "<", cutoff_date)).get()
    for sale in old_sales:
        sale_data = sale.to_dict()
        db.collection("sold_watches_archive").document(watch_id).collection("sales").document(sale.id).set(sale_data)
        sales_ref.document(sale.id).delete()

def assess_sales_quality(sale_data: List[Dict], watch_id: str, term_sets: Dict[str, Set[str]], 
                        weights: Dict[str, float], watch_models: Dict[str, Dict]) -> List[float]:
    """Assesses sale quality, filtering out fakes and low-quality items."""
    if not sale_data:
        return []

    prices = []
    for s in sale_data:
        price_value = s.get("lastSoldPrice").get("value")
        if price_value is not None:
            try:
                prices.append(float(price_value))
            except (ValueError, TypeError):
                logger.warning(f"Invalid 'soldPrice' value in sale: {s}")
    
    if not prices:
        logger.info(f"No valid prices extracted for watch_id {watch_id}")
        return []

    titles = [s["title"].lower() for s in sale_data]
    conditions = [s["condition"].lower() for s in sale_data]
    brands = [s.get("itemAspects", {}).get("Brand", "").lower() for s in sale_data]
    refs = [s.get("itemAspects", {}).get("Reference Number", "") or s["title"] for s in sale_data]
    sellers = [s.get("seller", 0) for s in sale_data]
    descriptions = [s.get("description", "").lower() for s in sale_data]

    if watch_models is None:
        logger.warning(f"Watch model {watch_id} not found in `watches` collection")
        model_brand = ""
        hist_avg_price = None
        rolling_avg = None
    else:
        model_brand = watch_models.get("brand", "").lower()
        daily_history = sorted(watch_models.get("history", {}).get("daily", []), key=lambda x: x["date"], reverse=True)
        hist_avg_price = float(daily_history[0]["avg_price"]) if daily_history else None
        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        rolling_prices = [float(h["avg_price"]) for h in daily_history if h["date"] >= seven_days_ago]
        rolling_avg = sum(rolling_prices) / len(rolling_prices) if rolling_prices else hist_avg_price

    if len(prices) > 1:
        median_price = median(prices)
        mad = median([abs(p - median_price) for p in prices]) or 1e-6
        z_scores = [0.6745 * (p - median_price) / mad for p in prices]
        filtered = [(p, t, c, b, r, s, d, z) for p, t, c, b, r, s, d, z in 
                   zip(prices, titles, conditions, brands, refs, sellers, descriptions, z_scores) if abs(z) <= 3.5]
    else:
        filtered = [(p, t, c, b, r, s, d, 0) for p, t, c, b, r, s, d in 
                   zip(prices, titles, conditions, brands, refs, sellers, descriptions)]

    MIN_PRICE_THRESHOLD = 5.0
    MAX_PRICE_BOOTSTRAP = 50000.0

    authentic = []
    rejected = []

    for price, title, condition, brand, ref, seller_score, description, z_score in filtered:
        text = title + " " + description
        
        watch_score = 0
        title_watch_count = sum(1 for kw in term_sets["WATCH_TERMS"] if kw in title)
        if title_watch_count > 0:
            watch_score += weights["watch_title_terms"]
        desc_watch_count = sum(1 for kw in term_sets["WATCH_TERMS"] if kw in description)
        if desc_watch_count > 0:
            watch_score += weights["watch_desc_terms"]
        if brand in term_sets["VALID_BRANDS"] and (not model_brand or brand == model_brand):
            watch_score += weights["watch_valid_brand"]
        if any(kw in text for kw in term_sets["FEATURE_TERMS"]):
            watch_score += weights["watch_features"]
        if ref and re.match(r"[a-z]{0,2}-?[0-9]{3,6}[a-z]{0,2}[-/]?[0-9a-z]*", ref, re.IGNORECASE):
            watch_score += weights["watch_ref_number"]
        title_accessory_count = sum(1 for kw in term_sets["ACCESSORY_TERMS"] if kw in title)
        if title_accessory_count > title_watch_count:
            watch_score += weights["watch_accessory_penalty"]
        
        if watch_score < weights["watch_min_score"]:
            rejected.append((price, f"Non-watch item (score {watch_score} < {weights['watch_min_score']})"))
            continue
        
        quality_score = 0
        if "new with tags" in condition or "new" in condition:
            quality_score += weights["quality_new"]
        elif "new with defects" in condition or "open box" in condition:
            quality_score += weights["quality_new_defects"]
        elif "pre-owned" in condition or "used" in condition:
            quality_score += weights["quality_used_good"] if "good" in condition or "excellent" in condition else weights["quality_used_fair"]
        elif condition in term_sets["BAD_CONDITIONS"]:
            quality_score += weights["quality_poor"]
        else:
            quality_score += weights["quality_unspecified"]
        
        if not any(kw in text for kw in term_sets["FAKE_TERMS"]):
            quality_score += weights["quality_no_fake"]
        if any(kw in text for kw in term_sets["FEATURE_TERMS"]):
            quality_score += weights["quality_features"]
        
        desc_length = len(description.split())
        if desc_length > 50:
            quality_score += weights["quality_desc_rich"]
        elif desc_length > 20:
            quality_score += weights["quality_desc_moderate"]
        
        if any(kw in text for kw in term_sets["AUTHORITY_TERMS"]):
            quality_score += weights["quality_authority"]
        
        base_avg = rolling_avg or hist_avg_price
        if base_avg is None:
            price_range_lower = MIN_PRICE_THRESHOLD
            price_range_upper = MAX_PRICE_BOOTSTRAP
            quality_score += weights["quality_price_in_range"]
        else:
            price_range_lower = max(base_avg * 0.3, MIN_PRICE_THRESHOLD)
            price_range_upper = base_avg * 10 if brand in term_sets["LUXURY_BRANDS"] else base_avg * 5
            if price_range_lower <= price <= price_range_upper:
                quality_score += weights["quality_price_in_range"]
            elif price < price_range_lower:
                penalty = min(weights["quality_price_penalty_max"], int((price_range_lower - price) / base_avg * 50))
                quality_score -= penalty
            else:
                penalty = min(weights["quality_price_penalty_max"], int((price - price_range_upper) / base_avg * 50))
                quality_score -= penalty
        
        if seller_score >= 50:
            quality_score += weights["quality_seller_high"]
        elif seller_score < 5:
            quality_score += weights["quality_seller_low"]
        
        min_quality = weights["quality_min_luxury_new"] if brand in term_sets["LUXURY_BRANDS"] and "new" in condition else weights["quality_min_standard"]
        if "vintage" in text or "1940" in text:
            min_quality += weights["quality_vintage_adjust"]
        
        if quality_score < min_quality:
            rejected.append((price, f"Low quality (score {quality_score} < {min_quality})"))
            continue
        
        authentic.append(price)
    
    if rejected:
        logger.debug(f"Rejected {len(rejected)} sales for watch_id {watch_id}: {rejected}")
    
    return authentic

def calculate_confidence_score(prices: List[float], hist_avg: Optional[float], sample_size: int) -> float:
    """Calculates confidence in price data based on variance and history."""
    if not prices:
        return 0.0
    variance = statistics.variance(prices) if len(prices) > 1 else 0
    avg_price = sum(prices) / len(prices)
    hist_deviation = abs(avg_price - hist_avg) / hist_avg if hist_avg else 0
    score = max(0, min(100, 100 - (variance / 1000) - (hist_deviation * 50) + (sample_size * 5)))
    return round(score, 2)

def summarize_text(text: str, max_length: int = 200) -> str:
    """Generate a concise summary of text using sentence scoring."""
    try:
        blob = TextBlob(text)
        sentences = blob.sentences
        if not sentences:
            return text[:max_length]
        
        scored_sentences = []
        for sentence in sentences:
            score = len(sentence.words) / 20.0
            if any(keyword in sentence.lower() for keyword in ["watch", "brand", "model", "release", "feature"]):
                score += 0.5
            scored_sentences.append((sentence, score))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        summary_sentences = [s[0] for s in scored_sentences[:2]]
        summary = " ".join(str(s) for s in summary_sentences)
        return summary[:max_length] + ("..." if len(summary) > max_length else "")
    except Exception as e:
        logger.warning(f"Summary generation failed: {str(e)}")
        return text[:max_length]

def extract_published_date(soup: BeautifulSoup, entry: dict, source_type: str) -> str:
    """Extract published date from article or RSS entry."""
    try:
        if source_type == "article":
            date_elem = soup.select_one("time, .date, .published")
            if date_elem and date_elem.get("datetime"):
                return datetime.fromisoformat(date_elem["datetime"].replace("Z", "+00:00")).isoformat()
        elif source_type == "rss":
            return entry.get("published", datetime.utcnow().isoformat())
        return datetime.utcnow().isoformat()
    except Exception:
        return datetime.utcnow().isoformat()