from fastapi import FastAPI, Depends, HTTPException, Header
import firebase_admin
from firebase_admin import credentials, auth
from google.cloud import firestore
from google.api_core import exceptions, retry
from bs4 import BeautifulSoup
import requests, re
from rapidfuzz import fuzz, process
from datetime import datetime, timedelta
import os
import statistics
import logging
import time
from requests.exceptions import HTTPError
from typing import List, Dict, Optional, Set, Tuple
from statistics import mean, median
from pytz import UTC
from pydantic import BaseModel
from textblob import TextBlob
from feedparser import parse
import hashlib
from urllib.parse import urljoin
import html
from google.cloud import secretmanager

firebase_admin.initialize_app()
app = FastAPI()
db = firestore.Client()

# Logging setup for debugging and tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Google Cloud Secret Manager client
secret_client = secretmanager.SecretManagerServiceClient()

# Get project ID from environment variable
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
if not PROJECT_ID:
    logger.error("GCP_PROJECT_ID environment variable is not set")
    raise HTTPException(status_code=500, detail="GCP_PROJECT_ID environment variable is not set")

# eBay API credentials and endpoints
EBAY_API_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_AUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"

def get_secret(secret_name: str) -> str:
    """Retrieve a secret from Google Cloud Secret Manager."""
    try:
        secret_path = f"projects/{PROJECT_ID}/secrets/{secret_name}/versions/latest"
        response = secret_client.access_secret_version(request={"name": secret_path})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Failed to retrieve secret {secret_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve secret: {secret_name}")

# Load eBay credentials from Secret Manager
try:
    EBAY_AUTH_ENCODED = get_secret("ebay-auth-encoded").strip()
except Exception as e:
    logger.error(f"Failed to initialize eBay credentials: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to initialize eBay credentials")

EBAY_API_TOKEN = None

async def get_current_user(authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "")
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities from text, preserving content."""
    if not text:
        return ""
    # First, decode HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    # Parse with BeautifulSoup to remove tags
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ").strip()
    # Remove extra whitespace and leftover HTML fragments
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text.strip()

def extract_image(soup: Optional[BeautifulSoup], entry: dict, source_type: str, article_url: str) -> Optional[str]:
    """Extract the header or featured image URL from an article or RSS entry."""
    try:
        if source_type == "article":
            logger.debug(f"Extracting image for article: {article_url}")
            # Try Open Graph meta tag
            og_image = soup.select_one('meta[property="og:image"]')
            if og_image and og_image.get("content"):
                logger.debug(f"Found og:image: {og_image['content']}")
                return og_image["content"]
            # Try common featured image classes
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
            # Fallback: first image in article content
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
            # Check enclosure for images
            if "enclosures" in entry:
                for enc in entry["enclosures"]:
                    if enc.get("type", "").startswith("image/"):
                        logger.debug(f"Found enclosure image: {enc.get('url')}")
                        return enc.get("url")
            # Check media:content
            if "media_content" in entry:
                for media in entry["media_content"]:
                    if media.get("medium") == "image":
                        logger.debug(f"Found media:content image: {media.get('url')}")
                        return media.get("url")
            # Parse content:encoded or content for embedded images
            content = entry.get("content", [{}])[0].get("value", "") or entry.get("summary", "")
            if content:
                soup = BeautifulSoup(content, "html.parser")
                # Try specific WordPress image classes
                img = soup.select_one('img.wp-post-image, img.attachment-post-thumbnail, img.featured-image')
                if img and img.get("src"):
                    img_url = img["src"]
                    if not img_url.startswith("http"):
                        img_url = urljoin(article_url, img_url)
                    logger.debug(f"Found embedded image with class: {img_url}")
                    return img_url
                # Fallback: first image
                img = soup.find("img")
                if img and img.get("src"):
                    img_url = img["src"]
                    if not img_url.startswith("http"):
                        img_url = urljoin(article_url, img_url)
                    logger.debug(f"Found first embedded image: {img_url}")
                    return img_url
            logger.debug("No image found in RSS content")
            # Fallback: try fetching the article page (optional, can be commented out for performance)
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

# Refreshes eBay API token for authenticated requests
def refresh_ebay_token() -> str:
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {EBAY_AUTH_ENCODED}"}
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    try:
        resp = requests.post(EBAY_AUTH_URL, headers=headers, data=data)
        resp.raise_for_status()
        token = resp.json()["access_token"]
        logger.info("Successfully refreshed eBay API token")
        return token
    except requests.RequestException as e:
        logger.error(f"Failed to refresh eBay token: {e}")
        raise HTTPException(status_code=500, detail="Unable to refresh eBay token")

# Loads dynamic terms, regex patterns, and weights from Firestore
def load_dynamic_criteria(db: firestore.Client) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, float]]:
    # Fetch term sets (e.g., VALID_BRANDS, WATCH_TERMS)
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

    # Fetch regex patterns for extraction (e.g., dial_color, movement)
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

    # Fetch scoring weights for matching and quality assessment
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

# Loads brand prefixes, syncing with VALID_BRANDS
def load_brand_prefixes(term_sets: Dict[str, Set[str]], db: firestore.Client) -> Dict[str, str]:
    prefixes_ref = db.collection("search_criteria").document("brand_prefixes").get()
    valid_brands = term_sets["VALID_BRANDS"]
    
    if not prefixes_ref.exists:
        prefixes = {}
        used_prefixes = set()
        for brand in sorted(valid_brands):  # Sort for consistent prefix generation
            base = brand[:3].upper()
            prefix = base
            counter = 1
            while prefix in used_prefixes:  # Resolve conflicts
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
        if missing_brands:  # Add prefixes for new brands
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

# Fetches recent eBay sales with retry logic for rate limits
def fetch_ebay_sales(category_id: str = "281", limit: int = 100, start_time: Optional[str] = None, 
                    end_time: Optional[str] = None, max_sales: Optional[int] = 50) -> List[Dict]:
    headers = {"Authorization": f"Bearer {EBAY_API_TOKEN}"}
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
                response = requests.get(EBAY_API_URL, headers=headers, params=params)
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
                    batch.append({
                        "itemId": item["itemId"],
                        "title": item["title"],
                        "lastSoldPrice": {"value": item["price"]["value"], "currency": item["price"]["currency"]},
                        "lastSoldDate": item.get("itemEndDate", datetime.utcnow().isoformat()),
                        "condition": detail.get("condition", "Unknown"),
                        "itemAspects": detail.get("itemAspects", {}),
                        "description": detail.get("description", "No description available"),
                        "seller": detail.get("seller", {}).get("feedbackScore", 0)
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

# Generates a unique SKU using brand prefix and model code
def generate_sku(brand: str, model_code: str, existing_skus: Set[str], brand_prefixes: Dict[str, str]) -> str:
    brand_lower = brand.lower()
    prefix = brand_prefixes.get(brand_lower, brand[:3].upper())
    base_sku = f"{prefix}-{model_code.replace('/', '-').replace(' ', '').upper()}"
    variant = 1
    sku = f"{base_sku}-{variant:02d}"
    while sku in existing_skus:  # Increment variant until unique
        variant += 1
        sku = f"{base_sku}-{variant:02d}"
    return sku

# Identifies or creates a watch model based on sale data, fully dynamic
def identify_watch_model(sale: Dict, watch_models: Dict[str, Dict], term_sets: Dict[str, Set[str]], 
                        brand_prefixes: Dict[str, str], regex_patterns: Dict[str, str], weights: Dict[str, float], 
                        db: firestore.Client) -> str:
    title = sale.get("title", "").lower()
    aspects = sale.get("itemAspects", {})
    description = sale.get("description", "").lower()
    full_text = f"{title} {description}".strip()

    # Helper to extract features, preferring structured aspects over text
    def extract_feature(pattern: str, text: str, aspect_key: str, default: str = None) -> tuple[str, float]:
        aspect_value = aspects.get(aspect_key, "").strip().lower()
        if aspect_value:
            return aspect_value, 0.95
        matches = re.findall(pattern, text, re.I)
        if matches:
            value = matches[0][0] if isinstance(matches[0], tuple) else matches[0]
            return value, 0.60
        return default, 0.20

    # Extract core identifiers
    brand, brand_conf = extract_feature(regex_patterns["brand"], full_text, "Brand", "unknown")
    brand = brand.title()
    reference_number, ref_conf = extract_feature(regex_patterns["reference_number"], full_text, "Reference Number", "")
    model_name, model_conf = extract_feature(regex_patterns["model_name"], title, "Model", "")
    if not model_name and not reference_number:  # Fallback to second word
        tokens = title.split()
        model_name = tokens[1] if len(tokens) > 1 else "unknown"
    model_name = model_name.title()
    year, year_conf = extract_feature(regex_patterns["year"], full_text, "Year of Manufacture", "")

    # Early validation: Check if all core identifiers are valid
    if not brand or brand.lower() == "unknown" or \
       not model_name or model_name.lower() == "unknown" or \
       not reference_number:
        logger.debug(f"Sale {sale.get('itemId', 'unknown')} missing valid brand, model, or reference. Returning 'Unidentified'")
        return "Unidentified"

    # Extract all model-specific attributes from regex_patterns
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

    # Rename fields for consistency with schema
    if "movement" in attributes:
        attributes["movement_type"] = attributes.pop("movement")
    if "band_type" in attributes:
        attributes["band"] = attributes.pop("band_type")

    # Build canonical_name with mandatory feature
    canonical_parts = [brand, model_name]
    if reference_number:
        canonical_parts.append(reference_number)
    
    # Select a feature in priority order
    feature_priority = [
        "movement_type", "complications", "dial_color", "band", 
        "case_size", "case_material", "bezel_type"
    ]
    feature = None
    for field in feature_priority:
        if field in attributes:
            feature = attributes[field][0].capitalize()
            break
    if not feature:  # Fallback to "Watch" if no features are found (rare case)
        feature = "Watch"
    canonical_parts.append(feature)
    
    canonical_name = " ".join(p.strip() for p in canonical_parts if p).strip()
    canonical_conf = 0.80  # Derived value, moderate confidence

    # Build candidate profile
    candidate = {
        "brand": (brand, brand_conf),
        "reference_number": (reference_number, ref_conf),
        "model_name": (model_name, model_conf),
        "year": (year, year_conf),
        "canonical_name": (canonical_name, canonical_conf),
        **attributes
    }

    # Score match against existing models
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

        # Match additional attributes
        attr_weight_per = weights["match_spec"]
        total_attr_weight = len(attributes) * attr_weight_per
        for field in model_fields:
            cand_value, cand_conf = candidate.get(field, ("", 0.20))
            existing_value = existing.get(field, "").lower() if existing.get(field) else ""
            if cand_value and existing_value and cand_value.lower() == existing_value:
                score += (attr_weight_per / (total_weight + total_attr_weight)) * 100 * cand_conf

        return min(round(score, 2), max_score)

    # Find best match
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

    # Enhance existing model
    if best_match and best_score >= weights["match_threshold"]:
        existing_model = watch_models[best_match]
        updates = {"updated_at": datetime.utcnow().isoformat()}
        for field, (value, conf) in candidate.items():
            if field not in {"brand", "reference_number"} and value:
                existing_value = existing_model.get(field, "")
                existing_conf = existing_model.get("confidence_score", 0.0) if field == "canonical_name" else 0.20
                if not existing_value or conf > existing_conf:
                    updates[field] = value
        if updates:
            db.collection("watches").document(best_match).update(updates)
            watch_models[best_match].update(updates)
            logger.info(f"Enhanced {best_match} with: {updates}")
        return best_match

    # Create new model
    existing_skus = set(watch_models.keys())
    model_code = reference_number if reference_number else re.sub(r"[^a-z0-9]", "", model_name.lower())
    sku = generate_sku(brand, model_code + "-" + model_name, existing_skus, brand_prefixes)

    # Calculate overall confidence
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
            "hourly": [],  # Array of hourly price aggregates
            "daily": []    # Array of daily price aggregates
        }
    }

    # Remove None values for cleaner Firestore storage
    new_model = {k: v for k, v in new_model.items() if v is not None}

    db.collection("watches").document(sku).set(new_model)
    watch_models[sku] = new_model
    logger.info(f"Created new SKU {sku} with confidence {confidence}")
    return sku

# Archives sales older than cutoff date
def archive_old_sales(watch_id: str, cutoff_date: str, db: firestore.Client) -> None:
    sales_ref = db.collection("sold_watches").document(watch_id).collection("sales")
    old_sales = sales_ref.where(filter=firestore.FieldFilter("soldDate", "<", cutoff_date)).get()
    for sale in old_sales:
        sale_data = sale.to_dict()
        db.collection("sold_watches_archive").document(watch_id).collection("sales").document(sale.id).set(sale_data)
        sales_ref.document(sale.id).delete()
    # logger.info(f"Archived {len(old_sales)} old sales for {watch_id}")

# Assesses sale quality dynamically, filtering out fakes and low-quality items
def assess_sales_quality(sale_data: List[Dict], watch_id: str, term_sets: Dict[str, Set[str]], 
                        weights: Dict[str, float], db: firestore.Client) -> List[float]:
    if not sale_data:
        return []

    # Extract prices from sale data
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

    # Extract sale details for quality assessment
    titles = [s["title"].lower() for s in sale_data]
    conditions = [s["condition"].lower() for s in sale_data]
    brands = [s.get("itemAspects", {}).get("Brand", "").lower() for s in sale_data]
    refs = [s.get("itemAspects", {}).get("Reference Number", "") or s["title"] for s in sale_data]
    sellers = [s.get("seller", 0) for s in sale_data]
    descriptions = [s.get("description", "").lower() for s in sale_data]

    # Get watch model and historical price data
    watch_ref = db.collection("watches").document(watch_id).get()
    if not watch_ref.exists:
        logger.warning(f"Watch model {watch_id} not found in `watches` collection")
        model_brand = ""
        hist_avg_price = None
        rolling_avg = None
    else:
        watch_model = watch_ref.to_dict()
        model_brand = watch_model.get("brand", "").lower()
        daily_history = sorted(watch_model.get("history", {}).get("daily", []), key=lambda x: x["date"], reverse=True)
        hist_avg_price = float(daily_history[0]["avg_price"]) if daily_history else None
        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        rolling_prices = [float(h["avg_price"]) for h in daily_history if h["date"] >= seven_days_ago]
        rolling_avg = sum(rolling_prices) / len(rolling_prices) if rolling_prices else hist_avg_price

    # Filter outliers using Median Absolute Deviation (MAD)
    if len(prices) > 1:
        median_price = median(prices)
        mad = median([abs(p - median_price) for p in prices]) or 1e-6
        z_scores = [0.6745 * (p - median_price) / mad for p in prices]
        filtered = [(p, t, c, b, r, s, d, z) for p, t, c, b, r, s, d, z in 
                   zip(prices, titles, conditions, brands, refs, sellers, descriptions, z_scores) if abs(z) <= 3.5]
    else:
        filtered = [(p, t, c, b, r, s, d, 0) for p, t, c, b, r, s, d in 
                   zip(prices, titles, conditions, brands, refs, sellers, descriptions)]

    MIN_PRICE_THRESHOLD = 5.0  # Minimum acceptable price
    MAX_PRICE_BOOTSTRAP = 50000.0  # Max price when no history

    authentic = []
    rejected = []

    for price, title, condition, brand, ref, seller_score, description, z_score in filtered:
        text = title + " " + description
        
        # Step 1: Determine if it’s a watch
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
        
        # Step 2: Assess quality
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
        
        # Price range check
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
        
        # Apply quality threshold
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

# Calculates confidence in price data based on variance and history
def calculate_confidence_score(prices: List[float], hist_avg: Optional[float], sample_size: int) -> float:
    if not prices:
        return 0.0
    variance = statistics.variance(prices) if len(prices) > 1 else 0
    avg_price = sum(prices) / len(prices)
    hist_deviation = abs(avg_price - hist_avg) / hist_avg if hist_avg else 0
    score = max(0, min(100, 100 - (variance / 1000) - (hist_deviation * 50) + (sample_size * 5)))
    return round(score, 2)

# Consoludate redundant SKUs and aggregate historical sales
@app.get("/consolidate-skus")
async def consolidate_skus():
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    # Load watch models
    watch_models = {doc.id: doc.to_dict() for doc in db.collection("watches").get()}
    term_sets, regex_patterns, weights = load_dynamic_criteria(db)

    # Prepare features for clustering (unchanged)
    sku_features = []
    sku_list = list(watch_models.keys())
    for sku in sku_list:
        model = watch_models[sku]
        brand = model.get("brand", "").lower()
        ref_number = model.get("reference_number", "")
        text = " ".join(filter(None, [
            brand * 3,
            model.get("model_name", ""),
            ref_number * 3,
            model.get("canonical_name", ""),
            model.get("movement_type", ""),
            model.get("dial_color", ""),
            model.get("case_material", ""),
            model.get("band", ""),
            model.get("complications", "")
        ])).lower()
        daily_history = model.get("history", {}).get("daily", [])
        sales_count = sum(entry["sales_qty"] for entry in daily_history) if daily_history else 0
        confidence = model.get("confidence_score", 0.0)
        age = (datetime.utcnow() - datetime.fromisoformat(model["created_at"])).days
        sku_features.append({
            "sku": sku,
            "text": text,
            "sales_count": sales_count,
            "confidence": confidence,
            "age": age,
            "brand": brand,
            "ref_number": ref_number
        })

    # Vectorize text and cluster (unchanged)
    texts = [f["text"] for f in sku_features]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    text_vectors = vectorizer.fit_transform(texts).toarray()
    numerical_features = np.array([[f["sales_count"], f["confidence"], f["age"]] for f in sku_features])
    numerical_features = (numerical_features - numerical_features.min(axis=0)) / (numerical_features.max(axis=0) - numerical_features.min(axis=0) + 1e-6)
    combined_features = np.hstack((text_vectors, numerical_features))
    dbscan = DBSCAN(eps=0.7, min_samples=2, metric="cosine").fit(combined_features)
    labels = dbscan.labels_

    # Group SKUs by cluster (unchanged)
    initial_clusters = {}
    for sku_idx, label in enumerate(labels):
        if label == -1:
            continue
        label = int(label)
        if label not in initial_clusters:
            initial_clusters[label] = []
        initial_clusters[label].append(sku_features[sku_idx])

    # Refine clusters by brand and reference_number (unchanged)
    clusters = {}
    cluster_id = 0
    for label, skus in initial_clusters.items():
        brand_groups = {}
        for sku_data in skus:
            brand = sku_data["brand"] or "no_brand"
            if brand not in brand_groups:
                brand_groups[brand] = []
            brand_groups[brand].append(sku_data)
        for brand, brand_skus in brand_groups.items():
            ref_groups = {}
            for sku_data in brand_skus:
                ref = sku_data["ref_number"] or "no_ref"
                if ref not in ref_groups:
                    ref_groups[ref] = []
                ref_groups[ref].append(sku_data["sku"])
            for ref, sku_group in ref_groups.items():
                if len(sku_group) >= 2:
                    clusters[cluster_id] = sku_group
                    logger.info(f"Cluster {cluster_id} for brand {brand}, ref {ref}: {sku_group}")
                    cluster_id += 1
                else:
                    logger.debug(f"Skipping singleton {sku_group} for brand {brand}, ref {ref}")

    # Consolidate clusters
    for cluster_id, skus in clusters.items():
        if len(skus) <= 1:
            continue

        # Score SKUs to pick the primary (unchanged)
        sku_scores = []
        for sku in skus:
            model = watch_models[sku]
            daily_history = model.get("history", {}).get("daily", [])
            sales_count = sum(entry["sales_qty"] for entry in daily_history) if daily_history else 0
            feature_count = sum(1 for key in ["movement_type", "dial_color", "case_material", "band", "complications"] if model.get(key))
            confidence = model.get("confidence_score", 0.0)
            score = (weights["quality_price_in_range"] * sales_count +
                     weights["match_spec"] * feature_count +
                     confidence) / (sales_count + feature_count + 1)
            sku_scores.append((sku, score))

        # Pick the primary SKU
        primary_sku, _ = max(sku_scores, key=lambda x: x[1])
        redundant_skus = [s for s in skus if s != primary_sku]

        # Start batch for updates
        batch = db.batch()

        # Reassign sales from redundant SKUs to primary SKU
        for redundant_sku in redundant_skus:
            sales_ref = db.collection("sales").where(filter=firestore.FieldFilter("sku", "==", redundant_sku)).get()
            for sale in sales_ref:
                batch.update(sale.reference, {"sku": primary_sku})

        # Merge historical data
        primary_model = watch_models[primary_sku]
        primary_hourly = primary_model.get("history", {}).get("hourly", [])
        primary_daily = primary_model.get("history", {}).get("daily", [])

        # Aggregate hourly and daily from redundant SKUs
        merged_hourly_dict = {entry["time"]: {"time": entry["time"], "avg_price": 0, "sales_qty": 0, "confidence": 0, "count": 0} 
                              for entry in primary_hourly}
        merged_daily_dict = {entry["date"]: {"date": entry["date"], "avg_price": 0, "sales_qty": 0, "confidence": 0, "count": 0} 
                             for entry in primary_daily}

        for redundant_sku in redundant_skus:
            redundant_model = watch_models[redundant_sku]
            redundant_hourly = redundant_model.get("history", {}).get("hourly", [])
            redundant_daily = redundant_model.get("history", {}).get("daily", [])

            # Merge hourly
            for entry in redundant_hourly:
                if entry["time"] in merged_hourly_dict:
                    merged_hourly_dict[entry["time"]]["avg_price"] += entry["avg_price"] * entry["sales_qty"]
                    merged_hourly_dict[entry["time"]]["sales_qty"] += entry["sales_qty"]
                    merged_hourly_dict[entry["time"]]["confidence"] += entry["confidence"]
                    merged_hourly_dict[entry["time"]]["count"] += 1
                else:
                    merged_hourly_dict[entry["time"]] = {
                        "time": entry["time"],
                        "avg_price": entry["avg_price"] * entry["sales_qty"],
                        "sales_qty": entry["sales_qty"],
                        "confidence": entry["confidence"],
                        "count": 1
                    }

            # Merge daily
            for entry in redundant_daily:
                if entry["date"] in merged_daily_dict:
                    merged_daily_dict[entry["date"]]["avg_price"] += entry["avg_price"] * entry["sales_qty"]
                    merged_daily_dict[entry["date"]]["sales_qty"] += entry["sales_qty"]
                    merged_daily_dict[entry["date"]]["confidence"] += entry["confidence"]
                    merged_daily_dict[entry["date"]]["count"] += 1
                else:
                    merged_daily_dict[entry["date"]] = {
                        "date": entry["date"],
                        "avg_price": entry["avg_price"] * entry["sales_qty"],
                        "sales_qty": entry["sales_qty"],
                        "confidence": entry["confidence"],
                        "count": 1
                    }

            # Archive redundant SKU
            redundant_data = redundant_model.copy()
            redundant_data["archived_at"] = datetime.utcnow().isoformat()
            batch.set(db.collection("watches_archive").document(redundant_sku), redundant_data)
            batch.delete(db.collection("watches").document(redundant_sku))

        # Finalize merged hourly and daily, filtering out zero sales_qty
        merged_hourly = []
        for entry in merged_hourly_dict.values():
            total_sales = entry["sales_qty"]
            if total_sales > 0:  # Only include non-zero entries
                avg_price = entry["avg_price"] / total_sales
                confidence = entry["confidence"] / entry["count"] if entry["count"] > 0 else 0
                merged_hourly.append({
                    "time": entry["time"],
                    "avg_price": avg_price,
                    "sales_qty": total_sales,
                    "confidence": confidence
                })

        merged_daily = []
        for entry in merged_daily_dict.values():
            total_sales = entry["sales_qty"]
            if total_sales > 0:  # Only include non-zero entries
                avg_price = entry["avg_price"] / total_sales
                confidence = entry["confidence"] / entry["count"] if entry["count"] > 0 else 0
                merged_daily.append({
                    "date": entry["date"],
                    "avg_price": avg_price,
                    "sales_qty": total_sales,
                    "confidence": confidence
                })

        batch.update(db.collection("watches").document(primary_sku), {
            "history.hourly": merged_hourly,
            "history.daily": merged_daily,
            "updated_at": datetime.utcnow().isoformat()
        })

        batch.commit()
        logger.info(f"Consolidated {redundant_skus} into {primary_sku}")

    return {"status": "Consolidated", "merged_skus": clusters}

# Pulls eBay data and processes sales
@app.get("/pull-ebay-data")
async def pull_ebay_data(max_sales_per_pull: int = 50, hours_window: int = 1):
    global EBAY_API_TOKEN
    EBAY_API_TOKEN = refresh_ebay_token()
    
    now = datetime.utcnow()
    start_time = (now - timedelta(hours=hours_window)).isoformat() + "Z"
    end_time = now.isoformat() + "Z"
    
    config_ref = db.collection("config").document("ebay_pull")
    config = config_ref.get().to_dict() or {"last_pull": "1970-01-01T00:00:00Z"}
    if config["last_pull"] > start_time:
        start_time = config["last_pull"]
    logger.info(f"Pulling sales from {start_time} to {end_time}")
    
    sales = fetch_ebay_sales(start_time=start_time, end_time=end_time, max_sales=max_sales_per_pull)
    
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
                    authentic_prices = assess_sales_quality(sale_data, sku, term_sets, weights, db)
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
                    authentic_prices = assess_sales_quality(sale_data, sku, term_sets, weights, db)
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

# New Precomputation Endpoint for Trending Lists
@app.get("/precompute-trending")
async def precompute_trending():
    """Precompute trending lists by sales and growth, storing them in a 'trending' collection."""
    watch_models = {doc.id: doc.to_dict() for doc in db.collection("watches").get()}
    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    
    periods = {
        "sales": ["last_1_days", "last_7_days", "last_30_days", "last_90_days"],
        "growth": ["last_7_days", "last_30_days", "last_90_days"]
    }
    max_items = 20
    
    # Compute trending by sales
    sales_data = {}
    for period in periods["sales"]:
        days = int(period.split("_")[1].replace("days", ""))
        start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        sales_totals = {}
        
        for watch_id, watch in watch_models.items():
            daily_history = watch.get("history", {}).get("daily", [])
            filtered = [entry for entry in daily_history if entry["date"] >= start_date]
            total_sales = sum(entry["sales_qty"] for entry in filtered)
            if total_sales > 0:
                latest_entry = max(filtered, key=lambda x: x["date"], default=None)
                sales_totals[watch_id] = {
                    "sales_count": total_sales,
                    "avg_price": float(latest_entry["avg_price"]) if latest_entry else 0.0,
                    "brand": watch.get("brand", ""),
                    "model_name": watch.get("model_name", "")
                }
        
        trending = sorted(sales_totals.items(), key=lambda x: x[1]["sales_count"], reverse=True)[:max_items]
        sales_data[period] = [
            {
                "watch_id": watch_id,
                "brand": data["brand"],
                "model_name": data["model_name"],
                "sales_count": data["sales_count"],
                "avg_price": data["avg_price"]
            } for watch_id, data in trending
        ]
    
    # Compute trending by growth
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
                        "model_name": watch.get("model_name", "")
                    }
        
        trending = sorted(growth_totals.items(), key=lambda x: x[1]["growth_percentage"], reverse=True)[:max_items]
        growth_data[period] = [
            {
                "watch_id": watch_id,
                "brand": data["brand"],
                "model_name": data["model_name"],
                "growth_percentage": data["growth_percentage"],
                "current_avg_price": data["current_avg_price"]
            } for watch_id, data in trending
        ]
    
    # Store in Firestore 'trending' collection
    batch = db.batch()
    trending_doc = {
        "date": date_str,
        "sales": sales_data,
        "growth": growth_data,
        "updated_at": now.isoformat()
    }
    batch.set(db.collection("trending").document(date_str), trending_doc)
    batch.commit()
    
    logger.info(f"Precomputed trending lists for {date_str}")
    return {"status": "Trending lists precomputed", "date": date_str}

# Updated External API Endpoints
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
        "total_sales": total_sales
    }
    return response

# Endpoint to return sales for watch_id
@app.get("/watches/{watch_id}/sales", dependencies=[Depends(get_current_user)])
async def get_watch_sales(watch_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = 20, offset: int = 0):
    """Fetch individual sales for a watch model with optional date filtering."""
    logger.info(f"Fetching sales for watch_id: {watch_id}, limit: {limit}, offset: {offset}, start_date: {start_date}, end_date: {end_date}")
    
    try:
        # Check if watch exists
        watch_ref = db.collection("watches").document(watch_id).get()
        if not watch_ref.exists:
            logger.warning(f"Watch not found: {watch_id}")
            raise HTTPException(status_code=404, detail="Watch not found")

        # Build sales query
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

        # Apply ordering and pagination
        sales_query = sales_query.order_by("lastSoldDate", direction=firestore.Query.DESCENDING).offset(offset).limit(limit)
        
        # Execute query
        sales = sales_query.get()
        sales_data = []
        
        for sale in sales:
            sale_dict = sale.to_dict()
            try:
                # Validate required fields
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
    # Validate inputs
    valid_periods = {"1d": 1, "30d": 30, "90d": 90, "1y": 365}
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail="Invalid period. Use: 1d, 30d, 90d, 1y")
    
    # Parse end_date or use now, ensuring UTC awareness
    if end_date:
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00")).replace(tzinfo=UTC)
    else:
        end = datetime.utcnow().replace(tzinfo=UTC)  # Make UTC-aware
    start = (end - timedelta(days=valid_periods[period])).replace(tzinfo=UTC)
    
    # Fetch watch data
    watch_ref = db.collection("watches").document(watch_id).get()
    if not watch_ref.exists:
        raise HTTPException(status_code=404, detail="Watch not found")
    watch = watch_ref.to_dict()
    
    # Choose aggregation
    is_hourly = period == "1d"
    history_key = "hourly" if is_hourly else "daily"
    history_data = watch.get("history", {}).get(history_key, [])
    
    # Extend range for 1d continuity if before noon UTC
    if is_hourly and end.hour < 12:
        start = (start - timedelta(days=1)).replace(tzinfo=UTC)
    
    # Debug raw data
    logger.info(f"Raw {history_key} data for {watch_id}: {history_data}")
    
    # Filter data based on format
    filtered_data = []
    for entry in history_data:
        key = "time" if is_hourly else "date"
        value = entry[key]
        if is_hourly:
            # Handle hourly timestamp (e.g., "2025-04-07T12:00:00Z")
            entry_dt = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=UTC)
            if start <= entry_dt <= end:
                filtered_data.append(entry)
        else:
            # Handle daily date (e.g., "2025-04-07")
            entry_dt = datetime.fromisoformat(value + "T00:00:00+00:00").replace(tzinfo=UTC)
            if start <= entry_dt <= end:
                filtered_data.append(entry)
    
    # Sort chronologically
    def get_timestamp(entry):
        key = "time" if is_hourly else "date"
        value = entry[key]
        return datetime.fromisoformat(value.replace("Z", "+00:00") if is_hourly else value + "T00:00:00+00:00").replace(tzinfo=UTC)
    
    filtered_data.sort(key=get_timestamp)
    
    # Transform to response format
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

@app.get("/watches", dependencies=[Depends(get_current_user)])
async def search_watches(brand: Optional[str] = None, model: Optional[str] = None, 
                        sort_by: str = "name", limit: int = 20, offset: int = 0):
    try:
        # Normalize case
        if brand:
            brand = brand.capitalize()
        if model:
            model = model.capitalize()

        # Base query
        query = db.collection("watches")
        watches = set()  # Use set to avoid duplicates
        total = 0

        # Search by brand
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

        # Search by model
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

        # Combine results
        final_watches = []
        if watches:
            # Fetch full documents for unique watch_ids
            for watch_id in watches:
                doc = db.collection("watches").document(watch_id[0]).get()
                if doc.exists:
                    final_watches.append({
                        "watch_id": doc.id,
                        "brand": doc.to_dict().get("brand", ""),
                        "model_name": doc.to_dict().get("model_name", ""),
                        "reference_number": doc.to_dict().get("reference_number", ""),
                        "canonical_name": doc.to_dict().get("canonical_name", "")
                    })

        # Sort and paginate
        if sort_by == "name":
            final_watches.sort(key=lambda x: x["model_name"] or "")
        elif sort_by == "sales":
            final_watches.sort(key=lambda x: db.collection("watches").document(x["watch_id"]).get().to_dict().get("total_sales", 0), reverse=True)
        elif sort_by == "price":
            final_watches.sort(key=lambda x: db.collection("watches").document(x["watch_id"]).get().to_dict().get("last_avg_price", 0), reverse=True)
        else:
            raise HTTPException(status_code=400, detail="Invalid sort_by. Use 'name', 'sales', or 'price'.")

        # Apply pagination
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

class AddWatchRequest(BaseModel):
    watch_id: str
    purchase_price: float | None = None  # Optional purchase price

# Add watch to user's collection
@app.post("/users/{user_id}/collection", response_model=dict)
async def add_to_collection(user_id: str, request: AddWatchRequest, current_user: str = Depends(get_current_user)):
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

# Get user's collection with current values
@app.get("/users/{user_id}/collection", dependencies=[Depends(get_current_user)])
async def get_user_collection(user_id: str):
    
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
                "reference_number": watch_data.get("reference_number", ""),
                "last_known_price": last_price,
                "added_at": doc.to_dict()["added_at"],
            })
    
    return {
        "watches": watches,
        "total_value": total_value,
    }

# Get collection value history
@app.get("/users/{user_id}/collection/history", dependencies=[Depends(get_current_user)])
async def get_collection_history(user_id: str, period: str = "30d"):
    
    valid_periods = {"1d": 1, "30d": 30, "90d": 90, "1y": 365}
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail="Invalid period. Use: 1d, 30d, 90d, 1y")
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=valid_periods[period])
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Get user's collection
    collection_ref = db.collection("users").document(user_id).collection("collection").stream()
    watch_ids = [doc.to_dict()["watch_id"] for doc in collection_ref]
    if not watch_ids:
        return {"history": []}
    
    # Aggregate history for all watches
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

# Chrono Pulse endpoints
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

def summarize_text(text: str, max_length: int = 200) -> str:
    """Generate a concise summary of text using sentence scoring."""
    try:
        blob = TextBlob(text)
        sentences = blob.sentences
        if not sentences:
            return text[:max_length]
        
        # Score sentences based on length and keyword presence
        scored_sentences = []
        for sentence in sentences:
            score = len(sentence.words) / 20.0  # Favor medium-length sentences
            if any(keyword in sentence.lower() for keyword in ["watch", "brand", "model", "release", "feature"]):
                score += 0.5
            scored_sentences.append((sentence, score))
        
        # Select top sentences
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

@app.get("/pulse/sync")
async def sync_pulse(token: Optional[str] = None):
    """Trigger scraping and sentiment analysis for watch content (cron job only)."""
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
                        # Fetch article content for summary and image
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

            # Identify trending topics (e.g., "limited edition", "new release")
            topics = []
            for topic, pattern in regex_patterns.get("topics", {}).items():
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
                model_key = f"{brand} {model}" if brand else model
                model_mentions[model_key] = model_mentions.get(model_key, {"count": 0, "sentiment": 0.0, "links": []})
                model_mentions[model_key]["count"] += 1
                model_mentions[model_key]["sentiment"] += sentiment_score
                model_mentions[model_key]["links"].append(link)
            for topic in topics:
                topic_trends[topic] = topic_trends.get(topic, {"count": 0, "sentiment": 0.0})
                topic_trends[topic]["count"] += 1
                topic_trends[topic]["sentiment"] += sentiment_score

        batch.commit()

        batch = db.batch()
        for brand, data in brand_mentions.items():
            avg_sentiment = data["sentiment"] / data["count"] if data["count"] > 0 else 0
            trend_ref = db.collection("trends").document(hashlib.sha1(brand.encode()).hexdigest())
            batch.set(trend_ref, {
                "name": brand,
                "type": "brand",
                "sentiment_score": avg_sentiment,
                "mentions": data["count"],
                "top_links": data["links"][:3],
                "sentiment_history": firestore.ArrayUnion([{
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "score": avg_sentiment,
                    "mentions": data["count"]
                }]),
                "updated_at": datetime.utcnow().isoformat()
            }, merge=True)

        for model_key, data in model_mentions.items():
            avg_sentiment = data["sentiment"] / data["count"] if data["count"] > 0 else 0
            trend_ref = db.collection("trends").document(hashlib.sha1(model_key.encode()).hexdigest())
            batch.set(trend_ref, {
                "name": model_key,
                "type": "model",
                "sentiment_score": avg_sentiment,
                "mentions": data["count"],
                "top_links": data["links"][:3],
                "sentiment_history": firestore.ArrayUnion([{
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "score": avg_sentiment,
                    "mentions": data["count"]
                }]),
                "updated_at": datetime.utcnow().isoformat()
            }, merge=True)

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

# Run: uvicorn main:app --reload
