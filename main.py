from fastapi import FastAPI, Depends, HTTPException, Header
import firebase_admin
from firebase_admin import credentials, auth
from google.cloud import firestore
from google.cloud import secretmanager
import os
import logging
from typing import Optional
from services import get_secret, refresh_ebay_token

# Initialize FastAPI app
app = FastAPI()

# Logging setup for debugging and tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Firebase
firebase_admin.initialize_app()

# Initialize Firestore client
db = firestore.Client()

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

# Load eBay credentials from Secret Manager
try:
    EBAY_AUTH_ENCODED = get_secret("ebay-auth-encoded", PROJECT_ID).strip()
except Exception as e:
    logger.error(f"Failed to initialize eBay credentials: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to initialize eBay credentials")

EBAY_API_TOKEN = None

async def get_current_user(authorization: str = Header(...)):
    """Authenticate user using Firebase token."""
    try:
        token = authorization.replace("Bearer ", "")
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Initialize eBay token
EBAY_API_TOKEN = refresh_ebay_token(EBAY_AUTH_ENCODED, EBAY_AUTH_URL)

# Import routes after app initialization
from backend_routes import *
from frontend_routes import *

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)