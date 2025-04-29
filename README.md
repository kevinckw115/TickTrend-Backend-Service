
# TickTrend 🕒

**TickTrend** is a cloud-native FastAPI application hosted on Google Cloud Platform (GCP). It aggregates eBay watch sales data to provide market trends, pricing insights, and user collection management. Built with Firestore, Cloud Build, and Cloud Run for scalability and security.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Setup](#-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

- **eBay Sales Aggregation**: Tracks watch sales from eBay's API, identifying models and prices.
- **Dynamic SKU Generation**: Creates unique SKUs based on brand and model.
- **Quality Filtering**: Removes fake or low-quality sales using Firestore criteria.
- **Trending Insights**: Precomputes trending watches by sales and price growth.
- **Chrono Pulse**: Scrapes watch content with sentiment analysis.
- **User Collections**: Manages authenticated user watch collections.
- **Cloud Deployment**: Runs on Cloud Run with CI/CD via Cloud Build.

---

## 🏗️ Architecture

- **Backend**: FastAPI for API endpoints and eBay integration.
- **Database**: Firestore for watch models, sales, and user data.
- **Secrets**: Google Cloud Secret Manager for secure API credentials.
- **CI/CD**: Cloud Build for automated builds and deployments.
- **Authentication**: Firebase Authentication for secure access.
- **External APIs**: eBay Browse API and RSS feeds for data and content.

---

## ✅ Prerequisites

- Google Cloud account with billing enabled
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- [Docker](https://www.docker.com/get-started)
- Python 3.11
- eBay API credentials ([eBay Developers Program](https://developer.ebay.com/))
- Firebase Admin SDK credentials

---

## 🚀 Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/ticktrend.git
cd ticktrend
```

### 2. Configure Google Cloud

- **Create Project**:
  ```bash
  gcloud projects create project-name --set-as-default
  ```

- **Enable APIs**:
  ```bash
  gcloud services enable cloudbuild.googleapis.com run.googleapis.com firestore.googleapis.com secretmanager.googleapis.com
  ```

- **Set Up Firebase**:
  Initialize Firebase in the [GCP Console](https://console.firebase.google.com/) and download Admin SDK credentials.

- **Configure IAM**:
  ```bash
  gcloud projects add-iam-policy-binding project-name \
    --member=serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com \
    --role=roles/run.admin
  gcloud projects add-iam-policy-binding project-name \
    --member=serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com \
    --role=roles/iam.serviceAccountUser
  ```

### 3. Set Up Secrets

- **Store eBay Credentials**:
  ```bash
  echo -n "your-ebay-auth-encoded" | gcloud secrets create ebay-auth-encoded --data-file=-
  ```

- **Grant Access**:
  ```bash
  gcloud secrets add-iam-policy-binding ebay-auth-encoded \
    --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor
  ```

### 4. Deploy

- **Build and Deploy**:
  ```bash
  gcloud builds submit --config cloudbuild.yaml .
  ```

- **Verify**:
  ```bash
  gcloud run services describe project-service --region us-central1
  ```

---

## 🛠️ Usage

### API Endpoints

| Endpoint | Method | Description | Authentication | Key Parameters |
|----------|--------|-------------|----------------|----------------|
| `/watches/{watch_id}` | GET | Get watch details | Required | `watch_id` (path) |
| `/watches/{watch_id}/sales` | GET | Fetch sales with filtering | Required | `watch_id` (path), `start_date`, `end_date`, `limit`, `offset` (query) |
| `/watches/{watch_id}/price-trend` | GET | Get price trends | Required | `watch_id` (path), `period` (1d, 30d, 90d, 1y), `end_date` (query) |
| `/watches` | GET | Search watches | Required | `brand`, `model`, `sort_by` (name, sales, price), `limit`, `offset` (query) |
| `/users/{user_id}/collection` | POST | Add watch to collection | Required | `user_id` (path), `watch_id`, `purchase_price` (body) |
| `/users/{user_id}/collection` | GET | Get user collection | Required | `user_id` (path) |
| `/users/{user_id}/collection/{watch_id}` | DELETE | Remove watch from collection | Required | `user_id`, `watch_id` (path) |
| `/trending/sales` | GET | Trending watches by sales | Required | `period` (last_7_days, last_30_days, last_90_days), `limit` (query) |
| `/trending` | GET | Trending brands/models | Required | `limit`, `sort_by` (trend_score, mentions, sales_count), `trend_type` (brand, model), `min_sentiment` (query) |
| `/pulse/content` | GET | Curated articles | Optional | `filter` (tag), `limit`, `offset` (query) |
| `/brands/{brand}` | GET | Brand details | Required | `brand` (path) |
| `/models/{model}` | GET | Model details | Required | `model` (path) |
| `/pull-ebay-data` | GET | Pull eBay sales | Optional | `max_sales_per_pull`, `hours_window` (query) |
| `/aggregate` | GET | Aggregate sales data | Optional | None |
| `/fetch-missing-image-urls` | GET | Fetch missing image URLs | Optional | `limit`, `start_date` (query) |
| `/consolidate-skus` | GET | Consolidate redundant SKUs | Optional | `limit` (query) |
| `/clean-html` | GET | Clean HTML from sales | Optional | None |
| `/reset-processed` | GET | Reset processed flags | Optional | `cutoff` (query, ISO date) |
| `/precompute-trending` | GET | Precompute trends | Optional | `max_items` (query) |
| `/pulse/sync` | GET | Sync content with sentiment | Optional | `token` (query) |
| `/` | GET | Health check | None | None |

**Example Request** (Watch Details):
```bash
curl -H "Authorization: Bearer <firebase-token>" https://your-service-url.a.run.app/watches/rolex_submariner_date_116610ln_2020
```

**Example Response**:
```json
{
  "watch_id": "rolex_submariner_date_116610ln_2020",
  "brand": "Rolex",
  "model_name": "Submariner Date",
  "last_known_price": { "date": "2025-04-01", "avg_price": 15000.0 },
  "total_sales": 120
}
```

### Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up NLTK**:
   ```bash
   python -m nltk.downloader punkt punkt_tab -d ./nltk_data
   export NLTK_DATA=./nltk_data
   ```

3. **Set Environment Variables**:
   ```bash
   export GCP_PROJECT_ID=project-name
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/firebase-adminsdk.json
   ```

4. **Run**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   Access at `http://localhost:8000`.

---

## 📂 Project Structure

```
ticktrend-api/
├── Dockerfile          # Docker configuration
├── cloudbuild.yaml     # Google Cloud Build CI/CD
├── requirements.txt    # Python dependencies
├── main.py             # FastAPI app initialization
├── frontend_routes.py  # User-facing API endpoints
├── backend_routes.py   # Data processing endpoints
├── services.py         # Business logic and utilities
└── seed_models.py      # Watch model seeding script
```

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push: `git push origin feature/your-feature`
5. Open a pull request.

Adhere to the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## 📜 License

MIT License. See [LICENSE](LICENSE).

---

## 📬 Contact

- **Maintainer**: Your Name (kevin.ckw115@gmail.com)

Thanks for exploring TickTrend! 🚀

