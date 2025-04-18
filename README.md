
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

| Endpoint | Description | Authentication |
|----------|-------------|----------------|
| `GET /` | Health check (`{ "status": "OK" }`) | None |
| `GET /watches/{watch_id}` | Watch details | Required |
| `GET /watches/{watch_id}/price-trend?period=30d` | Price trends | Required |
| `GET /trending/sales?period=last_30_days` | Trending by sales | Required |
| `GET /pulse/content` | Curated watch articles | None |
| `POST /users/{user_id}/collection` | Add watch to collection | Required |

Explore all endpoints in `main.py` or the upcoming API documentation.

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
ticktrend/
├── Dockerfile        # Container configuration
├── cloudbuild.yaml   # CI/CD configuration
├── main.py           # FastAPI application
├── requirements.txt  # Dependencies
├── README.md         # This file
└── nltk_data/        # NLTK data (generated)
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

