# TickTrend Backend

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0+-teal.svg)](https://fastapi.tiangolo.com/)

TickTrend Backend is a Python-based service built with FastAPI and Google Cloud Firestore to pull, process, and aggregate eBay sales data for watches. It identifies watch models, tracks pricing trends, and provides precomputed trending lists, serving as the backbone for a watch market analysis platform.

## Features

- **eBay Data Pull**: Fetches recent watch sales from eBay’s Buy API with deduplication.
- **Watch Identification**: Dynamically identifies watch models and assigns unique SKUs.
- **Aggregation**: Computes daily and 4-hour price aggregates, filtering low-quality sales.
- **Trending Lists**: Precomputes trending watches by sales volume and price growth.
- **API Endpoints**: Exposes RESTful endpoints for watch details, price history, and trends.

## Prerequisites

- Python 3.9+
- Google Cloud Project with Firestore enabled
- eBay Developer API credentials (App ID and Cert ID)
- Firebase Admin SDK service account key

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ticktrend-backend.git
   cd ticktrend-backend
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```plaintext
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   EBAY_APP_ID=your-ebay-app-id
   EBAY_CERT_ID=your-ebay-cert-id
   ```

5. **Initialize Firestore**
   Ensure your Google Cloud project has Firestore in Native Mode. Place the service account key file at the path specified in `GOOGLE_APPLICATION_CREDENTIALS`.

## Usage

1. **Run the Application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The server will start at `http://localhost:8000`.

2. **Access API Documentation**
   - OpenAPI (Swagger): `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

3. **Key Endpoints**
   - **Pull eBay Data**: `GET /pull-ebay-data?max_sales_per_pull=50&hours_window=1`
   - **Identify Sales**: `GET /identify-raw-sales`
   - **Aggregate Data**: `GET /aggregate`
   - **Precompute Trends**: `GET /precompute-trending`
   - **Get Watch Details**: `GET /watches/{watch_id}`
   - **Price History**: `GET /watches/{watch_id}/price-history?start_date=2025-01-01T00:00:00Z&end_date=2025-04-08T00:00:00Z`
   - **Trending by Sales**: `GET /trending/sales?period=last_30_days`
   - **Trending by Growth**: `GET /trending/growth?period=last_30_days`

## Project Structure

```
ticktrend-backend/
├── main.py           # Core FastAPI application and endpoints
├── requirements.txt  # Python dependencies
├── .env              # Environment variables (not tracked)
├── README.md         # Project documentation
└── LICENSE           # License file
```

## Configuration

- **Firestore Collections**:
  - `sales`: Stores raw eBay sales data.
  - `watches`: Stores watch models with history.
  - `trending`: Stores precomputed trending lists.
  - `search_criteria`: Dynamic terms, regex, and weights.
- **eBay API**: Uses the Buy Browse API (`item_summary/search`) with OAuth2 client credentials.

## Development

### Dependencies
Install required packages:
```bash
pip install fastapi uvicorn google-cloud-firestore requests beautifulsoup4 rapidfuzz scikit-learn numpy
```

### Running Locally
Use the `--reload` flag with `uvicorn` for hot-reloading during development.

### Testing
Add unit tests in a `tests/` directory using `pytest` (not included yet).

## Deployment

Deploy to Google Cloud Run or another serverless platform:
1. Build a Docker image:
   ```Dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
2. Deploy:
   ```bash
   gcloud run deploy ticktrend-backend --source . --region us-central1 --platform managed
   ```

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and include tests with new features.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or support, open an issue or contact [kevin.ckw115@gmail.com].
