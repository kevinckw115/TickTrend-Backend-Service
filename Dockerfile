# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set environment variables to ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
# Install Python dependencies including nltk
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir nltk

# Download nltk punkt and punkt_tab data
RUN python -m nltk.downloader punkt -d /usr/local/nltk_data \
    && python -m nltk.downloader punkt_tab -d /usr/local/nltk_data

ENV NLTK_DATA=/usr/local/nltk_data

# Copy your application code
COPY main.py .
COPY backend_routes.py .
COPY frontend_routes.py .
COPY services.py .

# Expose port 8080 (Cloud Run expects this)
EXPOSE 8080

# Command to run your FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]