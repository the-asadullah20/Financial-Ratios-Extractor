# Production Dockerfile for Financial Ratios Extractor & Evaluation Engine
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies for PyMuPDF and C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/
COPY eval_data/ ./eval_data/

# Create required runtime directories
RUN mkdir -p output uploads_tmp raw_markdown qdrant_db

EXPOSE 8000

# Healthcheck to monitor app status
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Start production FastAPI web server
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
