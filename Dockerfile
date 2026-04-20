FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements only
COPY docker/requirements-backend.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

# Copy the pre-trained DistilBERT model (baked into image)
COPY models/distilbert-base-uncased/20260420_001626/best_model/ /app/models/distilbert-base-uncased/best_model/

# Copy reference data for drift detection
COPY data/processed/reference.parquet /app/data/processed/reference.parquet

# Update config to use the baked-in model path
RUN sed -i 's|local_fallback_path:.*|local_fallback_path: "models/distilbert-base-uncased/best_model"|' configs/config.yaml

# Cloud Run uses port 8080 by default
ENV PORT=8080
ENV SKIP_MLFLOW=true
ENV PYTHONPATH=/app
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Start FastAPI, binding to Cloud Run's expected port
CMD uvicorn src.api.app:app --host 0.0.0.0 --port $PORT --workers 1
