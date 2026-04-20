# ============================================================
# Backend Dockerfile — FastAPI Inference Service
# ============================================================
# Bakes in the fine-tuned DistilBERT model (fetched from HuggingFace Hub).
# Target: Google Cloud Run (asia-south1).
# Image size: ~2.5 GB (PyTorch + transformers + 270MB model weights).
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (heavy layer, cached by Docker)
COPY docker/requirements-backend.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

# Download the fine-tuned DistilBERT from HuggingFace Hub.
# Model repo: https://huggingface.co/Jahanvi16/sentimentAnalysisDistilBERT
# This replaces the old "COPY models/..." approach so the build works in any
# environment (CI, laptop, Cloud Build) without requiring local weights.
ARG HF_MODEL_REPO=Jahanvi16/sentimentAnalysisDistilBERT
ENV HF_MODEL_REPO=${HF_MODEL_REPO}

RUN mkdir -p /app/models/distilbert-base-uncased/best_model && \
    python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='${HF_MODEL_REPO}', \
    local_dir='/app/models/distilbert-base-uncased/best_model', \
    allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt'] \
)" && \
    echo "Model downloaded from $HF_MODEL_REPO"

# Copy the drift-detection reference data
COPY data/processed/reference.parquet /app/data/processed/reference.parquet

# Update config to point at the baked-in model path
RUN sed -i 's|local_fallback_path:.*|local_fallback_path: "models/distilbert-base-uncased/best_model"|' configs/config.yaml

# Cloud Run uses port 8080 by default and injects PORT env var
ENV PORT=8080
ENV SKIP_MLFLOW=true
ENV PYTHONPATH=/app
EXPOSE 8080

# Health check (Cloud Run also does its own TCP check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Start FastAPI with single worker (Cloud Run scales horizontally via instances)
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1
