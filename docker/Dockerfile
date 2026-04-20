FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-backend.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

ARG HF_MODEL_REPO=Jahanvi16/Sentiment_analys_ak_jg
ENV HF_MODEL_REPO=${HF_MODEL_REPO}

RUN mkdir -p /app/models/distilbert-base-uncased/best_model && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${HF_MODEL_REPO}', local_dir='/app/models/distilbert-base-uncased/best_model', allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt'])"

COPY data/processed/reference.parquet /app/data/processed/reference.parquet

RUN sed -i 's|local_fallback_path:.*|local_fallback_path: "models/distilbert-base-uncased/best_model"|' configs/config.yaml

ENV PORT=8080
ENV SKIP_MLFLOW=true
ENV PYTHONPATH=/app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]