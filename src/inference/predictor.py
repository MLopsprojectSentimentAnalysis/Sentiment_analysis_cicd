"""
Inference engine — loads model from local folder (primary) or MLflow registry (fallback).
"""

import time
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PREDICTION_COUNTER = Counter("sentiment_predictions_total", "Total predictions", ["label"])
PREDICTION_LATENCY = Histogram("sentiment_prediction_latency_seconds", "Prediction latency")
MODEL_VERSION_GAUGE = Gauge("sentiment_model_version", "Active model version")
CONFIDENCE_HISTOGRAM = Histogram(
    "sentiment_confidence_score",
    "Prediction confidence",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_model_and_tokenizer(model_path: str, device: str):
    logger.info(f"Loading model from {model_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(device)
    return model, tokenizer


def load_production_model(cfg: dict):
    """Load from local path first (reliable); track MLflow version for metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    local_path = cfg["model"].get("local_fallback_path")
    if local_path and Path(local_path).exists():
        logger.info(f"Loading from local path: {local_path}")
        model, tokenizer = get_model_and_tokenizer(local_path, device)

        try:
            import mlflow
            mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
            client = mlflow.tracking.MlflowClient()
            prod_vers = client.get_latest_versions(
                cfg["mlflow"]["registered_model_name"], stages=["Production"]
            )
            version = int(prod_vers[0].version) if prod_vers else 0
            MODEL_VERSION_GAUGE.set(version)
            logger.info(f"Tracking MLflow Production version: {version}")
        except Exception as e:
            logger.warning(f"Could not read MLflow version (non-fatal): {e}")
            MODEL_VERSION_GAUGE.set(0)

        return model, tokenizer, device

    raise RuntimeError(
        f"local_fallback_path not set or doesn't exist: {local_path}. "
        f"Set it in configs/config.yaml after training."
    )


class SentimentPredictor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = load_config(config_path)
        self.max_length = self.cfg["model"]["max_length"]
        self.model, self.tokenizer, self.device = load_production_model(self.cfg)

    def predict(self, text: Union[str, list]):
        single = isinstance(text, str)
        texts = [text] if single else text

        start = time.perf_counter()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        latency = time.perf_counter() - start
        PREDICTION_LATENCY.observe(latency / len(texts))

        results = []
        for i, t in enumerate(texts):
            label_id = int(np.argmax(probs[i]))
            label = self.model.config.id2label[label_id]
            confidence = float(probs[i][label_id])

            PREDICTION_COUNTER.labels(label=label).inc()
            CONFIDENCE_HISTOGRAM.observe(confidence)

            results.append({
                "text": t,
                "label": label,
                "confidence": confidence,
                "score_positive": float(probs[i][1]),
                "score_negative": float(probs[i][0]),
                "latency_ms": round(latency * 1000 / len(texts), 2),
            })

        return results[0] if single else results

    def predict_batch(self, texts: list, batch_size: int = 32) -> list:
        all_results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            all_results.extend(self.predict(chunk))
        return all_results
