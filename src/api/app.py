"""
FastAPI inference service.
"""

import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, start_http_server
from pydantic import BaseModel, Field

from src.inference.predictor import SentimentPredictor
from src.monitoring.drift_detector import DriftDetector, BackgroundMonitor


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)


class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    score_positive: float
    score_negative: float
    latency_ms: float
    timestamp: str


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


predictor: Optional[SentimentPredictor] = None
drift_detector: Optional[DriftDetector] = None
prediction_log_path: str = "logs/predictions.jsonl"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, drift_detector, prediction_log_path

    cfg = load_config()
    prediction_log_path = cfg["api"]["prediction_log_path"]
    Path(prediction_log_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model...")
    predictor = SentimentPredictor()
    drift_detector = DriftDetector()

    prom_port = cfg["monitoring"]["prometheus_port"]
    start_http_server(prom_port)
    logger.info(f"Prometheus metrics on :{prom_port}")

    monitor = BackgroundMonitor(
        drift_detector,
        prediction_log_path,
        interval_seconds=cfg["monitoring"]["drift_check_interval_seconds"],
    )
    t = threading.Thread(target=monitor.run_forever, daemon=True)
    t.start()
    logger.info("Background drift monitor started.")

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Sentiment Analysis API",
    description="End-to-end MLOps pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _log_prediction(result: dict) -> None:
    with open(prediction_log_path, "a") as f:
        f.write(json.dumps({**result, "timestamp": datetime.utcnow().isoformat()}) + "\n")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict(request.text)
        result["timestamp"] = datetime.utcnow().isoformat()
        background_tasks.add_task(_log_prediction, result)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest, background_tasks: BackgroundTasks):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        results = predictor.predict_batch(request.texts)
        ts = datetime.utcnow().isoformat()
        for r in results:
            r["timestamp"] = ts
            background_tasks.add_task(_log_prediction, r)
        return BatchPredictResponse(
            results=[PredictResponse(**r) for r in results],
            total=len(results),
            timestamp=ts,
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor/check", tags=["Monitoring"])
async def trigger_drift_check(min_samples: int = 100):
    import pandas as pd
    if not Path(prediction_log_path).exists():
        return {"message": "No prediction log found yet."}

    preds = []
    with open(prediction_log_path) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    if len(preds) < min_samples:
        return {"message": f"Only {len(preds)} predictions logged; need {min_samples}."}

    current_df = pd.DataFrame([{"text": p["text"]} for p in preds])
    results = drift_detector.run_full_check(current_df, current_predictions=preds)
    return results


@app.get("/monitor/history", tags=["Monitoring"])
async def drift_history(n: int = 10):
    log = Path("logs/drift_history.jsonl")
    if not log.exists():
        return {"results": []}
    with open(log) as f:
        lines = f.readlines()
    records = [json.loads(l) for l in lines[-n:] if l.strip()]
    return {"results": records}


if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "src.api.app:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        workers=cfg["api"]["workers"],
        reload=False,
    )
