"""
Evaluate the current Production model on a held-out test set.
"""

import argparse
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)

from src.inference.predictor import SentimentPredictor


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(data_path: str, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    source = cfg["data"]["active_source"]

    if data_path is None:
        data_path = str(
            Path(cfg["data"]["processed_dir"]) / source / "test_clean.parquet"
        )

    logger.info(f"Loading test data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Test samples: {len(df):,}")

    predictor = SentimentPredictor(config_path)

    logger.info("Running predictions...")
    results = predictor.predict_batch(df["text"].tolist(), batch_size=64)

    y_true = df["label"].tolist()
    y_pred = [1 if r["label"] == "positive" else 0 for r in results]
    y_proba = [r["score_positive"] for r in results]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["negative", "positive"])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1": f1,
            "test_roc_auc": roc_auc,
        })
        mlflow.log_dict(
            {"confusion_matrix": cm.tolist(), "report": report},
            "evaluation_results.json"
        )
    logger.info("Evaluation results logged to MLflow.")

    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, help="Path to test parquet file")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    evaluate(args.data_path, args.config)
