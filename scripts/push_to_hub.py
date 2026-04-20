"""
Push the best Production model to the HuggingFace Hub.
Usage: python scripts/push_to_hub.py --repo your-username/sentiment-model
"""

import argparse

import mlflow
import yaml
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def push_to_hub(repo_id: str, local_model_path: str = None):
    cfg = load_config()
    model_name = cfg["mlflow"]["registered_model_name"]

    if local_model_path is None:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        model_uri = f"models:/{model_name}/Production"
        local_model_path = "models/hub_export"
        model = mlflow.pytorch.load_model(model_uri)
        model.save_pretrained(local_model_path)
        logger.info(f"Model exported to {local_model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["active_model"])

    logger.info(f"Pushing to HuggingFace Hub: {repo_id}")
    model.push_to_hub(repo_id, private=False)
    tokenizer.push_to_hub(repo_id, private=False)
    logger.success(f"Model pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. username/model-name")
    parser.add_argument("--local_path", default=None, help="Local model path (optional)")
    args = parser.parse_args()
    push_to_hub(args.repo, args.local_path)
