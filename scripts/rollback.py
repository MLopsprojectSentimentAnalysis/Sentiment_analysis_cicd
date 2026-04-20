"""
Instant model rollback - promotes the previous Production version back to active.
Usage: python scripts/rollback.py [--version N]
"""

import argparse
import mlflow
import yaml
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def rollback(target_version: int = None):
    cfg = load_config()
    client = mlflow.tracking.MlflowClient(cfg["mlflow"]["tracking_uri"])
    model_name = cfg["mlflow"]["registered_model_name"]

    all_versions = sorted(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda v: int(v.version),
    )

    prod = client.get_latest_versions(model_name, stages=["Production"])
    if not prod:
        logger.error("No Production model found.")
        return

    current_version = int(prod[0].version)
    logger.info(f"Current Production version: {current_version}")

    if target_version:
        rollback_v = target_version
    else:
        archived = [v for v in all_versions if v.current_stage == "Archived"]
        if not archived:
            logger.error("No archived versions to roll back to.")
            return
        rollback_v = int(archived[-1].version)

    client.transition_model_version_stage(
        name=model_name, version=str(current_version), stage="Archived"
    )
    client.transition_model_version_stage(
        name=model_name, version=str(rollback_v), stage="Production"
    )
    logger.success(f"Rolled back: v{current_version} -> Archived, v{rollback_v} -> Production")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=None, help="Target version to restore")
    args = parser.parse_args()
    rollback(args.version)
