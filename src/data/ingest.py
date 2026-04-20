# src/data/ingest.py
"""
Data ingestion: downloads datasets from HuggingFace / GCS,
saves locally, and registers with DVC for versioning.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_dataset(source: str, max_samples: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download dataset from HuggingFace and return train/val/test splits."""
    logger.info(f"Downloading dataset: {source}")

    if source == "imdb":
        dataset = load_dataset("imdb")
        train_df = pd.DataFrame(dataset["train"]).rename(columns={"text": "text", "label": "label"})
        test_df  = pd.DataFrame(dataset["test"]).rename(columns={"text": "text", "label": "label"})
        # IMDb has no official val split — carve 10 % from train
        val_size = int(0.1 * len(train_df))
        val_df   = train_df.sample(n=val_size, random_state=42)
        train_df = train_df.drop(val_df.index).reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)

    elif source == "amazon_polarity":
        dataset = load_dataset("amazon_polarity")
        train_df = pd.DataFrame(dataset["train"])[["content", "label"]].rename(columns={"content": "text"})
        test_df  = pd.DataFrame(dataset["test"])[["content", "label"]].rename(columns={"content": "text"})
        val_size = int(0.1 * len(train_df))
        val_df   = train_df.sample(n=val_size, random_state=42)
        train_df = train_df.drop(val_df.index).reset_index(drop=True)

    elif source == "yelp_polarity":
        dataset = load_dataset("yelp_polarity")
        train_df = pd.DataFrame(dataset["train"])[["text", "label"]]
        test_df  = pd.DataFrame(dataset["test"])[["text", "label"]]
        val_size = int(0.1 * len(train_df))
        val_df   = train_df.sample(n=val_size, random_state=42)
        train_df = train_df.drop(val_df.index).reset_index(drop=True)

    else:
        raise ValueError(f"Unknown source: {source}")

    # Optional cap for fast iteration
    if max_samples:
        train_df = train_df.sample(n=min(max_samples, len(train_df)), random_state=42).reset_index(drop=True)
        val_df   = val_df.sample(n=min(max_samples // 5, len(val_df)), random_state=42).reset_index(drop=True)
        test_df  = test_df.sample(n=min(max_samples // 5, len(test_df)), random_state=42).reset_index(drop=True)

    logger.info(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                raw_dir: str, source: str) -> None:
    """Persist raw splits as parquet."""
    out = Path(raw_dir) / source
    out.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out / "train.parquet", index=False)
    val_df.to_parquet(out  / "val.parquet",   index=False)
    test_df.to_parquet(out / "test.parquet",  index=False)

    # Save a metadata sidecar
    meta = {
        "source": source,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "label_distribution": {
            "train": train_df["label"].value_counts().to_dict(),
            "val":   val_df["label"].value_counts().to_dict(),
        }
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved splits to {out}")


def dvc_add_and_push(raw_dir: str) -> None:
    """Stage data directory with DVC and push to remote."""
    logger.info("Adding data to DVC tracking…")
    try:
        subprocess.run(["dvc", "add", raw_dir], check=True)
        subprocess.run(["dvc", "push"],          check=True)
        logger.info("DVC push complete.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"DVC command failed (non-fatal in dev mode): {e}")


def upload_to_gcs(local_path: str, bucket: str, gcs_path: str) -> None:
    """Upload a local directory to Google Cloud Storage."""
    try:
        from google.cloud import storage  # type: ignore
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        local = Path(local_path)
        for file in local.rglob("*"):
            if file.is_file():
                blob_path = f"{gcs_path}/{file.relative_to(local)}"
                blob = bucket_obj.blob(blob_path)
                blob.upload_from_filename(str(file))
                logger.info(f"Uploaded {file} → gs://{bucket}/{blob_path}")
    except Exception as e:
        logger.warning(f"GCS upload skipped: {e}")


def main():
    cfg = load_config()
    source      = cfg["data"]["active_source"]
    max_samples = cfg["data"].get("max_samples")
    raw_dir     = cfg["data"]["raw_dir"]

    train_df, val_df, test_df = download_dataset(source, max_samples)
    save_splits(train_df, val_df, test_df, raw_dir, source)
    dvc_add_and_push(raw_dir)

    if os.getenv("UPLOAD_TO_GCS", "false").lower() == "true":
        upload_to_gcs(raw_dir, cfg["data"]["gcs_bucket"], cfg["data"]["gcs_raw_path"])


if __name__ == "__main__":
    main()
