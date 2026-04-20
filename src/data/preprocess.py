# src/data/preprocess.py
"""
Text cleaning, tokenisation, and HuggingFace Dataset construction.
"""

import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Basic cleaning: strip HTML, collapse whitespace, lower-case."""
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text) # remove URLs
    text = re.sub(r"[^\w\s'!?.,]", " ", text)     # keep alphanumeric + light punct
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"]  = df["text"].astype(str).apply(clean_text)
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 10]             # drop near-empty strings
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_dataset(dataset: DatasetDict, model_name: str, max_length: int) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_hf_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> DatasetDict:
    return DatasetDict({
        "train": Dataset.from_pandas(train_df[["text", "label"]]),
        "validation": Dataset.from_pandas(val_df[["text", "label"]]),
        "test": Dataset.from_pandas(test_df[["text", "label"]]),
    })


def run_preprocessing(config_path: str = "configs/config.yaml") -> DatasetDict:
    cfg = load_config(config_path)
    source       = cfg["data"]["active_source"]
    raw_dir      = Path(cfg["data"]["raw_dir"]) / source
    processed_dir = Path(cfg["data"]["processed_dir"]) / source
    model_name   = cfg["model"]["active_model"]
    max_length   = cfg["model"]["max_length"]

    # Load raw parquet
    train_df = pd.read_parquet(raw_dir / "train.parquet")
    val_df   = pd.read_parquet(raw_dir / "val.parquet")
    test_df  = pd.read_parquet(raw_dir / "test.parquet")

    # Clean
    logger.info("Cleaning text…")
    train_df = clean_dataframe(train_df)
    val_df   = clean_dataframe(val_df)
    test_df  = clean_dataframe(test_df)

    # Save reference data for drift detection (training distribution)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(processed_dir / "train_clean.parquet", index=False)
    val_df.to_parquet(  processed_dir / "val_clean.parquet",   index=False)
    test_df.to_parquet( processed_dir / "test_clean.parquet",  index=False)

    # Save reference snapshot for monitoring baseline
    reference_path = Path(cfg["monitoring"]["reference_data_path"])
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.sample(n=min(5000, len(train_df)), random_state=42).to_parquet(reference_path, index=False)
    logger.info(f"Reference data saved to {reference_path}")

    # Build + tokenise HuggingFace datasets
    logger.info(f"Tokenising with {model_name}…")
    raw_ds      = build_hf_dataset(train_df, val_df, test_df)
    tokenized_ds = tokenize_dataset(raw_ds, model_name, max_length)

    # Cache tokenised dataset
    tokenized_ds.save_to_disk(str(processed_dir / "tokenized"))
    logger.info(f"Tokenised dataset saved to {processed_dir / 'tokenized'}")

    return tokenized_ds


if __name__ == "__main__":
    run_preprocessing()
