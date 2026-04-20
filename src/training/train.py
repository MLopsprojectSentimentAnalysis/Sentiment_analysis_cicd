"""
Fine-tune BERT / DistilBERT / MiniLM for binary sentiment classification.
All metrics, params, and artefacts are logged to MLflow.
"""

import os
from pathlib import Path

import numpy as np
import mlflow
import mlflow.pytorch
import yaml
from datasets import load_from_disk
from loguru import logger
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


def train_model(model_name: str, cfg: dict) -> dict:
    logger.info(f"Training model: {model_name}")

    source = cfg["data"]["active_source"]
    processed_dir = Path(cfg["data"]["processed_dir"]) / source / "tokenized"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["model"]["output_dir"]) / model_name.replace("/", "_") / timestamp

    dataset = load_from_disk(str(processed_dir))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=cfg["model"]["num_labels"],
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    tc = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tc["num_epochs"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        learning_rate=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
        warmup_ratio=tc["warmup_ratio"],
        fp16=tc["fp16"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        eval_strategy=tc["eval_strategy"],
        save_strategy=tc["save_strategy"],
        load_best_model_at_end=tc["load_best_model_at_end"],
        metric_for_best_model=tc["metric_for_best_model"],
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name.split("/")[-1]) as run:
        mlflow.log_params({
            "model_name": model_name,
            "dataset": source,
            "num_epochs": tc["num_epochs"],
            "batch_size": tc["batch_size"],
            "learning_rate": tc["learning_rate"],
            "max_length": cfg["model"]["max_length"],
            "runtimestemp": timestamp,
        })

        trainer.train()

        # Log final validation metrics (after training completes)
        val_results = trainer.evaluate(dataset["validation"])
        mlflow.log_metrics({f"val_{k.replace('eval_', '')}": v for k, v in val_results.items()})

        # Log test metrics (held-out set — the number you report)
        test_results = trainer.evaluate(dataset["test"])
        mlflow.log_metrics({f"test_{k.replace('eval_', '')}": v for k, v in test_results.items()})

        # Also log without prefix for backwards compat with selection logic
        mlflow.log_metrics({k.replace("eval_", ""): v for k, v in test_results.items()})

        eval_results = test_results   # keep original var for return statement below

        # eval_results = trainer.evaluate(dataset["test"])

        # mlflow.log_metrics({k.replace("eval_", ""): v for k, v in eval_results.items()})

        trainer.save_model(str(output_dir / "best_model"))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(output_dir / "best_model"))
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        mlflow.pytorch.log_model(
            unwrapped_model,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )
        mlflow.log_artifact(str(output_dir / "best_model"))

        logger.info(f"Run ID: {run.info.run_id} | Metrics: {eval_results}")
        return {"run_id": run.info.run_id, "metrics": eval_results, "output_dir": str(output_dir / "best_model")}


def select_best_model(results, cfg):
    threshold = cfg["mlflow"]["min_accuracy_threshold"]
    valid = [r for r in results if r["metrics"].get("eval_accuracy", 0) >= threshold]
    if not valid:
        logger.warning("No model met the minimum accuracy threshold!")
        return None
    best = max(valid, key=lambda r: r["metrics"].get("eval_f1", 0))
    logger.info(f"Best model: {best}")
    return best


def promote_model_to_registry(run_id: str, cfg: dict) -> None:
    # import yaml
    # with open(config_path) as f:
    #     config_dict = yaml.safe_load(f)

    # # Extract the winning model name from the result
    # winner_path = Path(best["output_dir"]).parent.parent.name   # e.g. "bert-base-uncased"
    # winner_model = winner_path.replace("_", "/")                  # convert back if it was a path
    # config_dict["model"]["active_model"] = winner_path
    # with open(config_path, "w") as f:
    #     yaml.safe_dump(config_dict, f)
    # logger.info(f"Updated active_model to: {winner_path}")

    client = mlflow.tracking.MlflowClient(cfg["mlflow"]["tracking_uri"])
    model_name = cfg["mlflow"]["registered_model_name"]

    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.run_id == run_id:
            for prod_v in client.get_latest_versions(model_name, stages=["Production"]):
                client.transition_model_version_stage(
                    name=model_name, version=prod_v.version, stage="Archived"
                )
            client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Production"
            )
            logger.info(f"Model version {v.version} promoted to Production.")
            return


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    candidates = cfg["model"]["candidates"]

    results = []
    for model_name in candidates:
        try:
            result = train_model(model_name, cfg)
            results.append(result)
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")

    best = select_best_model(results, cfg)
    if best:
        promote_model_to_registry(best["run_id"], cfg)
        logger.info(f"Pipeline complete. Best model at: {best['output_dir']}")
    else:
        logger.error("No model promoted - check thresholds or training logs.")


if __name__ == "__main__":
    main()
