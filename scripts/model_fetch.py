# """
# Upload the fine-tuned DistilBERT sentiment model to Hugging Face Hub.

# Prerequisites:
#     - Hugging Face account (https://huggingface.co/join)
#     - Write-access token from https://huggingface.co/settings/tokens
#     - Local fine-tuned model at models/distilbert-base-uncased/20260420_001626/best_model/

# Usage:
#     export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
#     export HF_USERNAME="your-hf-username"
#     python scripts/upload_to_huggingface.py

# Optional args:
#     --repo-name <name>    Repository name on HF (default: sentiment-imdb-distilbert)
#     --private             Make the repo private (default: public)
#     --model-path <path>   Local model path (default: models/distilbert-base-uncased/20260420_001626/best_model)
# """
# import argparse
# import os
# import sys
# from pathlib import Path

# from huggingface_hub import HfApi, create_repo, upload_folder
# from loguru import logger


# MODEL_CARD_TEMPLATE = """---
# language:
# en
# license: apache-2.0
# library_name: transformers
# tags:
# sentiment-analysis
# text-classification
# distilbert
# imdb
# mlops
# datasets:
# imdb
# metrics:
# accuracy
# f1
# model-index:
# name: sentiment-imdb-distilbert
#   results:
#   - task:
#       type: text-classification
#       name: Sentiment Classification
#     dataset:
#       name: IMDb
#       type: imdb
#     metrics:
#     - type: accuracy
#       value: 0.8904
#     - type: f1
#       value: 0.8904
#     - type: precision
#       value: 0.8907
#     - type: recall
#       value: 0.8904
# ---"""

# """
# # Sentiment Analysis DistilBERT (IMDb)

# DistilBERT fine-tuned on the IMDb movie review dataset for binary sentiment classification.
# Part of an end-to-end MLOps pipeline for sentiment analysis, delivered as an M.Tech project
# at IIT Jodhpur.

# ## Model Details

# **Base model:** distilbert-base-uncased
# **Task:** Binary sentiment classification (positive / negative)
# **Training data:** 25,000 IMDb movie reviews (balanced)
# **Epochs:** 3
# **Batch size:** 32
# **Learning rate:** 2e-5
# **Max sequence length:** 128
# **Precision:** fp16 mixed-precision

# ## Performance

# Evaluated on 5,000 held-out IMDb reviews:

# | Metric    | Value  |
# |-----------|--------|
# | Accuracy  | 0.8904 |
# | F1-score  | 0.8904 |
# | Precision | 0.8907 |
# | Recall    | 0.8904 |

# ## Usage

# python

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# MODEL_ID = "{username}/{repo_name}"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
# model.eval()

# text = "This movie was absolutely fantastic!"
# inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

# with torch.no_grad():
#     logits = model(**inputs).logits
#     probs = torch.softmax(logits, dim=-1)

# label_id = int(torch.argmax(probs))
# label = model.config.id2label[label_id]
# confidence = float(probs[0][label_id])

# print(f"{{label}} ({{confidence:.2%}})")



# ## MLOps Integration

# This model is served via:
# **FastAPI backend** on Google Cloud Run
# **Streamlit dashboard** on Google Cloud Run
# **MLflow registry** for versioning and rollback
# **GitHub Actions CI** for automated testing

# ## Citation

# This is part of an academic project. If you use this model, please cite:"""

# def parse_args():
#     parser = argparse.ArgumentParser(description="Upload fine-tuned model to HuggingFace Hub")
#     parser.add_argument(
#         "--model-path",
#         default="/home/m25csa012/MLops_project2/Sentiment_Analysis/models/distilbert-base-uncased/20260419_234536/best_model",
#         help="Path to the local fine-tuned model directory",
#     )
#     parser.add_argument(
#         "--repo-name",
#         default="sentiment-imdb-distilbert",
#         help="Repository name on Hugging Face Hub",
#     )
#     parser.add_argument(
#         "--private",
#         action="store_true",
#         help="Make the repository private (default: public)",
#     )
#     parser.add_argument(
#         "--commit-message",
#         default="Upload fine-tuned DistilBERT for IMDb sentiment classification",
#     )
#     return parser.parse_args()


# def main():
#     arg = parse_args()

#     # Validate env vars
#     hf_token = os.environ.get("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")
#     hf_username = os.environ.get("jahanvi16")

#      # Create repo
#     repo_id = f"{hf_username}/{arg.repo_name}"
#     logger.info(f"Creating/verifying repo: {arg.repo_name}")
#     # if not hf_token:
#     #     logger.error("Missing HF_TOKEN environment variable")
#     #     logger.error("Get one from: https://huggingface.co/settings/tokens")
#     #     logger.error("Then run: export HF_TOKEN='hf_xxxxx'")
#     #     sys.exit(1)

#     # if not hf_username:
#     #     logger.error("Missing HF_USERNAME environment variable")
#     #     logger.error("Run: export HF_USERNAME='your-hf-username'")
#     #     sys.exit(1)

#     # Validate model path
#     model_path = Path(arg.model_path)
#     if not model_path.exists():
#         logger.error(f"Model path does not exist: {model_path}")
#         sys.exit(1)

#     required_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
#     missing = [f for f in required_files if not (model_path / f).exists()]
#     if missing:
#         logger.error(f"Model folder missing required files: {missing}")
#         sys.exit(1)

#     logger.info(f"Uploading {model_path} to {hf_username}/{arg.repo_name}")

#     # Authenticate
#     api = HfApi(token=hf_token)

#     try:
#         whoami = api.whoami()
#         logger.info(f"Authenticated as: {whoami['name']}")
#     except Exception as e:
#         logger.error(f"Authentication failed: {e}")
#         logger.error("Check your HF_TOKEN is valid and has write access")
#         sys.exit(1)


#     try:
#         create_repo(
#             repo_id=repo_id,
#             token=hf_token,
#             repo_type="model",
#             private=arg.private,
#             exist_ok=True,
#         )
#         logger.info("Repo ready")
#     except Exception as e:
#         logger.error(f"Failed to create repo: {e}")
#         sys.exit(1)

#     # Write model card
#     model_card_path = model_path / "README.md"
#     model_card = MODEL_CARD_TEMPLATE.format(username=hf_username, repo_name=args.repo_name)
#     model_card_path.write_text(model_card)
#     logger.info(f"Created model card: {model_card_path}")

#     # Upload
#     logger.info("Uploading files (this may take 1-3 minutes for ~270 MB)...")
#     try:
#         upload_folder(
#             folder_path=str(model_path),
#             repo_id=repo_id,
#             token=hf_token,
#             commit_message=args.commit_message,
#         )
#         logger.info("Upload complete!")
#         logger.info(f"Model available at: https://huggingface.co/{repo_id}")
#     except Exception as e:
#         logger.error(f"Upload failed: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()


import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger

MODEL_REPO = os.environ.get("HF_MODEL_REPO", "Jahanvi16/Sentiment_analys_ak_jg")
# LOCAL_PATH = "/home/m25csa012/MLops_project2/Sentiment_Analysis/models/bert-base-uncased/20260420_001343/best_model"


def main():
    # target = Path(LOCAL_PATH)
    # target.mkdir(parents=True, exist_ok=True)

    # logger.info(f"Downloading {MODEL_REPO} to {target}...")

    # hf_token = os.environ.get("HF_TOKEN")  # optional; required only for private repos

    try:
        snapshot_download(
            repo_id=MODEL_REPO,
            # local_dir=str(target),
            # token=hf_token,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
            ],
        )
        logger.info(f"Model downloaded to {target}")

        # Verify critical files
        required = ["config.json", "model.safetensors", "tokenizer.json"]
        missing = [f for f in required if not (target / f).exists()]
        if missing:
            logger.error(f"Missing required files: {missing}")
            sys.exit(1)

        logger.info("All required files present. Download complete.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()