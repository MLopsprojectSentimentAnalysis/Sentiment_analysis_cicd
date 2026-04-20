End-to-End MLOps Pipeline for Sentiment Analysis




A production-grade MLOps pipeline for binary sentiment classification of IMDb movie reviews, delivered as two live Google Cloud Run services. Compares three transformer architectures (BERT, DistilBERT, MiniLM) with full experiment tracking, model registry, rollback, drift detection, and conditional CI/CD.
IIT Jodhpur · M.Tech Artificial Intelligence · April 2026
Authors: Jahanvi Gajera (M25CSA012) · Akanksha Kapil (M25CSA033)


🌐 Live Deployment




Service

URL





🚀 Backend API (Swagger)

https://sentiment-api-565865319827.asia-south1.run.app/docs



📊 Streamlit Dashboard

https://sentiment-dashboard-565865319827.asia-south1.run.app



🤗 HuggingFace Model

https://huggingface.co/Jahanvi16/sentiment_analys_ak_jg



🎥 Demo Video

Google Drive link

https://drive.google.com/file/d/1iAUuw-9ELeMqYnP0GE88_-EaJ50rH7OG/view?usp=sharing


Click the URLs above to verify the pipeline is live. Both services auto-scale on Google Cloud Run (asia-south1). First request after idle has ~30 s cold-start latency; subsequent calls ~14 ms.



✨ Key Features
🧠 Three transformer models compared — BERT (89.36 %), DistilBERT (89.04 %, Production), MiniLM (87.84 %)
📊 MLflow — 12 training runs tracked, 3 versions registered, atomic rollback demonstrated
🐳 Docker multi-service — separate backend (FastAPI) and frontend (Streamlit) images
☁️ Cloud Run — two independent services deployed publicly in asia-south1
🤗 HuggingFace Hub — fine-tuned model published with full model card
📈 Drift Detection — PSI, Jensen-Shannon, Kolmogorov-Smirnov with triggered demo (PSI = 21.3)
🧪 Conditional CI/CD — 4-path GitHub Actions workflow saves up to ~77 % CI time
📦 DVC + GCS — content-addressable data versioning on Google Cloud Storage
🔒 Security-aware — no service-account keys; manual deploy gate per org policy


🏗️ Architecture
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐
│  Data   │───▶│  Train  │───▶│ Registry │───▶│  Serve  │───▶│  Deploy  │
│ IMDb +  │    │ BERT /  │    │ MLflow + │    │ FastAPI │    │ Cloud    │
│ DVC/GCS │    │ DistilB │    │ Rollback │    │ + Strml │    │ Run (×2) │
└─────────┘    └─────────┘    └──────────┘    └─────────┘    └──────────┘
                    ▲              │                │              │
                    │              ▼                ▼              ▼
                    │         ┌────────────────────────────────────┐
                    │         │ Monitor (PSI / JS / KS, Prometheus)│
                    │         └────────────────────────────────────┘
                    │                          │
                    └── drift detected ◀───────┘
                         (human-initiated)

            ┌───────────────┐
            │   CI (4 paths)│ ──▶ validates training code + builds images
            │ GitHub Actions│
            └───────────────┘


📐 Model Comparison

| Model            | Accuracy | F1     | Precision | Recall  | Status              |
|------------------|----------|--------|-----------|---------|---------------------|
| BERT-base        | 0.8936   | 0.8936 | 0.8939    | 0.8936  | Archived (v1)       |
| DistilBERT       | 0.8904   | 0.8904 | 0.8907    | 0.8904  | Production (v2) ⭐   |
| MiniLM-L6-v2     | 0.8784   | 0.8784 | 0.8794    | 0.8784  | Archived (v3)       |


Evaluated on 5,000 held-out IMDb reviews. DistilBERT selected for its accuracy-to-size trade-off on Cloud Run.


🔀 Conditional CI
Every push is routed through one of four execution paths based on which files changed:





| Path                                    | Trigger                                 | Jobs Run                        | Runtime   |
|-----------------------------------------|-----------------------------------------|---------------------------------|-----------|
| `src/training/`, `configs/`, `data/`    | training-smoke-test + backend build     | ~12 min                         | ⏱️ 12m    |
| `src/api/`, `backend Dockerfile`        | backend build only                      | ~10 min                         | ⏱️ 10m    |
| `dashboards/`, `frontend Dockerfile`    | frontend build only                     | ~3 min                          | ⏱️ 3m     |
| `*.md`, `docs`                          | lint + test only                        | ~45 s                           | ⚡ 45s     |


Implemented with dorny/paths-filter@v3. Reduces average CI time by ~60 %.


🚀 Quick Start
Option 1 — Use the live service (recommended)
# Single prediction
curl -X POST https://sentiment-api-565865319827.asia-south1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'

# Response: {"label":"positive","confidence":0.996,"latency_ms":14}

Or open the Streamlit dashboard: https://sentiment-dashboard-565865319827.asia-south1.run.app

Option 2 — Use the model via HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_ID = "Jahanvi16/sentiment_analys_ak_jg"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)

label_id = int(torch.argmax(probs))
label = model.config.id2label[label_id]
print(f"{label} ({float(probs[0][label_id]):.2%})")

Option 3 — Run locally
# Clone the repo
git clone https://github.com/MLopsprojectSentimentAnalysis/Sentiment_analysis_cicd.git
cd Sentiment_analysis_cicd

# Set up Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull data (requires GCP credentials)
dvc pull

# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5555 &

# Run training (needs GPU)
python src/training/train.py

# Launch backend API
uvicorn src.api.app:app --host 0.0.0.0 --port 8060

# Launch dashboard (separate terminal)
streamlit run dashboards/app.py


📁 Project Structure
Sentiment_analysis_cicd/
├── .github/workflows/
│   └── mlops_pipeline.yml      # Conditional 4-path CI
├── configs/
│   └── config.yaml             # Model & pipeline config
├── src/
│   ├── api/                    # FastAPI backend
│   │   └── app.py
│   ├── training/               # Training pipeline
│   │   └── train.py
│   ├── inference/              # Prediction service
│   ├── monitoring/             # Drift detection
│   │   └── drift_detector.py
│   └── data_loaders/
│   └── ci_training_smoke_test.py     # for conditional ci check 
├── dashboards/
│   └── app.py                  # Streamlit frontend
├── scripts/
│   ├── rollback.py             # MLflow Production rollback
│   ├── upload_to_huggingface.py
├── data/
│   └── processed/              # DVC-tracked parquet files
├── tests/                      # pytest suite
├── docker/
│   └── Dockerfile.frontend
├── Dockerfile                  # Backend (root)
├── requirements.txt
├── mlflow.db
└── README.md


🧪 Testing
# Run all unit tests with coverage
pytest tests/ -v --cov=src --cov-report=term

# Run specific test file
pytest tests/test_drift_detector.py -v

# Lint
flake8 src/ --count --select=E9,F63,F7,F82 --show-source
black --check src/


📊 Monitoring
Prometheus metrics
curl https://sentiment-api-565865319827.asia-south1.run.app/metrics | grep sentiment_

Exposed metrics:

sentiment_predictions_total{label="..."} — Counter
sentiment_prediction_latency_seconds — Histogram (p50, p95, p99)
sentiment_confidence_score — Histogram
sentiment_model_version — Gauge (active MLflow version)

Trigger a drift check
curl -X POST https://sentiment-api-565865319827.asia-south1.run.app/monitor/check

Returns PSI / JS / KS statistics per input feature, flagging drift when PSI > 0.2 or JS > 0.1.


🛠️ Tech Stack

| Layer          | Tools                                                                 |
|----------------|-----------------------------------------------------------------------|
| **ML**         | PyTorch, HuggingFace Transformers, Accelerate (fp16)                  |
| **Tracking**   | MLflow (tracking + registry + rollback)                               |
| **Data**       | DVC + Google Cloud Storage                                            |
| **API**        | FastAPI + uvicorn                                                     |
| **Frontend**   | Streamlit                                                             |
| **Drift**      | PSI, Jensen-Shannon, Kolmogorov-Smirnov                               |
| **CI/CD**      | GitHub Actions + dorny/paths-filter                                   |
| **Deploy**     | Docker multi-stage, Google Cloud Run                                  |
| **Monitoring** | Prometheus metrics endpoint                                           |



🔬 Drift Detection Demo
To validate drift detection, 162 predictions with deliberately skewed short-text inputs ("OK", "meh") were sent against the IMDb reference distribution:


| Feature           | PSI    | Threshold | Flag           |
|-------------------|--------|-----------|----------------|
| avg_word_length   | 21.3   | 0.2       | 🚨 DRIFT       |
| word_count        | 19.1   | 0.2       | 🚨 DRIFT       |
| text_length       | 5.79   | 0.2       | 🚨 DRIFT       |

The detector correctly flagged the distribution shift — two orders of magnitude above threshold.


🚧 Known Design Choices
Manual deployment. CI builds and validates images but does not deploy. GCP org policy (iam.disableServiceAccountKeyCreation) blocks service-account keys, and manual gcloud run deploy acts as an intentional human-approval gate.
Training stays on GPU. The CI training-smoke-test job validates pipeline imports on CPU runners; full retraining happens on the DGX cluster with 2× V100 GPUs.
Model weights out of Git. Weights are published to HuggingFace Hub and fetched at Docker build time via snapshot_download, keeping the repo lean.


📄 Report
Source: report/M25CSA012_M25CSA033_Project_Report.pdf


🎓 Acknowledgements
IIT Jodhpur AI group for DGX cluster access (2× NVIDIA V100 GPUs)
HuggingFace for transformer models and datasets library
IMDb dataset curators (Maas et al., 2011)


👥 Authors
Jahanvi Gajera (M25CSA012) 

Akanksha Kapil (M25CSA033)

