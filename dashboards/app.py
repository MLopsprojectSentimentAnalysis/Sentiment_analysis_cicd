"""
Streamlit client dashboard.
Run: streamlit run dashboards/app.py
"""

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

import os
API_URL = os.environ.get("API_BASE_URL", "http://localhost:8060")

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("Sentiment Analysis - MLOps Dashboard")
st.caption("End-to-end MLOps Pipeline - BERT / DistilBERT / MiniLM")


with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API: {health['status'].upper()}")
        st.write(f"Model loaded: {health['model_loaded']}")
        st.write(f"Last checked: {health['timestamp'][:19]}")
    except Exception:
        st.error("API Unreachable")

    st.divider()
    api_url_input = st.text_input("API URL", value=API_URL)
    if api_url_input:
        API_URL = api_url_input


tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Batch", "Drift Monitor", "History"])


with tab1:
    st.subheader("Single Text Prediction")
    text_input = st.text_area("Enter text to analyse:", height=150,
                              placeholder="This movie was absolutely fantastic!")

    if st.button("Analyse", type="primary"):
        if text_input.strip():
            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={"text": text_input},
                        timeout=30
                    )
                    result = resp.json()

                    col1, col2, col3 = st.columns(3)
                    label = result["label"].upper()
                    col1.metric("Sentiment", label)
                    col2.metric("Confidence", f"{result['confidence']:.1%}")
                    col3.metric("Latency", f"{result['latency_ms']} ms")

                    st.progress(result["score_positive"],
                                text=f"Positive: {result['score_positive']:.1%}  |  "
                                     f"Negative: {result['score_negative']:.1%}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text.")


with tab2:
    st.subheader("Batch Prediction")
    uploaded = st.file_uploader("Upload CSV (must have a 'text' column)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        elif st.button("Run Batch Inference", type="primary"):
            texts = df["text"].dropna().tolist()
            with st.spinner(f"Processing {len(texts)} texts..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"texts": texts[:500]},
                        timeout=120,
                    )
                    results = resp.json()["results"]
                    result_df = pd.DataFrame(results)[["text", "label", "confidence", "latency_ms"]]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Processed", len(result_df))
                        dist = result_df["label"].value_counts()
                        st.bar_chart(dist)
                    with col2:
                        st.metric("Mean Confidence", f"{result_df['confidence'].mean():.1%}")
                        st.dataframe(result_df, use_container_width=True)

                    csv = result_df.to_csv(index=False).encode()
                    st.download_button("Download Results", csv, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")


with tab3:
    st.subheader("Drift Detection")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Drift Check Now"):
            with st.spinner("Checking for drift..."):
                try:
                    resp = requests.post(f"{API_URL}/monitor/check?min_samples=50", timeout=60)
                    result = resp.json()

                    if result.get("overall_drift"):
                        st.error("DRIFT DETECTED!")
                    else:
                        st.success("No significant drift detected.")

                    with st.expander("Detailed Report"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        history_path = Path("logs/drift_history.jsonl")
        if history_path.exists():
            records = []
            with open(history_path) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

            if records:
                chart_data = []
                for r in records[-20:]:
                    feat_drift = r.get("input_drift", {}).get("feature_drift", {})
                    max_psi = max((v.get("psi", 0) for v in feat_drift.values()), default=0)
                    max_js = max((v.get("js_divergence", 0) for v in feat_drift.values()), default=0)
                    chart_data.append({
                        "timestamp": r["timestamp"][:16],
                        "max_psi": max_psi,
                        "max_js": max_js,
                        "drifted": int(r.get("overall_drift", False)),
                    })

                chart_df = pd.DataFrame(chart_data).set_index("timestamp")
                st.line_chart(chart_df[["max_psi", "max_js"]],
                              use_container_width=True)
                st.caption("PSI > 0.2 or JS > 0.1 triggers a drift alert")


with tab4:
    st.subheader("Prediction Log")
    log_path = Path("logs/predictions.jsonl")
    if log_path.exists():
        preds = []
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))

        if preds:
            pred_df = pd.DataFrame(preds[-200:])
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", len(preds))
            if "label" in pred_df.columns:
                pos_rate = (pred_df["label"] == "positive").mean()
                col2.metric("Positive Rate", f"{pos_rate:.1%}")
            if "confidence" in pred_df.columns:
                col3.metric("Avg Confidence", f"{pred_df['confidence'].mean():.1%}")

            st.dataframe(
                pred_df[["timestamp", "text", "label", "confidence"]].tail(50),
                use_container_width=True,
            )
        else:
            st.info("No predictions logged yet.")
    else:
        st.info("Prediction log not found. Make some predictions first!")
