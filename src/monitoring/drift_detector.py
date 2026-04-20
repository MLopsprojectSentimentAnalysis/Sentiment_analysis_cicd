"""
Data drift + prediction drift detection.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from prometheus_client import Counter, Gauge
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not installed - skipping HTML drift reports.")


DRIFT_DETECTED_GAUGE = Gauge("sentiment_drift_detected", "1 if drift detected, 0 otherwise")
PSI_GAUGE = Gauge("sentiment_psi_score", "Latest PSI score")
JS_DIVERGENCE_GAUGE = Gauge("sentiment_js_divergence", "Latest JS divergence score")
PREDICTION_DRIFT_GAUGE = Gauge("sentiment_prediction_drift", "Prediction distribution drift")
DRIFT_ALERT_COUNTER = Counter("sentiment_drift_alerts_total", "Total drift alerts fired")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _text_length_feature(df):
    return df["text"].str.len().values.astype(float)


def _word_count_feature(df):
    return df["text"].str.split().str.len().values.astype(float)


def _avg_word_length_feature(df):
    return df["text"].apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0
    ).values.astype(float)


def _compute_histogram(values, bins=20):
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist + 1e-10
    return hist / hist.sum()


def _psi(expected, actual, bins=20):
    e_hist = _compute_histogram(expected, bins)
    a_hist = _compute_histogram(actual, bins)
    return float(np.sum((a_hist - e_hist) * np.log(a_hist / e_hist)))


def _js_divergence(expected, actual, bins=20):
    e_hist = _compute_histogram(expected, bins)
    a_hist = _compute_histogram(actual, bins)
    return float(jensenshannon(e_hist, a_hist))


def _ks_test(reference, current):
    stat, p_value = ks_2samp(reference, current)
    return float(stat), float(p_value)


class DriftDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = load_config(config_path)
        mc = self.cfg["monitoring"]
        self.psi_thresh = mc["psi_threshold"]
        self.js_thresh = mc["js_divergence_threshold"]
        self.pred_thresh = mc["prediction_drift_threshold"]
        self.min_samples = mc["min_samples_for_drift"]
        self.report_dir = Path("logs/drift_reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

        ref_path = mc["reference_data_path"]
        if Path(ref_path).exists():
            self.reference_df = pd.read_parquet(ref_path)
            logger.info(f"Reference dataset loaded: {len(self.reference_df):,} samples")
        else:
            self.reference_df = None
            logger.warning(f"Reference data not found at {ref_path}. Run preprocessing first.")

    def check_input_drift(self, current_df):
        if self.reference_df is None:
            return {"error": "No reference data"}
        if len(current_df) < self.min_samples:
            return {"skipped": True, "reason": f"Too few samples ({len(current_df)})"}

        results = {}
        features = {
            "text_length": (_text_length_feature, self.reference_df, current_df),
            "word_count": (_word_count_feature, self.reference_df, current_df),
            "avg_word_length": (_avg_word_length_feature, self.reference_df, current_df),
        }

        any_drift = False
        for feat_name, (extractor, ref, cur) in features.items():
            ref_vals = extractor(ref)
            cur_vals = extractor(cur)

            psi = _psi(ref_vals, cur_vals)
            js = _js_divergence(ref_vals, cur_vals)
            ks_stat, ks_p = _ks_test(ref_vals, cur_vals)

            drifted = psi > self.psi_thresh or js > self.js_thresh
            if drifted:
                any_drift = True

            results[feat_name] = {
                "psi": round(psi, 4),
                "js_divergence": round(js, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": round(ks_p, 4),
                "drift_detected": drifted,
            }

        all_psi = [v["psi"] for v in results.values()]
        all_js = [v["js_divergence"] for v in results.values()]
        PSI_GAUGE.set(max(all_psi))
        JS_DIVERGENCE_GAUGE.set(max(all_js))
        DRIFT_DETECTED_GAUGE.set(1 if any_drift else 0)
        if any_drift:
            DRIFT_ALERT_COUNTER.inc()

        return {"feature_drift": results, "drift_detected": any_drift}

    def check_prediction_drift(self, reference_preds, current_preds):
        def label_dist(preds):
            labels = [p["label"] for p in preds]
            total = len(labels)
            pos = labels.count("positive") / total
            neg = 1 - pos
            return np.array([neg + 1e-10, pos + 1e-10])

        ref_dist = label_dist(reference_preds)
        cur_dist = label_dist(current_preds)
        js = float(jensenshannon(ref_dist / ref_dist.sum(), cur_dist / cur_dist.sum()))

        drifted = js > self.pred_thresh
        PREDICTION_DRIFT_GAUGE.set(js)
        if drifted:
            DRIFT_ALERT_COUNTER.inc()

        return {
            "reference_positive_rate": round(float(ref_dist[1]), 4),
            "current_positive_rate": round(float(cur_dist[1]), 4),
            "js_divergence": round(js, 4),
            "drift_detected": drifted,
        }

    def check_confidence_degradation(self, predictions):
        confidences = [p["confidence"] for p in predictions]
        mean_conf = float(np.mean(confidences))
        low_conf_pct = float(np.mean(np.array(confidences) < 0.7) * 100)

        alert = low_conf_pct > 20
        return {
            "mean_confidence": round(mean_conf, 4),
            "low_confidence_pct": round(low_conf_pct, 2),
            "degradation_alert": alert,
        }

    def generate_evidently_report(self, current_df):
        if not EVIDENTLY_AVAILABLE:
            return None
        if self.reference_df is None:
            return None

        ref = self.reference_df.copy()
        cur = current_df.copy()

        for df in [ref, cur]:
            df["text_length"] = df["text"].str.len()
            df["word_count"] = df["text"].str.split().str.len()
            df["avg_word_length"] = df["text"].apply(
                lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0
            )

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.report_dir / f"drift_report_{ts}.html"
        report.save_html(str(out_path))
        logger.info(f"Drift report saved: {out_path}")
        return str(out_path)

    def run_full_check(self, current_df, current_predictions=None, reference_predictions=None):
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "samples_checked": len(current_df),
        }

        results["input_drift"] = self.check_input_drift(current_df)

        if current_predictions and reference_predictions:
            results["prediction_drift"] = self.check_prediction_drift(
                reference_predictions, current_predictions
            )
            results["confidence"] = self.check_confidence_degradation(current_predictions)

        input_drifted = results["input_drift"].get("drift_detected", False)
        pred_drifted = results.get("prediction_drift", {}).get("drift_detected", False)
        results["overall_drift"] = input_drifted or pred_drifted

        if results["overall_drift"]:
            logger.warning(f"DRIFT DETECTED | {json.dumps(results, indent=2)}")
        else:
            logger.info("No significant drift detected.")

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / "drift_history.jsonl", "a") as f:
            f.write(json.dumps(results) + "\n")

        results["evidently_report"] = self.generate_evidently_report(current_df)

        return results


class BackgroundMonitor:
    def __init__(self, detector, prediction_log_path, interval_seconds=300):
        self.detector = detector
        self.prediction_log = Path(prediction_log_path)
        self.interval = interval_seconds
        self._seen_lines = 0

    def _read_new_predictions(self):
        if not self.prediction_log.exists():
            return []
        with open(self.prediction_log) as f:
            lines = f.readlines()
        new_lines = lines[self._seen_lines:]
        self._seen_lines = len(lines)
        return [json.loads(l) for l in new_lines if l.strip()]

    def run_forever(self):
        logger.info(f"BackgroundMonitor started (interval={self.interval}s)")
        while True:
            time.sleep(self.interval)
            try:
                preds = self._read_new_predictions()
                if len(preds) < self.detector.min_samples:
                    logger.debug(f"Waiting for more predictions ({len(preds)} so far)...")
                    continue

                current_df = pd.DataFrame([{"text": p["text"]} for p in preds])
                self.detector.run_full_check(current_df, current_predictions=preds)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
