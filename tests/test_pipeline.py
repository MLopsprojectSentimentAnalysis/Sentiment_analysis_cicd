"""
Unit and integration tests for the MLOps pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import clean_text, clean_dataframe
from src.monitoring.drift_detector import (
    DriftDetector, _psi, _js_divergence, _ks_test,
)


class TestTextCleaning:
    def test_removes_html_tags(self):
        assert "<b>" not in clean_text("Hello <b>world</b>")

    def test_removes_urls(self):
        assert "http" not in clean_text("Visit http://example.com today")

    def test_collapses_whitespace(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_clean_dataframe_drops_short(self):
        df = pd.DataFrame({"text": ["ok", "A" * 50, ""], "label": [0, 1, 0]})
        cleaned = clean_dataframe(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["text"] == "A" * 50


class TestDriftMetrics:
    def _same_dist(self, n=1000):
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, n), rng.normal(0, 1, n)

    def _shifted_dist(self, n=1000):
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, n), rng.normal(3, 1, n)

    def test_psi_near_zero_same_dist(self):
        ref, cur = self._same_dist(n=5000)
        assert _psi(ref, cur) < 0.1

    def test_psi_high_shifted_dist(self):
        ref, cur = self._shifted_dist()
        psi_shifted = abs(_psi(ref, cur))
        ref_same, cur_same = self._same_dist(n=5000)
        psi_same = abs(_psi(ref_same, cur_same))
        # Shifted should be meaningfully larger than same-dist baseline
        assert psi_shifted > psi_same * 1.5

    def test_js_divergence_same_dist(self):
        ref, cur = self._same_dist(n=5000)
        assert _js_divergence(ref, cur) < 0.1

    def test_js_divergence_shifted_dist(self):
        ref, cur = self._shifted_dist()
        assert _js_divergence(ref, cur) > 0.1

    def test_ks_test_returns_tuple(self):
        ref, cur = self._same_dist()
        stat, p = _ks_test(ref, cur)
        assert 0 <= stat <= 1
        assert 0 <= p <= 1

    def test_ks_test_detects_shift(self):
        ref, cur = self._shifted_dist()
        _, p = _ks_test(ref, cur)
        assert p < 0.05


class TestDriftDetector:
    @pytest.fixture
    def tmp_config(self, tmp_path):
        ref_path = tmp_path / "reference.parquet"
        ref_data = pd.DataFrame({
            "text": ["This is a great movie! " * 5] * 500 +
                    ["Absolutely terrible experience." * 3] * 500,
            "label": [1] * 500 + [0] * 500,
        })
        ref_data.to_parquet(ref_path, index=False)

        config = {
            "monitoring": {
                "reference_data_path": str(ref_path),
                "psi_threshold": 0.2,
                "js_divergence_threshold": 0.1,
                "prediction_drift_threshold": 0.15,
                "min_samples_for_drift": 10,
            }
        }
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return str(config_path)

    def test_no_drift_same_distribution(self, tmp_config):
        detector = DriftDetector(tmp_config)
        current_df = pd.DataFrame({
            "text": ["This is a great movie! " * 5] * 50 +
                    ["Absolutely terrible experience." * 3] * 50
        })
        result = detector.check_input_drift(current_df)
        assert "feature_drift" in result

    def test_drift_detected_on_short_texts(self, tmp_config):
        detector = DriftDetector(tmp_config)
        current_df = pd.DataFrame({"text": ["ok"] * 50 + ["bad"] * 50})
        result = detector.check_input_drift(current_df)
        assert result.get("drift_detected") is True

    def test_skip_if_too_few_samples(self, tmp_config):
        detector = DriftDetector(tmp_config)
        current_df = pd.DataFrame({"text": ["hello"] * 3})
        result = detector.check_input_drift(current_df)
        assert result.get("skipped") is True

    def test_prediction_drift(self, tmp_config):
        detector = DriftDetector(tmp_config)
        ref_preds = [{"label": "positive"}] * 50 + [{"label": "negative"}] * 50
        cur_preds = [{"label": "positive"}] * 90 + [{"label": "negative"}] * 10
        result = detector.check_prediction_drift(ref_preds, cur_preds)
        assert "js_divergence" in result
        assert isinstance(result["drift_detected"], bool)

    def test_confidence_degradation(self, tmp_config):
        detector = DriftDetector(tmp_config)
        preds = [{"confidence": 0.6}] * 30 + [{"confidence": 0.95}] * 70
        result = detector.check_confidence_degradation(preds)
        assert result["degradation_alert"] is True

    def test_no_confidence_degradation(self, tmp_config):
        detector = DriftDetector(tmp_config)
        preds = [{"confidence": 0.95}] * 100
        result = detector.check_confidence_degradation(preds)
        assert result["degradation_alert"] is False


class TestAPISchemas:
    def test_predict_request_requires_text(self):
        from pydantic import ValidationError
        from src.api.app import PredictRequest
        with pytest.raises(ValidationError):
            PredictRequest()

    def test_batch_request_requires_texts(self):
        from pydantic import ValidationError
        from src.api.app import BatchPredictRequest
        with pytest.raises(ValidationError):
            BatchPredictRequest()

    def test_valid_predict_request(self):
        from src.api.app import PredictRequest
        r = PredictRequest(text="This is great!")
        assert r.text == "This is great!"
