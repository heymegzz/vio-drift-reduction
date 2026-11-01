"""
Unit tests for model loading, prediction shapes, and inference latency using standard library unittest.

Tests validate that saved model artifacts load correctly and produce
valid predictions.  There are NO fallback paths — if a model or scaler
fails to load, the test fails immediately so the problem is surfaced.
"""

import sys
import os
import time
import pickle
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import compute_metrics, TARGET_COLUMNS


class TestModelInference(unittest.TestCase):

    def _test_single_model(self, model_name, model_file, scaler_file):
        self.assertTrue(
            os.path.exists(model_file),
            f"Model checkpoint {model_file} not found — "
            f"run the training script to generate it."
        )
        self.assertTrue(
            os.path.exists(scaler_file),
            f"Scaler file {scaler_file} not found — "
            f"run the training script to generate it."
        )

        # 1000 samples with 142 features (standard flight batch)
        X_dummy = np.random.randn(1000, 142)

        # Load scaler — fail hard on any error
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X_dummy)

        # Load model — fail hard on any error
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # Warmup
        _ = model.predict(X_scaled[:10])

        # Timed inference
        t0 = time.time()
        y_pred = model.predict(X_scaled)
        elapsed_s = time.time() - t0
        latency_ms_per_sample = (elapsed_s / len(X_scaled)) * 1000.0

        self.assertEqual(y_pred.shape, (1000, 3),
                         f"{model_name} prediction shape mismatch")
        self.assertFalse(np.isnan(y_pred).any(),
                         f"{model_name} produced NaN predictions")
        self.assertLess(latency_ms_per_sample, 50.0,
                        f"{model_name} per-sample latency too high: "
                        f"{latency_ms_per_sample:.3f} ms")

    def test_ridge_inference(self):
        self._test_single_model("Ridge", "saved_models/ridge_model.pkl", "saved_models/ridge_scaler.pkl")

    def test_random_forest_inference(self):
        self._test_single_model("Random Forest", "saved_models/rf_model.pkl", "saved_models/rf_scaler.pkl")

    def test_gradient_boosting_inference(self):
        self._test_single_model("Gradient Boosting", "saved_models/gbm_model.pkl", "saved_models/gbm_scaler.pkl")

    def test_metrics_computation(self):
        y_true = np.array([[0.1, 0.2, -0.3], [0.0, 0.1, -0.2]])
        y_pred = np.array([[0.1, 0.2, -0.3], [0.0, 0.1, -0.2]])

        metrics = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["rmse_mean"], 0.0)
        self.assertAlmostEqual(metrics["mae_mean"], 0.0)
        self.assertAlmostEqual(metrics["r2_mean"], 1.0)
        self.assertAlmostEqual(metrics["bias_mean"], 0.0)

    def test_metrics_bias(self):
        """Verify bias is computed as mean(pred - true)."""
        y_true = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        y_pred = np.array([[0.1, -0.2, 0.3], [0.1, -0.2, 0.3]])

        metrics = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["bias_N"], 0.1)
        self.assertAlmostEqual(metrics["bias_E"], -0.2)
        self.assertAlmostEqual(metrics["bias_D"], 0.3)


if __name__ == "__main__":
    unittest.main()
