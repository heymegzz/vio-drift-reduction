"""
Unit tests for model loading, prediction shapes, and inference latency using standard library unittest.
"""

import sys
import os
import time
import pickle
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import compute_metrics, TARGET_COLUMNS


class TestModelInference(unittest.TestCase):

    def _test_single_model(self, model_name, model_file, scaler_file):
        self.assertTrue(os.path.exists(model_file), f"Model checkpoint {model_file} not found")
        self.assertTrue(os.path.exists(scaler_file), f"Scaler file {scaler_file} not found")

        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)

        # 1000 samples with 142 features (standard flight batch)
        X_dummy = np.random.randn(1000, 142)
        X_scaled = scaler.transform(X_dummy)

        # Warmup
        _ = model.predict(X_scaled[:10])

        t0 = time.time()
        y_pred = model.predict(X_scaled)
        latency_ms_per_sample = ((time.time() - t0) / 1000) * 1000.0

        self.assertEqual(y_pred.shape, (1000, 3))
        self.assertFalse(np.isnan(y_pred).any())
        self.assertLess(latency_ms_per_sample, 50.0)

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


if __name__ == "__main__":
    unittest.main()
