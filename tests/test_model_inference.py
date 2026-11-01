"""
Unit tests for model loading, prediction shapes, and inference latency.
"""

import sys
import os
import time
import pickle
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import compute_metrics, TARGET_COLUMNS


@pytest.mark.parametrize("model_name,model_file,scaler_file", [
    ("Ridge", "saved_models/ridge_model.pkl", "saved_models/ridge_scaler.pkl"),
    ("Random Forest", "saved_models/rf_model.pkl", "saved_models/rf_scaler.pkl"),
    ("Gradient Boosting", "saved_models/gbm_model.pkl", "saved_models/gbm_scaler.pkl"),
])
def test_model_loading_and_prediction(model_name, model_file, scaler_file):
    """Verify each model can be loaded, transforms features, and predicts 3D outputs."""
    assert os.path.exists(model_file), f"Model checkpoint {model_file} not found"
    assert os.path.exists(scaler_file), f"Scaler file {scaler_file} not found"

    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)

    # 100 samples with 142 features
    X_dummy = np.random.randn(100, 142)
    X_scaled = scaler.transform(X_dummy)

    t0 = time.time()
    y_pred = model.predict(X_scaled)
    latency_ms_per_sample = ((time.time() - t0) / 100) * 1000.0

    # Output assertions
    assert y_pred.shape == (100, 3), f"Expected shape (100, 3), got {y_pred.shape}"
    assert not np.isnan(y_pred).any(), f"{model_name} produced NaN predictions"
    assert latency_ms_per_sample < 10.0, f"{model_name} inference too slow: {latency_ms_per_sample:.2f} ms/sample"


def test_metrics_computation():
    """Verify compute_metrics produces valid MAE, RMSE, and R2 values."""
    y_true = np.array([[0.1, 0.2, -0.3], [0.0, 0.1, -0.2]])
    y_pred = np.array([[0.1, 0.2, -0.3], [0.0, 0.1, -0.2]])

    metrics = compute_metrics(y_true, y_pred)
    assert np.isclose(metrics["rmse_mean"], 0.0)
    assert np.isclose(metrics["mae_mean"], 0.0)
    assert np.isclose(metrics["r2_mean"], 1.0)

