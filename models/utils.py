"""
Shared utilities for model training and evaluation.

Provides:
    - Dataset loading and leakage-safe feature selection
    - Log-based train/validation/test splitting
    - Evaluation metrics (MAE, RMSE, R²) per target component
    - Standard comparison plots
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────────────────────────────────────────────────
# Column definitions
# ──────────────────────────────────────────────────────────────────────

# IMU-only features safe for GNSS-denied inference
IMU_FEATURES = [
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
    "accel_magnitude_m_s2", "gyro_magnitude_rad_s",
]

# Columns that leak GPS information — must NEVER be used as input features
LEAKY_COLUMNS = [
    "gps_vel_n_m_s", "gps_vel_e_m_s", "gps_vel_d_m_s",
    "gps_speed_m_s", "gps_eph_m", "gps_epv_m",
    "gps_satellites_used", "fix_type", "vel_ned_valid",
    "v_xy_valid", "v_z_valid", "xy_valid", "z_valid",
    "dead_reckoning",
    "px4_vx_m_s", "px4_vy_m_s", "px4_vz_m_s",   # part of target computation
]

# Target columns
TARGET_COLUMNS = [
    "target_delta_v_n_m_s",
    "target_delta_v_e_m_s",
    "target_delta_v_d_m_s",
]

TARGET_LABELS = ["Δv_N", "Δv_E", "Δv_D"]

# ──────────────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────────────

# Default log-based split (prevents temporal leakage)
TRAIN_LOGS = [
    "log_51_2026-9-8-18-47-00",
    "log_53_2026-9-8-18-50-18",
]
VAL_LOGS = [
    "log_52_2026-9-8-18-48-50",
]
TEST_LOGS = [
    "log_54_2026-9-8-18-51-28",
]


def load_dataset(path: str = "imu_drift_feature_dataset.csv") -> pd.DataFrame:
    """Load the raw CSV dataset."""
    return pd.read_csv(path)


def log_based_split(df: pd.DataFrame):
    """
    Split the dataframe by log_id into train / val / test.

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
    """
    df_train = df[df["log_id"].isin(TRAIN_LOGS)].copy()
    df_val = df[df["log_id"].isin(VAL_LOGS)].copy()
    df_test = df[df["log_id"].isin(TEST_LOGS)].copy()

    print(f"Train: {len(df_train):,} rows  ({', '.join(TRAIN_LOGS)})")
    print(f"Val:   {len(df_val):,} rows  ({', '.join(VAL_LOGS)})")
    print(f"Test:  {len(df_test):,} rows  ({', '.join(TEST_LOGS)})")

    return df_train, df_val, df_test


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute per-component and mean evaluation metrics.

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (n_samples, 3)

    Returns
    -------
    dict with keys like 'mae_N', 'rmse_N', 'r2_N', ... and mean versions.
    """
    components = ["N", "E", "D"]
    metrics = {}

    for i, comp in enumerate(components):
        metrics[f"mae_{comp}"] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        metrics[f"rmse_{comp}"] = np.sqrt(
            mean_squared_error(y_true[:, i], y_pred[:, i])
        )
        metrics[f"r2_{comp}"] = r2_score(y_true[:, i], y_pred[:, i])

    metrics["mae_mean"] = np.mean([metrics[f"mae_{c}"] for c in components])
    metrics["rmse_mean"] = np.mean([metrics[f"rmse_{c}"] for c in components])
    metrics["r2_mean"] = np.mean([metrics[f"r2_{c}"] for c in components])

    return metrics


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'='*60}")
    print(f"  {'Component':<12} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print(f"  {'-'*42}")
    for comp in ["N", "E", "D"]:
        print(
            f"  Δv_{comp:<8} "
            f"{metrics[f'mae_{comp}']:>10.4f} "
            f"{metrics[f'rmse_{comp}']:>10.4f} "
            f"{metrics[f'r2_{comp}']:>10.4f}"
        )
    print(f"  {'-'*42}")
    print(
        f"  {'Mean':<12} "
        f"{metrics['mae_mean']:>10.4f} "
        f"{metrics['rmse_mean']:>10.4f} "
        f"{metrics['r2_mean']:>10.4f}"
    )
    print(f"{'='*60}\n")


def metrics_to_dataframe(metrics: dict, model_name: str) -> pd.DataFrame:
    """Convert a metrics dict to a single-row DataFrame for comparison."""
    row = {"model": model_name}
    row.update(metrics)
    return pd.DataFrame([row])


# ──────────────────────────────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create output directories if they don't exist."""
    dirs = [
        "results",
        "results/plots",
        "saved_models",
        "configs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

