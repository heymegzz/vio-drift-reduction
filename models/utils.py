#!/usr/bin/env python3
"""
utils.py — Shared helpers for VIO drift reduction model pipeline.

Provides:
  - Constants (paths, feature/target lists)
  - Data loading + split helpers
  - Windowed sequence builder (for CNN/LSTM)
  - Metric computation
  - Saving/loading utilities
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(PROJECT_DIR, "feature_engineering_output", "feature_engineered_dataset.csv")
CONFIGS_DIR = os.path.join(PROJECT_DIR, "configs")
SAVED_DIR   = os.path.join(PROJECT_DIR, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature / Target Definitions
# ─────────────────────────────────────────────────────────────────────────────

TARGET_COLS = [
    "target_delta_v_n_m_s",
    "target_delta_v_e_m_s",
    "target_delta_v_d_m_s",
]

TARGET_LABELS = ["Δv_N (North)", "Δv_E (East)", "Δv_D (Down)"]

# All GPS-denied-safe predictors (34 features)
ALL_FEATURE_COLS = [
    # Raw IMU (8)
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "gyro_x_rad_s",  "gyro_y_rad_s",  "gyro_z_rad_s",
    "accel_magnitude_m_s2", "gyro_magnitude_rad_s",
    # Temporal (2)
    "dt_s", "time_since_gap_s",
    # Derivatives (8)
    "jerk_x_m_s3", "jerk_y_m_s3", "jerk_z_m_s3", "jerk_magnitude_m_s3",
    "gyro_dot_x_rad_s2", "gyro_dot_y_rad_s2", "gyro_dot_z_rad_s2", "gyro_dot_magnitude_rad_s2",
    # Rolling statistics (12)
    "accel_roll_mean_x", "accel_roll_mean_y", "accel_roll_mean_z",
    "accel_roll_std_x",  "accel_roll_std_y",  "accel_roll_std_z",
    "gyro_roll_mean_x",  "gyro_roll_mean_y",  "gyro_roll_mean_z",
    "gyro_roll_std_x",   "gyro_roll_std_y",   "gyro_roll_std_z",
    # Cross-axis products (4)
    "accel_xy_product", "accel_xz_product",
    "gyro_xy_product",  "gyro_xz_product",
]

# Smaller feature set for CNN (no rolling stats — CNN computes own context)
CNN_FEATURE_COLS = [
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "gyro_x_rad_s",  "gyro_y_rad_s",  "gyro_z_rad_s",
    "accel_magnitude_m_s2", "gyro_magnitude_rad_s",
    "dt_s", "time_since_gap_s",
    "jerk_x_m_s3", "jerk_y_m_s3", "jerk_z_m_s3", "jerk_magnitude_m_s3",
    "gyro_dot_x_rad_s2", "gyro_dot_y_rad_s2", "gyro_dot_z_rad_s2", "gyro_dot_magnitude_rad_s2",
    "accel_xy_product", "accel_xz_product",
    "gyro_xy_product",  "gyro_xz_product",
]

# Flight IDs
TRAIN_LOGS = ["log_51_2026-9-8-18-47-00", "log_53_2026-9-8-18-50-18"]
VAL_LOG    = "log_52_2026-9-8-18-48-50"
TEST_LOG   = "log_54_2026-9-8-18-51-28"

LARGE_GAP_THRESHOLD = 0.1  # seconds

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load the feature-engineered dataset."""
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Flights: {df['log_id'].unique().tolist()}")
    return df


def split_data(df):
    """Split by flight: train=51+53, val=52, test=54."""
    train_df = df[df["log_id"].isin(TRAIN_LOGS)].copy()
    val_df   = df[df["log_id"] == VAL_LOG].copy()
    test_df  = df[df["log_id"] == TEST_LOG].copy()
    print(f"  Train: {len(train_df)} rows ({train_df['log_id'].unique().tolist()})")
    print(f"  Val:   {len(val_df)}   rows ({val_df['log_id'].unique().tolist()})")
    print(f"  Test:  {len(test_df)}  rows ({test_df['log_id'].unique().tolist()})")
    return train_df, val_df, test_df


def prepare_tabular(df, feature_cols=None, drop_nan=True):
    """
    Extract X, y arrays from a dataframe for tabular models (Ridge, RF, MLP).
    Drops rows with any NaN in features if drop_nan=True.
    Returns (X, y, valid_index) — valid_index preserves original row positions.
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS

    subset = df[feature_cols + TARGET_COLS].copy()
    if drop_nan:
        before = len(subset)
        subset = subset.dropna(subset=feature_cols)
        after = len(subset)
        if before != after:
            print(f"    Dropped {before - after} NaN rows ({100*(before-after)/before:.1f}%)")

    X = subset[feature_cols].values.astype(np.float32)
    y = subset[TARGET_COLS].values.astype(np.float32)
    return X, y, subset.index


# ─────────────────────────────────────────────────────────────────────────────
# Scaler Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(X_train):
    """Fit StandardScaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler, X):
    return scaler.transform(X).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Windowed Sequence Builder (for CNN and LSTM)
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(df, feature_cols, window_size=50, stride=5, gap_threshold=0.1):
    """
    Build overlapping sliding windows for sequence models.

    Rules:
      - Windows do NOT cross log_id boundaries.
      - Windows containing any dt > gap_threshold at an internal step are excluded.
      - Label assigned to the LAST timestep of the window.
      - Past-only: window[t-W+1 : t+1] predicts target at t.

    Returns:
      X_seq : (N_windows, window_size, n_features) float32
      y_seq : (N_windows, 3) float32
      meta  : list of dicts with {log_id, start_idx, end_idx}
    """
    X_list, y_list, meta_list = [], [], []

    for lid in df["log_id"].unique():
        ldf = df[df["log_id"] == lid].reset_index(drop=True)
        n = len(ldf)

        # Fill NaN in dt_s with 0 for gap checking (first row of log has NaN dt)
        dt_vals = ldf["dt_s"].fillna(0.0).values

        feat_vals   = ldf[feature_cols].values.astype(np.float32)
        target_vals = ldf[TARGET_COLS].values.astype(np.float32)

        # Check for NaN in features — mark invalid rows
        feat_nan_mask = np.isnan(feat_vals).any(axis=1)  # True where row has NaN

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size  # exclusive

            # Skip if any feature row in this window has NaN
            if feat_nan_mask[start:end].any():
                continue

            # Skip if any internal step has a large gap
            internal_dt = dt_vals[start+1:end]  # dt at steps start+1..end-1
            if (internal_dt > gap_threshold).any():
                continue

            X_list.append(feat_vals[start:end])          # (W, F)
            y_list.append(target_vals[end - 1])           # label at last step
            meta_list.append({"log_id": lid, "start": start, "end": end - 1})

    if len(X_list) == 0:
        raise ValueError("No valid windows found — check feature columns and data.")

    X_seq = np.stack(X_list, axis=0)   # (N, W, F)
    y_seq = np.stack(y_list, axis=0)   # (N, 3)
    print(f"    Built {X_seq.shape[0]} windows of shape {X_seq.shape[1:]}")
    return X_seq, y_seq, meta_list


def scale_sequences(X_seq_train, X_seq_val, X_seq_test):
    """
    Fit StandardScaler on training sequences (per feature across all timesteps),
    then apply to val and test.
    X_seq: (N, W, F)
    """
    N_tr, W, F = X_seq_train.shape
    scaler = StandardScaler()
    scaler.fit(X_seq_train.reshape(-1, F))  # fit on all (N*W, F) rows

    X_tr  = scaler.transform(X_seq_train.reshape(-1, F)).reshape(N_tr, W, F).astype(np.float32)
    N_v   = X_seq_val.shape[0]
    X_v   = scaler.transform(X_seq_val.reshape(-1, F)).reshape(N_v, W, F).astype(np.float32)
    N_te  = X_seq_test.shape[0]
    X_te  = scaler.transform(X_seq_test.reshape(-1, F)).reshape(N_te, W, F).astype(np.float32)
    return X_tr, X_v, X_te, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, prefix=""):
    """
    Compute per-axis and aggregate metrics.
    y_true, y_pred: (N, 3) arrays
    Returns a dict of metric_name -> value.
    """
    axis_names = ["N", "E", "D"]
    metrics = {}

    maes, rmses, r2s = [], [], []
    for i, ax in enumerate(axis_names):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        p95  = np.percentile(np.abs(y_true[:, i] - y_pred[:, i]), 95)

        metrics[f"{prefix}MAE_{ax}"]  = round(float(mae), 6)
        metrics[f"{prefix}RMSE_{ax}"] = round(float(rmse), 6)
        metrics[f"{prefix}R2_{ax}"]   = round(float(r2), 6)
        metrics[f"{prefix}P95_{ax}"]  = round(float(p95), 6)

        maes.append(mae); rmses.append(rmse); r2s.append(r2)

    metrics[f"{prefix}MAE_mean"]  = round(float(np.mean(maes)), 6)
    metrics[f"{prefix}RMSE_mean"] = round(float(np.mean(rmses)), 6)
    metrics[f"{prefix}R2_mean"]   = round(float(np.mean(r2s)), 6)
    return metrics


def print_metrics(metrics, model_name="Model"):
    print(f"\n{'='*60}")
    print(f"  {model_name} — Metrics")
    print(f"{'='*60}")
    for k, v in metrics.items():
        try:
            print(f"  {k:30s}: {float(v):.6f}")
        except (ValueError, TypeError):
            print(f"  {k:30s}: {v}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_sklearn_model(model, scaler, name):
    """Save a scikit-learn model and scaler as pickle files."""
    model_path  = os.path.join(SAVED_DIR, f"{name}_model.pkl")
    scaler_path = os.path.join(SAVED_DIR, f"{name}_scaler.pkl")
    with open(model_path, "wb")  as f: pickle.dump(model, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    print(f"  Saved model  → {model_path}")
    print(f"  Saved scaler → {scaler_path}")


def load_sklearn_model(name):
    """Load a scikit-learn model and scaler."""
    model_path  = os.path.join(SAVED_DIR, f"{name}_model.pkl")
    scaler_path = os.path.join(SAVED_DIR, f"{name}_scaler.pkl")
    with open(model_path, "rb")  as f: model  = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    return model, scaler


def save_metrics(metrics, name):
    """Append metrics dict to results/metrics_table.csv."""
    row = {"model": name, **metrics}
    path = os.path.join(RESULTS_DIR, "metrics_table.csv")
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        # Remove existing row for this model if present
        df_old = df_old[df_old["model"] != name]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(path, index=False)
    print(f"  Metrics saved → {path}")


def load_config(name):
    """Load a JSON config by model name (e.g. 'ridge', 'rf', 'mlp', 'cnn', 'lstm')."""
    path = os.path.join(CONFIGS_DIR, f"{name}_config.json")
    with open(path) as f:
        return json.load(f)
