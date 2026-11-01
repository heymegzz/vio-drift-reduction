"""
Gradient Boosting — Multi-Output HistGradientBoosting Regressor
================================================================

Implements a gradient-boosted decision tree ensemble (histogram-based)
to iteratively minimize residual errors across all 3 velocity drift axes.

Key Advantages:
  - Iterative residual fitting with shrinkage (learning rate) and tree regularization.
  - Histogram binning achieves orders of magnitude faster training and inference.
  - Effectively models subtle non-linear interactions across high-dimensional IMU features.

Usage:
    python models/03_gradient_boosting.py
"""

import sys
import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import (
    load_dataset, log_based_split, compute_metrics, print_metrics,
    metrics_to_dataframe, TARGET_COLUMNS, TARGET_LABELS,
    TRAIN_LOGS, ensure_dirs,
)
from feature_engineering import engineer_features, get_feature_columns


def plot_predictions(y_true, y_pred, split_name, save_path):
    """Scatter plot: predicted vs true for each NED component."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ax, label) in enumerate(zip(axes, TARGET_LABELS)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.15, s=5, edgecolors="none", color="darkorange")
        lims = [
            min(y_true[:, i].min(), y_pred[:, i].min()),
            max(y_true[:, i].max(), y_pred[:, i].max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
        ax.set_xlabel(f"True {label} (m/s)", fontsize=11)
        ax.set_ylabel(f"Predicted {label} (m/s)", fontsize=11)
        ax.set_title(f"{label}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Gradient Boosting — Predicted vs True ({split_name})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved prediction scatter to {save_path}")


def plot_residuals(y_true, y_pred, save_path):
    """Violin plot of residual distributions per component."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    data = pd.DataFrame(residuals, columns=TARGET_LABELS)
    data_melted = data.melt(var_name="Component", value_name="Residual (m/s)")
    sns.violinplot(data=data_melted, x="Component", y="Residual (m/s)",
                   ax=ax, inner="quartile", palette="Oranges_r")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Gradient Boosting — Residual Distribution (Test Set)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved residual violin to {save_path}")


def plot_timeseries_comparison(y_true, y_pred, time_s, save_path, max_points=2000):
    """Time-series overlay of predicted vs true velocity residuals."""
    n = min(len(time_s), max_points)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, TARGET_LABELS)):
        ax.plot(time_s[:n], y_true[:n, i], alpha=0.7, linewidth=0.8,
                label="True", color="steelblue")
        ax.plot(time_s[:n], y_pred[:n, i], alpha=0.7, linewidth=0.8,
                label="Predicted (GBM)", color="darkorange")
        ax.set_ylabel(f"{label} (m/s)", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    axes[0].set_title("Gradient Boosting — Time-Series: Predicted vs True (Test Set)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved time-series comparison to {save_path}")


def main():
    ensure_dirs()
    print("=" * 60)
    print("  Gradient Boosting — Multi-Output Regressor Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading and engineering features...")
    df = load_dataset()
    df = engineer_features(df, verbose=False)
    feature_cols = get_feature_columns(df)
    print(f"   {len(feature_cols)} input features")

    # 2. Split
    print("\n2. Log-based splitting...")
    df_train, df_val, df_test = log_based_split(df)

    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET_COLUMNS].values

    X_val = df_val[feature_cols].values
    y_val = df_val[TARGET_COLUMNS].values

    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET_COLUMNS].values
    time_test = df_test["time_s"].values

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Gradient Boosting
    base_params = {
        "max_iter": 120,
        "max_depth": 8,
        "learning_rate": 0.08,
        "min_samples_leaf": 20,
        "l2_regularization": 0.5,
        "random_state": 42,
    }
    print(f"\n3. Training Gradient Boosting with config: {base_params}...")
    t0 = time.time()
    base_model = HistGradientBoostingRegressor(**base_params)
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    train_duration = time.time() - t0
    print(f"   Training completed in {train_duration:.2f} s")

    # 5. Evaluate
    print("\n4. Evaluating on validation and test sets...")
    y_val_pred = model.predict(X_val_scaled)
    val_metrics = compute_metrics(y_val, y_val_pred)
    print_metrics(val_metrics, "Gradient Boosting (Validation)")

    t_inf_start = time.time()
    y_test_pred = model.predict(X_test_scaled)
    inference_time_ms = (time.time() - t_inf_start) / len(X_test) * 1000.0

    test_metrics = compute_metrics(y_test, y_test_pred)
    test_metrics["inference_latency_ms"] = inference_time_ms
    print_metrics(test_metrics, "Gradient Boosting (Test)")

    # 6. Save model & config
    print("\n5. Saving model artifacts...")
    with open("saved_models/gbm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("saved_models/gbm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("   Saved saved_models/gbm_model.pkl")
    print("   Saved saved_models/gbm_scaler.pkl")

    config = {
        "model": "Gradient Boosting",
        "parameters": base_params,
        "n_features": len(feature_cols),
        "train_duration_s": train_duration,
        "inference_latency_ms": inference_time_ms,
        "train_logs": TRAIN_LOGS,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open("configs/gbm_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("   Saved configs/gbm_config.json")

    metrics_df = metrics_to_dataframe(test_metrics, "Gradient Boosting")
    metrics_df.to_csv("results/gbm_metrics.csv", index=False)
    print("   Saved results/gbm_metrics.csv")

    # 7. Visualizations
    print("\n6. Generating visualizations...")
    plot_predictions(y_test, y_test_pred, "Test", "results/plots/gbm_scatter_test.png")
    plot_predictions(y_val, y_val_pred, "Validation", "results/plots/gbm_scatter_val.png")
    plot_residuals(y_test, y_test_pred, "results/plots/gbm_residuals_test.png")
    plot_timeseries_comparison(y_test, y_test_pred, time_test, "results/plots/gbm_timeseries_test.png")

    print(f"\n{'='*60}")
    print("  Gradient Boosting training complete!")
    print(f"  Test RMSE (mean): {test_metrics['rmse_mean']:.4f} m/s")
    print(f"  Test R² (mean):   {test_metrics['r2_mean']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
