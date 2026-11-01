"""
Ridge Regression — Multi-Output Baseline Model
================================================

Implements regularized linear regression (L2 penalty) as the baseline
model for predicting velocity drift residuals from engineered IMU features.

Why Ridge Regression?
  - Provides an interpretable linear baseline against which non-linear
    models (Random Forest, Gradient Boosting) can be compared.
  - L2 regularization handles multicollinearity from correlated IMU
    features and rolling-window statistics.
  - Multi-output formulation predicts all three NED components jointly.
  - Extremely fast training and inference — suitable for real-time
    onboard correction.

Mathematical formulation:
  Minimise  ‖Xw − y‖² + α‖w‖²
  where α is the regularization strength.

Hyperparameter tuning:
  α is selected via leave-one-log-out cross-validation on the training
  logs to respect temporal structure.

Usage:
    python models/01_ridge_regression.py
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import (
    load_dataset, log_based_split, compute_metrics, print_metrics,
    metrics_to_dataframe, TARGET_COLUMNS, TARGET_LABELS,
    TRAIN_LOGS, ensure_dirs,
)
from feature_engineering import engineer_features, get_feature_columns


# ──────────────────────────────────────────────────────────────────────
# Hyperparameter search space
# ──────────────────────────────────────────────────────────────────────

ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]


def leave_one_log_out_cv(X_train_full, y_train_full, log_ids_train,
                         alphas):
    """
    Custom CV respecting log boundaries.

    For each fold, one training log is held out as the validation fold.
    This prevents temporal leakage during hyperparameter selection.
    """
    unique_logs = sorted(log_ids_train.unique())
    print(f"\n  Leave-one-log-out CV across {len(unique_logs)} training logs:")

    results = []
    for alpha in alphas:
        fold_scores = []
        for held_out_log in unique_logs:
            # Split
            train_mask = log_ids_train != held_out_log
            val_mask = log_ids_train == held_out_log

            X_tr, y_tr = X_train_full[train_mask], y_train_full[train_mask]
            X_va, y_va = X_train_full[val_mask], y_train_full[val_mask]

            # Scale
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            # Fit
            model = Ridge(alpha=alpha)
            model.fit(X_tr_s, y_tr)

            # Evaluate
            y_pred = model.predict(X_va_s)
            rmse = np.sqrt(np.mean((y_va - y_pred) ** 2))
            fold_scores.append(rmse)

        mean_rmse = np.mean(fold_scores)
        std_rmse = np.std(fold_scores)
        results.append({
            "alpha": alpha,
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "fold_rmses": fold_scores,
        })
        print(f"    α={alpha:<8} → RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")

    return results


def plot_alpha_search(results: list, save_path: str):
    """Plot RMSE vs alpha for hyperparameter selection."""
    alphas = [r["alpha"] for r in results]
    means = [r["mean_rmse"] for r in results]
    stds = [r["std_rmse"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(alphas, means, yerr=stds, marker="o", capsize=5,
                linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Regularization α (log scale)", fontsize=12)
    ax.set_ylabel("Mean RMSE (m/s)", fontsize=12)
    ax.set_title("Ridge Regression — Leave-One-Log-Out CV", fontsize=14)
    ax.grid(True, alpha=0.3)

    best_idx = np.argmin(means)
    ax.axvline(alphas[best_idx], color="red", linestyle="--", alpha=0.5,
               label=f"Best α = {alphas[best_idx]}")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved alpha search plot to {save_path}")


def plot_predictions(y_true, y_pred, split_name, save_path):
    """Scatter plot: predicted vs true for each NED component."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ax, label) in enumerate(zip(axes, TARGET_LABELS)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.15, s=5,
                   edgecolors="none")
        lims = [
            min(y_true[:, i].min(), y_pred[:, i].min()),
            max(y_true[:, i].max(), y_pred[:, i].max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel(f"True {label} (m/s)", fontsize=11)
        ax.set_ylabel(f"Predicted {label} (m/s)", fontsize=11)
        ax.set_title(f"{label}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Ridge Regression — Predicted vs True ({split_name})",
                 fontsize=14, y=1.02)
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
                   ax=ax, inner="quartile", palette="Set2")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Ridge Regression — Residual Distribution (Test Set)",
                 fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved residual violin to {save_path}")


def plot_coefficients(model, feature_names, save_path, top_n=20):
    """Bar plot of the top-N most important coefficients (by abs value)."""
    # Average absolute coefficients across the 3 targets
    coef_abs_mean = np.abs(model.coef_).mean(axis=0)
    top_indices = np.argsort(coef_abs_mean)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_indices))
    ax.barh(y_pos, coef_abs_mean[top_indices], color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |Coefficient| across targets", fontsize=11)
    ax.set_title(f"Ridge Regression — Top {top_n} Feature Coefficients",
                 fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved coefficient plot to {save_path}")


def plot_timeseries_comparison(y_true, y_pred, time_s, save_path,
                               max_points=2000):
    """Time-series overlay of predicted vs true velocity residuals."""
    n = min(len(time_s), max_points)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, TARGET_LABELS)):
        ax.plot(time_s[:n], y_true[:n, i], alpha=0.7, linewidth=0.8,
                label="True", color="steelblue")
        ax.plot(time_s[:n], y_pred[:n, i], alpha=0.7, linewidth=0.8,
                label="Predicted", color="coral")
        ax.set_ylabel(f"{label} (m/s)", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    axes[0].set_title(
        "Ridge Regression — Time-Series: Predicted vs True (Test Set)",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved time-series comparison to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    print("=" * 60)
    print("  Ridge Regression — Multi-Output Baseline")
    print("=" * 60)

    # 1. Load and engineer features
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
    log_ids_train = df_train["log_id"]

    X_val = df_val[feature_cols].values
    y_val = df_val[TARGET_COLUMNS].values

    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET_COLUMNS].values
    time_test = df_test["time_s"].values

    # 3. Hyperparameter search
    print("\n3. Hyperparameter search (leave-one-log-out CV)...")
    cv_results = leave_one_log_out_cv(
        X_train, y_train, log_ids_train, ALPHA_CANDIDATES
    )
    best_result = min(cv_results, key=lambda r: r["mean_rmse"])
    best_alpha = best_result["alpha"]
    print(f"\n   Best α = {best_alpha} "
          f"(CV RMSE = {best_result['mean_rmse']:.4f})")

    plot_alpha_search(cv_results, "results/plots/ridge_alpha_search.png")

    # 4. Train final model
    print(f"\n4. Training final model with α = {best_alpha}...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate
    print("\n5. Evaluating on all splits...")

    # Validation set
    y_val_pred = model.predict(X_val_scaled)
    val_metrics = compute_metrics(y_val, y_val_pred)
    print_metrics(val_metrics, "Ridge Regression (Validation)")

    # Test set
    y_test_pred = model.predict(X_test_scaled)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print_metrics(test_metrics, "Ridge Regression (Test)")

    # 6. Save model artifacts
    print("6. Saving model artifacts...")
    with open("saved_models/ridge_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("saved_models/ridge_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("   Saved saved_models/ridge_model.pkl")
    print("   Saved saved_models/ridge_scaler.pkl")

    # Save config
    config = {
        "model": "Ridge Regression",
        "alpha": best_alpha,
        "n_features": len(feature_cols),
        "n_parameters": int(np.prod(model.coef_.shape) + model.intercept_.shape[0]),
        "train_logs": TRAIN_LOGS,
        "cv_best_rmse": best_result["mean_rmse"],
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
    }
    with open("configs/ridge_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("   Saved configs/ridge_config.json")

    # Save metrics
    metrics_df = metrics_to_dataframe(test_metrics, "Ridge Regression")
    metrics_df.to_csv("results/ridge_metrics.csv", index=False)
    print("   Saved results/ridge_metrics.csv")

    # Save coefficients
    coef_df = pd.DataFrame(
        model.coef_.T,
        columns=["coef_N", "coef_E", "coef_D"],
        index=feature_cols,
    )
    coef_df["abs_mean"] = coef_df.abs().mean(axis=1)
    coef_df = coef_df.sort_values("abs_mean", ascending=False)
    coef_df.to_csv("results/ridge_coefficients.csv")
    print("   Saved results/ridge_coefficients.csv")

    # 7. Generate plots
    print("\n7. Generating visualizations...")
    plot_predictions(y_test, y_test_pred, "Test", 
                     "results/plots/ridge_scatter_test.png")
    plot_predictions(y_val, y_val_pred, "Validation",
                     "results/plots/ridge_scatter_val.png")
    plot_residuals(y_test, y_test_pred,
                   "results/plots/ridge_residuals_test.png")
    plot_coefficients(model, feature_cols,
                      "results/plots/ridge_coefficients.png")
    plot_timeseries_comparison(y_test, y_test_pred, time_test,
                              "results/plots/ridge_timeseries_test.png")

    print(f"\n{'='*60}")
    print("  Ridge Regression training complete!")
    print(f"  Test RMSE (mean): {test_metrics['rmse_mean']:.4f} m/s")
    print(f"  Test R² (mean):   {test_metrics['r2_mean']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

