"""
Cross-Model Benchmark & Comparison Pipeline  (with baselines)
==============================================================

Aggregates predictions and evaluation metrics across:
  0. Zero-Correction Baseline  (predicted residual = 0)
  1. Mean-Predictor Baseline   (predicted residual = training-set target mean)
  2. Ridge Regression (L2 Regularized Linear Model)
  3. Random Forest (Bagged Decision Tree Ensemble)
  4. Gradient Boosting (Boosted Histogram Tree Ensemble)

Evaluates actual velocity correction:
  corrected_velocity = px4_velocity - predicted_residual

Generates:
  - metrics_table.csv: Comprehensive evaluation table across all axes
  - model_comparison_report.md: Markdown report with statistical analysis
  - correction_evaluation.csv: Whether correction improves over raw PX4
  - Comparison plots

Usage:
    python models/04_comparison.py
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

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_dataset, log_based_split, compute_metrics, print_metrics,
    TARGET_COLUMNS, TARGET_LABELS, ensure_dirs,
)
from src.feature_engineering import engineer_features, get_feature_columns


# ──────────────────────────────────────────────────────────────────────
# Baseline predictors
# ──────────────────────────────────────────────────────────────────────

def zero_correction_baseline(y_test):
    """
    Zero-correction baseline: predict residual = 0.

    This represents "do nothing" — the PX4 velocity is left uncorrected.
    RMSE of zero-correction = RMS of the actual target residuals.
    """
    return np.zeros_like(y_test)


def mean_predictor_baseline(y_train, y_test):
    """
    Mean-predictor baseline: predict residual = training-set target mean.

    This is the constant predictor that minimises MSE on the training set.
    By definition, R² of the mean predictor on the training set = 0.
    """
    train_means = y_train.mean(axis=0)  # shape (3,)
    return np.tile(train_means, (len(y_test), 1))


# ──────────────────────────────────────────────────────────────────────
# Correction evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_correction(df_raw_test, y_pred, model_name):
    """
    Evaluate actual velocity correction quality.

    corrected_v = px4_v - predicted_residual
    original_error = px4_v - gps_v  (= target, by definition)
    corrected_error = corrected_v - gps_v = (px4_v - pred) - gps_v
                    = target - pred

    We compare |original_error| vs |corrected_error| per-axis.
    """
    px4_cols = ["px4_vx_m_s", "px4_vy_m_s", "px4_vz_m_s"]
    gps_cols = ["gps_vel_n_m_s", "gps_vel_e_m_s", "gps_vel_d_m_s"]

    px4_v = df_raw_test[px4_cols].values
    gps_v = df_raw_test[gps_cols].values

    original_error = px4_v - gps_v   # = target
    corrected_v = px4_v - y_pred
    corrected_error = corrected_v - gps_v  # = target - y_pred

    results = {"model": model_name}
    components = ["N", "E", "D"]
    for i, comp in enumerate(components):
        orig_rmse = np.sqrt(np.mean(original_error[:, i] ** 2))
        corr_rmse = np.sqrt(np.mean(corrected_error[:, i] ** 2))
        improvement_pct = (orig_rmse - corr_rmse) / orig_rmse * 100
        results[f"original_rmse_{comp}"] = orig_rmse
        results[f"corrected_rmse_{comp}"] = corr_rmse
        results[f"improvement_pct_{comp}"] = improvement_pct
        results[f"worsens_{comp}"] = corr_rmse > orig_rmse

    # Aggregate: mean of per-axis RMSEs
    results["original_rmse_mean"] = np.mean(
        [results[f"original_rmse_{c}"] for c in components])
    results["corrected_rmse_mean"] = np.mean(
        [results[f"corrected_rmse_{c}"] for c in components])
    improvement = (results["original_rmse_mean"] - results["corrected_rmse_mean"])
    results["improvement_pct_mean"] = improvement / results["original_rmse_mean"] * 100
    results["worsens_overall"] = results["corrected_rmse_mean"] > results["original_rmse_mean"]

    return results


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_metric_bar(df_metrics, metric_prefix, title, ylabel, save_path, palette="viridis"):
    """Plot grouped bar chart for a metric across components and models."""
    fig, ax = plt.subplots(figsize=(11, 6))
    components = ["N", "E", "D", "mean"]
    cols = [f"{metric_prefix}_{c}" for c in components]
    labels = [r"$\Delta v_N$", r"$\Delta v_E$", r"$\Delta v_D$", "Overall Mean"]

    plot_df = pd.DataFrame()
    for col, lbl in zip(cols, labels):
        temp = df_metrics[["model", col]].copy()
        temp["Component"] = lbl
        temp.rename(columns={col: "Value"}, inplace=True)
        plot_df = pd.concat([plot_df, temp])

    sns.barplot(data=plot_df, x="Component", y="Value", hue="model", ax=ax, palette=palette)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Model", fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved comparison plot to {save_path}")


def plot_timeseries_all(models_dict, X_test_dict, y_test, time_test, save_path, max_points=1200):
    """Plot predictions from all models together against the true targets."""
    n = min(len(time_test), max_points)
    t = time_test[:n]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    colors = {
        "Zero Correction": "gray",
        "Mean Predictor": "purple",
        "Ridge": "crimson",
        "Random Forest": "forestgreen",
        "Gradient Boosting": "darkorange",
    }

    for i, (ax, comp) in enumerate(zip(axes, TARGET_LABELS)):
        ax.plot(t, y_test[:n, i], label="Ground Truth", color="black", linewidth=1.2, alpha=0.85)

        for name in models_dict:
            y_pred = models_dict[name]
            ax.plot(t, y_pred[:n, i], label=name, color=colors.get(name, "blue"),
                    linewidth=0.9, alpha=0.75, linestyle="--")

        ax.set_ylabel(f"{comp} (m/s)", fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=8)

    axes[-1].set_xlabel("Elapsed Flight Time (s)", fontsize=11)
    fig.suptitle("PX4–GPS Velocity Residuals — Ground Truth vs. All Models (Test Set: log_54)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved multi-model timeseries to {save_path}")


def plot_correction_comparison(df_raw_test, predictions_dict, time_test, save_path, max_points=1200):
    """Plot original PX4 velocity, corrected velocity, and GPS reference."""
    n = min(len(time_test), max_points)
    t = time_test[:n]

    px4_cols = ["px4_vx_m_s", "px4_vy_m_s", "px4_vz_m_s"]
    gps_cols = ["gps_vel_n_m_s", "gps_vel_e_m_s", "gps_vel_d_m_s"]
    axis_labels = ["North (m/s)", "East (m/s)", "Down (m/s)"]

    px4_v = df_raw_test[px4_cols].values
    gps_v = df_raw_test[gps_cols].values

    # Pick best ML model for the correction plot
    best_model = None
    best_rmse = float("inf")
    for name, pred in predictions_dict.items():
        if name in ("Zero Correction", "Mean Predictor"):
            continue
        corrected_error = (px4_v - pred) - gps_v
        rmse = np.sqrt(np.mean(corrected_error ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = name

    if best_model is None:
        return

    corrected_v = px4_v - predictions_dict[best_model]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, axis_labels)):
        ax.plot(t, gps_v[:n, i], label="GPS Reference", color="black", linewidth=1.0, alpha=0.8)
        ax.plot(t, px4_v[:n, i], label="Original PX4", color="steelblue", linewidth=0.8, alpha=0.7)
        ax.plot(t, corrected_v[:n, i], label=f"Corrected ({best_model})",
                color="crimson", linewidth=0.8, alpha=0.7, linestyle="--")
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel("Elapsed Flight Time (s)", fontsize=11)
    fig.suptitle(f"Velocity Correction Evaluation — {best_model} (Test: log_54)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved correction comparison to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────

def generate_markdown_report(metrics_table, correction_table, save_path):
    """Generate detailed markdown evaluation report."""
    md = []
    md.append("# Phase 1: Model Comparison Report\n")

    md.append("## 1. Executive Summary\n")
    md.append("This report benchmarks baseline and classical ML models for predicting "
              "PX4–GPS velocity residuals (Δv_N, Δv_E, Δv_D) using IMU-derived features.\n")
    md.append("> **Important caveat:** The PX4 velocity estimates were recorded from the "
              "PX4 EKF2 estimator, which may have used GPS aiding during the flight. "
              "Removing GPS *features* from the ML model inputs does not remove GPS "
              "dependence from the recorded PX4 estimator output. This experiment should "
              "be described as *PX4–GPS velocity residual prediction*, not as validated "
              "GNSS-denied or IMU-only drift reduction. See `docs/target_definition.md` "
              "for details.\n")
    md.append("Evaluation on the held-out test flight (`log_54`, 8,306 samples):\n")

    # Metrics table
    try:
        md.append(metrics_table.to_markdown(index=False))
    except (ImportError, Exception):
        headers = list(metrics_table.columns)
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = [header_row, sep_row]
        for _, row in metrics_table.iterrows():
            formatted = []
            for col in headers:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    formatted.append(f"{val:.6f}")
                else:
                    formatted.append(str(val))
            rows.append("| " + " | ".join(formatted) + " |")
        md.append("\n".join(rows))

    md.append("\n\n## 2. Baseline Comparison\n")

    # Check if any model beats zero-correction
    zero_rmse = metrics_table.loc[
        metrics_table["model"] == "Zero Correction", "rmse_mean"].values
    if len(zero_rmse) > 0:
        zero_rmse = zero_rmse[0]
        ml_models = metrics_table[~metrics_table["model"].isin(
            ["Zero Correction", "Mean Predictor"])]
        any_beats_zero = (ml_models["rmse_mean"] < zero_rmse).any()

        if any_beats_zero:
            best = ml_models.loc[ml_models["rmse_mean"].idxmin()]
            md.append(f"- **{best['model']}** achieves lower RMSE ({best['rmse_mean']:.4f} m/s) "
                      f"than zero correction ({zero_rmse:.4f} m/s), indicating the ML model "
                      f"provides meaningful residual prediction.\n")
        else:
            md.append("- **No ML model achieves lower mean RMSE than zero correction** "
                      f"({zero_rmse:.4f} m/s). This means that on this test flight, "
                      "applying any of the trained corrections **worsens** the overall "
                      "velocity estimate compared to leaving PX4 output uncorrected.\n")
            md.append("- This is a common outcome when the residuals are small, noisy, "
                      "and weakly correlated with the available IMU features. "
                      "The result is reported honestly as a negative finding.\n")

    md.append("\n## 3. Velocity Correction Evaluation\n")
    md.append("The actual correction is: `corrected_velocity = px4_velocity - predicted_residual`.\n")
    md.append("Comparing corrected vs. original PX4 velocity against GPS reference:\n")

    if correction_table is not None:
        try:
            md.append(correction_table.to_markdown(index=False))
        except (ImportError, Exception):
            headers = list(correction_table.columns)
            header_row = "| " + " | ".join(headers) + " |"
            sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
            rows = [header_row, sep_row]
            for _, row in correction_table.iterrows():
                formatted = []
                for col in headers:
                    val = row[col]
                    if isinstance(val, (float, np.floating)):
                        formatted.append(f"{val:.4f}")
                    elif isinstance(val, (bool, np.bool_)):
                        formatted.append("Yes" if val else "No")
                    else:
                        formatted.append(str(val))
                rows.append("| " + " | ".join(formatted) + " |")
            md.append("\n".join(rows))

    md.append("\n\n## 4. Methodology Notes\n")
    md.append("- **mean(axis RMSE)** is reported — the mean of per-axis RMSEs, "
              "not pooled RMSE across all axes.\n")
    md.append("- **Bias** is mean(predicted − true): positive = model over-predicts.\n")
    md.append("- Feature importance for Random Forest uses impurity-based importance "
              "(variance reduction for regression), not Gini importance (which applies "
              "to classification).\n")
    md.append("- Temporal features (rolling windows, lags, jerk) are reset at segment "
              "boundaries to avoid crossing temporal gaps > 50 ms.\n")
    md.append("- Jerk is computed as da/dt using actual elapsed time between samples.\n")

    md.append("\n## 5. Artifacts and Reproducibility\n")
    md.append("- Comparison plots and time-series overlays are in `results/plots/`.\n")
    md.append("- Model parameters and scalers are in `configs/` and `saved_models/`.\n")
    md.append("- Test flight `log_54` has already been inspected during development; "
              "it is no longer a pristine unseen test. A new independent recording "
              "is recommended for stronger confirmation.\n")

    with open(save_path, "w") as f:
        f.write("\n".join(md))
    print(f"  Saved comparison report to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    print("=" * 60)
    print("  Cross-Model Benchmark & Comparison Pipeline (with baselines)")
    print("=" * 60)

    # 1. Load data & splits
    print("\n1. Loading dataset and preparing splits...")
    df_raw = load_dataset()
    df = engineer_features(df_raw.copy(), verbose=False)
    feature_cols = get_feature_columns(df)
    df_train, df_val, df_test = log_based_split(df)

    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET_COLUMNS].values
    y_train = df_train[TARGET_COLUMNS].values
    time_test = df_test["time_s"].values

    # Get raw test data (with GPS/PX4 columns) for correction evaluation
    df_raw_test = df_raw[df_raw["log_id"].isin(df_test["log_id"].unique())].copy()
    df_raw_test = df_raw_test.sort_values(["log_id", "timestamp"]).reset_index(drop=True)

    # 2. Baselines
    print("\n2. Evaluating baselines...")

    # Zero-correction baseline
    y_pred_zero = zero_correction_baseline(y_test)
    zero_metrics = compute_metrics(y_test, y_pred_zero)
    print_metrics(zero_metrics, "Zero Correction (predicted residual = 0)")

    # Mean-predictor baseline
    y_pred_mean = mean_predictor_baseline(y_train, y_test)
    mean_metrics = compute_metrics(y_test, y_pred_mean)
    print_metrics(mean_metrics, "Mean Predictor (constant = train-set mean)")

    metrics_list = [
        {**zero_metrics, "model": "Zero Correction"},
        {**mean_metrics, "model": "Mean Predictor"},
    ]
    predictions_dict = {
        "Zero Correction": y_pred_zero,
        "Mean Predictor": y_pred_mean,
    }

    # 3. Load trained ML models
    print("\n3. Loading trained ML models...")
    model_keys = [
        ("Ridge", "saved_models/ridge_model.pkl", "saved_models/ridge_scaler.pkl"),
        ("Random Forest", "saved_models/rf_model.pkl", "saved_models/rf_scaler.pkl"),
        ("Gradient Boosting", "saved_models/gbm_model.pkl", "saved_models/gbm_scaler.pkl"),
    ]

    for name, m_path, s_path in model_keys:
        if not (os.path.exists(m_path) and os.path.exists(s_path)):
            print(f"   [Warning] {name} artifact not found at {m_path}")
            continue

        with open(m_path, "rb") as f:
            model = pickle.load(f)
        with open(s_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"   Loaded {name}")

        X_s = scaler.transform(X_test)
        y_pred = model.predict(X_s)

        m = compute_metrics(y_test, y_pred)
        m["model"] = name
        metrics_list.append(m)
        predictions_dict[name] = y_pred
        print_metrics(m, f"{name} (Test)")

    # 4. Build metrics table
    df_metrics = pd.DataFrame(metrics_list)
    cols_order = [
        "model", "rmse_mean", "mae_mean", "r2_mean", "bias_mean",
        "rmse_N", "rmse_E", "rmse_D",
        "mae_N", "mae_E", "mae_D",
        "r2_N", "r2_E", "r2_D",
        "bias_N", "bias_E", "bias_D",
    ]
    df_metrics = df_metrics[[c for c in cols_order if c in df_metrics.columns]]
    df_metrics = df_metrics.sort_values("rmse_mean", ascending=True)

    print("\n" + "=" * 60)
    print("  Benchmark Summary (Test Set: log_54)")
    print("=" * 60)
    print(df_metrics[["model", "rmse_mean", "mae_mean", "r2_mean", "bias_mean"]].to_string(index=False))

    # Highlight zero-correction comparison
    zero_rmse = df_metrics.loc[df_metrics["model"] == "Zero Correction", "rmse_mean"].values[0]
    print(f"\n  Zero-correction baseline RMSE: {zero_rmse:.6f} m/s")
    for _, row in df_metrics.iterrows():
        if row["model"] not in ("Zero Correction", "Mean Predictor"):
            delta = row["rmse_mean"] - zero_rmse
            status = "WORSE" if delta > 0 else "BETTER"
            print(f"  {row['model']:>20s}: {row['rmse_mean']:.6f} m/s "
                  f"({status} by {abs(delta):.6f} m/s)")

    # 5. Save metrics table
    df_metrics.to_csv("results/models/metrics_table.csv", index=False)
    print("\n  Saved results/models/metrics_table.csv")

    # 6. Correction evaluation
    print("\n4. Evaluating velocity correction quality...")
    correction_results = []
    for name, y_pred in predictions_dict.items():
        result = evaluate_correction(df_raw_test, y_pred, name)
        correction_results.append(result)
        worsens = result["worsens_overall"]
        print(f"   {name:>20s}: corrected RMSE = {result['corrected_rmse_mean']:.6f} m/s "
              f"(original = {result['original_rmse_mean']:.6f} m/s) "
              f"{'⚠ WORSENS' if worsens else '✓ IMPROVES'}")

    df_correction = pd.DataFrame(correction_results)
    corr_cols = ["model", "original_rmse_mean", "corrected_rmse_mean", "improvement_pct_mean",
                 "worsens_overall"]
    for c in ["original_rmse_N", "corrected_rmse_N", "improvement_pct_N",
              "original_rmse_E", "corrected_rmse_E", "improvement_pct_E",
              "original_rmse_D", "corrected_rmse_D", "improvement_pct_D"]:
        if c in df_correction.columns:
            corr_cols.append(c)
    df_correction = df_correction[[c for c in corr_cols if c in df_correction.columns]]
    df_correction.to_csv("results/models/correction_evaluation.csv", index=False)
    print("  Saved results/models/correction_evaluation.csv")

    # 7. Visualizations
    print("\n5. Generating benchmark comparison plots...")
    plot_metric_bar(df_metrics, "rmse",
                    "RMSE Comparison — All Models Including Baselines (m/s)",
                    "RMSE (m/s)", "results/plots/comparison_rmse_bar.png", palette="crest")
    plot_metric_bar(df_metrics, "mae",
                    "MAE Comparison — All Models Including Baselines (m/s)",
                    "MAE (m/s)", "results/plots/comparison_mae_bar.png", palette="flare")
    plot_timeseries_all(predictions_dict, None, y_test, time_test,
                        "results/plots/comparison_timeseries_all_models.png")

    # Correction comparison plot
    plot_correction_comparison(df_raw_test, predictions_dict, time_test,
                               "results/plots/correction_velocity_comparison.png")

    # 8. Generate markdown report
    generate_markdown_report(df_metrics, df_correction,
                             "results/models/model_comparison_report.md")

    print(f"\n{'='*60}")
    print("  Phase 1 Benchmark Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
