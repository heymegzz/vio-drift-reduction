#!/usr/bin/env python3
"""
06_comparison.py
================
Loads all trained models, runs inference on the held-out test flight (log_54),
and generates:
  - results/metrics_table.csv (final)
  - results/plots/ (10+ plots)
  - results/model_comparison_report.md
"""

import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
try:
    import seaborn as sns
except ImportError:
    sns = None


def _load_module(name, path):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BASE = os.path.dirname(__file__)
_mlp_mod  = _load_module("_mlp_mod",  os.path.join(_BASE, "03_mlp.py"))
_cnn_mod  = _load_module("_cnn_mod",  os.path.join(_BASE, "04_cnn1d.py"))
_lstm_mod = _load_module("_lstm_mod", os.path.join(_BASE, "05_lstm.py"))

DriftMLP   = _mlp_mod.DriftMLP
DriftCNN1D = _cnn_mod.DriftCNN1D
DriftLSTM  = _lstm_mod.DriftLSTM

import torch
import torch.nn as nn

from utils import (
    load_data, split_data, prepare_tabular, build_windows, scale_sequences,
    apply_scaler, compute_metrics, ALL_FEATURE_COLS, CNN_FEATURE_COLS,
    TARGET_COLS, TARGET_LABELS, SAVED_DIR, RESULTS_DIR, PLOTS_DIR
)

# Model classes already loaded via dynamic import above

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Ridge":       "#4ECDC4",
    "RandomForest":"#FF6B6B",
    "MLP":         "#A78BFA",
    "CNN1D":       "#FB923C",
    "LSTM":        "#34D399",
    "truth":       "#94A3B8",
}
MODEL_ORDER = ["Ridge", "RandomForest", "MLP", "CNN1D", "LSTM"]

# ─────────────────────────────────────────────────────────────────────────────
# Helper — styled figure
# ─────────────────────────────────────────────────────────────────────────────
def styled_fig(figsize=(14, 6)):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=figsize, facecolor="#0F1117")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Load all model predictions on test set
# ─────────────────────────────────────────────────────────────────────────────

def load_all_predictions(test_df):
    """Run inference for all 5 models and return dict of predictions."""
    preds = {}
    y_true_tab = None

    # ── Ridge ─────────────────────────────────────────────────────────────────
    print("\n[Ridge] Loading and predicting...")
    with open(os.path.join(SAVED_DIR, "ridge_model.pkl"), "rb") as f:
        ridge_model = pickle.load(f)
    with open(os.path.join(SAVED_DIR, "ridge_scaler.pkl"), "rb") as f:
        ridge_scaler = pickle.load(f)
    X_te, y_te_tab, idx_tab = prepare_tabular(test_df)
    X_te_s = apply_scaler(ridge_scaler, X_te)
    preds["Ridge"] = {"y_pred": ridge_model.predict(X_te_s), "y_true": y_te_tab, "idx": idx_tab}
    y_true_tab = y_te_tab

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("[RandomForest] Loading and predicting...")
    with open(os.path.join(SAVED_DIR, "rf_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(SAVED_DIR, "rf_scaler.pkl"), "rb") as f:
        rf_scaler = pickle.load(f)
    X_te_s = apply_scaler(rf_scaler, X_te)
    preds["RandomForest"] = {"y_pred": rf_model.predict(X_te_s), "y_true": y_te_tab, "idx": idx_tab}

    # ── MLP ───────────────────────────────────────────────────────────────────
    print("[MLP] Loading and predicting...")
    with open(os.path.join(SAVED_DIR, "mlp_scaler.pkl"), "rb") as f:
        mlp_scaler = pickle.load(f)
    X_te_s = apply_scaler(mlp_scaler, X_te)
    mlp_model = DriftMLP(X_te_s.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(torch.load(os.path.join(SAVED_DIR, "mlp_model.pt"),
                                          map_location=DEVICE))
    mlp_model.eval()
    with torch.no_grad():
        y_pred_mlp = mlp_model(torch.tensor(X_te_s).to(DEVICE)).cpu().numpy()
    preds["MLP"] = {"y_pred": y_pred_mlp, "y_true": y_te_tab, "idx": idx_tab}

    # ── CNN1D ─────────────────────────────────────────────────────────────────
    print("[CNN1D] Loading and predicting...")
    X_cnn, y_cnn, _ = build_windows(test_df, CNN_FEATURE_COLS, window_size=50, stride=5)
    with open(os.path.join(SAVED_DIR, "cnn_scaler.pkl"), "rb") as f:
        cnn_scaler = pickle.load(f)
    N, W, F = X_cnn.shape
    X_cnn_s = cnn_scaler.transform(X_cnn.reshape(-1, F)).reshape(N, W, F).astype(np.float32)
    cnn_model = DriftCNN1D(F).to(DEVICE)
    cnn_model.load_state_dict(torch.load(os.path.join(SAVED_DIR, "cnn_model.pt"),
                                          map_location=DEVICE))
    cnn_model.eval()
    with torch.no_grad():
        y_pred_cnn = cnn_model(torch.tensor(X_cnn_s).to(DEVICE)).cpu().numpy()
    preds["CNN1D"] = {"y_pred": y_pred_cnn, "y_true": y_cnn}

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("[LSTM] Loading and predicting...")
    X_lstm, y_lstm, _ = build_windows(test_df, ALL_FEATURE_COLS, window_size=100, stride=5)
    with open(os.path.join(SAVED_DIR, "lstm_scaler.pkl"), "rb") as f:
        lstm_scaler = pickle.load(f)
    N2, W2, F2 = X_lstm.shape
    X_lstm_s = lstm_scaler.transform(X_lstm.reshape(-1, F2)).reshape(N2, W2, F2).astype(np.float32)
    lstm_model = DriftLSTM(F2).to(DEVICE)
    lstm_model.load_state_dict(torch.load(os.path.join(SAVED_DIR, "lstm_model.pt"),
                                           map_location=DEVICE))
    lstm_model.eval()
    with torch.no_grad():
        y_pred_lstm = lstm_model(torch.tensor(X_lstm_s).to(DEVICE)).cpu().numpy()
    preds["LSTM"] = {"y_pred": y_pred_lstm, "y_true": y_lstm}

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_heatmap(metrics_df, outpath):
    """Heatmap of all per-axis metrics across models."""
    fig = styled_fig(figsize=(16, 7))
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0F1117")

    cols = [c for c in metrics_df.columns if any(x in c for x in ["MAE_", "RMSE_", "R2_", "P95_"])
            and c not in ["MAE_mean", "RMSE_mean", "R2_mean"]]

    hm_data = metrics_df.set_index("model")[cols].astype(float)

    # Normalise per column for colouring (lower=better for MAE/RMSE/P95, higher=better for R2)
    norm_data = hm_data.copy()
    for col in cols:
        rng = hm_data[col].max() - hm_data[col].min()
        if rng > 0:
            if "R2" in col:
                norm_data[col] = (hm_data[col] - hm_data[col].min()) / rng
            else:
                norm_data[col] = 1 - (hm_data[col] - hm_data[col].min()) / rng

    cmap = LinearSegmentedColormap.from_list("drift", ["#FF4B4B", "#FFA500", "#00FF88"])
    im = ax.imshow(norm_data.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=45, ha="right",
                        fontsize=9, color="white")
    ax.set_yticks(range(len(hm_data)))
    ax.set_yticklabels(hm_data.index, fontsize=11, color="white")

    # Annotate with actual values
    for i in range(len(hm_data)):
        for j, col in enumerate(cols):
            val = hm_data.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=7.5, color="black" if norm_data.values[i, j] > 0.5 else "white")

    plt.colorbar(im, ax=ax, label="Relative performance (green=best)")
    ax.set_title("Model Performance Heatmap — Test Flight log_54",
                  fontsize=14, color="white", pad=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_rmse_bar(metrics_df, outpath):
    """Bar chart of RMSE per axis and mean."""
    fig = styled_fig(figsize=(14, 5))
    axes_keys = ["RMSE_N", "RMSE_E", "RMSE_D", "RMSE_mean"]
    axes_labels = ["RMSE Δv_N", "RMSE Δv_E", "RMSE Δv_D", "RMSE Mean"]

    for subplot_idx, (key, label) in enumerate(zip(axes_keys, axes_labels)):
        ax = fig.add_subplot(1, 4, subplot_idx + 1)
        ax.set_facecolor("#161B22")
        vals = [float(metrics_df[metrics_df["model"] == m][key].values[0])
                if key in metrics_df.columns else 0
                for m in MODEL_ORDER if m in metrics_df["model"].values]
        models_present = [m for m in MODEL_ORDER if m in metrics_df["model"].values]
        bar_colors = [COLORS[m] for m in models_present]
        bars = ax.bar(range(len(vals)), vals, color=bar_colors, edgecolor="none", width=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(models_present, rotation=30, ha="right", fontsize=8, color="#CBD5E1")
        ax.set_ylabel("RMSE (m/s)", color="#CBD5E1", fontsize=9)
        ax.set_title(label, color="white", fontsize=10)
        ax.tick_params(colors="#CBD5E1")
        ax.spines[:].set_color("#2D3748")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7, color="white")

    fig.suptitle("RMSE Comparison on Held-Out Flight (log_54)", color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_timeseries(preds, test_df, outpath):
    """Time-series prediction overlay for each axis."""
    # Use tabular (Ridge) time index as common x-axis
    ridge_data = preds["Ridge"]
    idx = ridge_data["idx"]
    time_s = test_df.loc[idx, "time_s"].values

    fig = styled_fig(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)

    axis_keys  = [0, 1, 2]
    axis_titles = TARGET_LABELS

    for row, (ak, title) in enumerate(zip(axis_keys, axis_titles)):
        ax = fig.add_subplot(gs[row])
        ax.set_facecolor("#161B22")

        # Ground truth
        ax.plot(time_s, ridge_data["y_true"][:, ak],
                color=COLORS["truth"], lw=1.2, alpha=0.9, label="Ground Truth", zorder=5)

        # Tabular model predictions
        for name in ["Ridge", "RandomForest", "MLP"]:
            if name in preds:
                ax.plot(time_s, preds[name]["y_pred"][:, ak],
                        color=COLORS[name], lw=0.8, alpha=0.7, label=name, zorder=4)

        ax.set_ylabel("m/s", color="#CBD5E1", fontsize=9)
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="#CBD5E1")
        ax.spines[:].set_color("#2D3748")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.3,
                      facecolor="#0F1117", edgecolor="#4A5568", labelcolor="white")

    fig.suptitle("Prediction vs Ground Truth — Test Flight (log_54) — Tabular Models",
                  color="white", fontsize=13, y=1.01)
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_residuals_violin(preds, outpath):
    """Violin plot of absolute residuals per model per axis."""
    fig = styled_fig(figsize=(16, 6))
    axis_names = ["N", "E", "D"]

    for subplot_i, (ai, ax_name) in enumerate(zip([0, 1, 2], axis_names)):
        ax = fig.add_subplot(1, 3, subplot_i + 1)
        ax.set_facecolor("#161B22")

        data, labels, colors_list = [], [], []
        for name in MODEL_ORDER:
            if name in preds:
                residuals = np.abs(preds[name]["y_pred"][:, ai] - preds[name]["y_true"][:, ai])
                data.append(residuals)
                labels.append(name)
                colors_list.append(COLORS[name])

        parts = ax.violinplot(data, positions=range(len(data)), showmedians=True)
        for i, (pc, col) in enumerate(zip(parts["bodies"], colors_list)):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)
        parts["cmaxes"].set_color("#4A5568")
        parts["cmins"].set_color("#4A5568")
        parts["cbars"].set_color("#4A5568")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color="#CBD5E1")
        ax.set_ylabel("|Error| (m/s)", color="#CBD5E1", fontsize=9)
        ax.set_title(f"Δv_{ax_name} Residuals", color="white", fontsize=11)
        ax.tick_params(colors="#CBD5E1")
        ax.spines[:].set_color("#2D3748")

    fig.suptitle("Absolute Residual Distributions — Test Flight (log_54)",
                  color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_scatter_pred_vs_true(preds, outpath):
    """Scatter plots: predicted vs true for each model (mean across axes)."""
    models_avail = [m for m in MODEL_ORDER if m in preds]
    n = len(models_avail)
    fig = styled_fig(figsize=(4 * n, 4))

    for i, name in enumerate(models_avail):
        ax = fig.add_subplot(1, n, i + 1)
        ax.set_facecolor("#161B22")
        y_true = preds[name]["y_true"].ravel()
        y_pred = preds[name]["y_pred"].ravel()
        ax.scatter(y_true, y_pred, c=COLORS[name], s=1, alpha=0.3)
        lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "w--", lw=1, alpha=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("True (m/s)", color="#CBD5E1", fontsize=8)
        ax.set_ylabel("Predicted (m/s)", color="#CBD5E1", fontsize=8)
        ax.set_title(name, color="white", fontsize=10)
        ax.tick_params(colors="#CBD5E1")
        ax.spines[:].set_color("#2D3748")

    fig.suptitle("Predicted vs True — All Axes Combined",
                  color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_training_curves(outpath):
    """Overlay training/val loss curves for MLP, CNN, LSTM."""
    fig = styled_fig(figsize=(15, 4))

    names = ["MLP", "CNN1D", "LSTM"]
    files = ["mlp_training_history.csv", "cnn_training_history.csv", "lstm_training_history.csv"]
    colors_tr = ["#A78BFA", "#FB923C", "#34D399"]
    colors_v  = ["#7C3AED", "#C2410C", "#059669"]

    for i, (name, fname, ct, cv) in enumerate(zip(names, files, colors_tr, colors_v)):
        fpath = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        hist = pd.read_csv(fpath)
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_facecolor("#161B22")
        ax.plot(hist["epoch"], hist["train_loss"], color=ct, lw=1.5, label="Train")
        ax.plot(hist["epoch"], hist["val_loss"],   color=cv, lw=1.5, label="Val", linestyle="--")
        ax.set_xlabel("Epoch", color="#CBD5E1", fontsize=9)
        ax.set_ylabel("MSE Loss", color="#CBD5E1", fontsize=9)
        ax.set_title(f"{name} Training Curve", color="white", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.3, facecolor="#0F1117",
                  edgecolor="#4A5568", labelcolor="white")
        ax.tick_params(colors="#CBD5E1")
        ax.spines[:].set_color("#2D3748")

    fig.suptitle("Deep Learning Training Curves", color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_r2_radar(metrics_df, outpath):
    """Radar chart of R² per axis per model."""
    import matplotlib.patches as mpatches
    categories = ["R²_N", "R²_E", "R²_D"]
    cat_cols    = ["R2_N", "R2_E", "R2_D"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig = styled_fig(figsize=(8, 8))
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#161B22")
    fig.patch.set_facecolor("#0F1117")

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, color="white", fontsize=11)
    ax.set_rlim(-0.1, 1.0)
    ax.set_rticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors="white")
    ax.grid(color="#2D3748", linewidth=0.8)

    legend_patches = []
    for name in MODEL_ORDER:
        row = metrics_df[metrics_df["model"] == name]
        if row.empty:
            continue
        vals = [max(0, float(row[c].values[0])) for c in cat_cols if c in row.columns]
        if len(vals) < 3:
            continue
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[name], lw=2)
        ax.fill(angles, vals, color=COLORS[name], alpha=0.15)
        legend_patches.append(mpatches.Patch(color=COLORS[name], label=name))

    ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.3, 1.1),
              framealpha=0.3, facecolor="#0F1117", edgecolor="#4A5568", labelcolor="white")
    ax.set_title("R² Radar Chart — Per-Axis Performance", color="white", fontsize=13, pad=20)

    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_feature_importance_rf(outpath):
    """Bar chart of RF feature importances."""
    fpath = os.path.join(RESULTS_DIR, "rf_feature_importances.csv")
    if not os.path.exists(fpath):
        return
    df_imp = pd.read_csv(fpath).head(15)
    fig = styled_fig(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.set_facecolor("#161B22")
    bars = ax.barh(range(len(df_imp)), df_imp["importance"], color="#FF6B6B", edgecolor="none")
    ax.set_yticks(range(len(df_imp)))
    ax.set_yticklabels(df_imp["feature"], fontsize=9, color="#CBD5E1")
    ax.set_xlabel("Mean Importance (averaged across axes)", color="#CBD5E1", fontsize=9)
    ax.set_title("Random Forest — Top 15 Feature Importances", color="white", fontsize=12)
    ax.tick_params(colors="#CBD5E1")
    ax.spines[:].set_color("#2D3748")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(metrics_df, all_metrics):
    report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "model_comparison_report.md")
    metric_cols = ["MAE_N", "MAE_E", "MAE_D", "RMSE_N", "RMSE_E", "RMSE_D",
                   "R2_N", "R2_E", "R2_D", "RMSE_mean", "MAE_mean", "R2_mean"]

    best_rmse = metrics_df.loc[metrics_df["RMSE_mean"].astype(float).idxmin(), "model"] \
        if "RMSE_mean" in metrics_df.columns else "N/A"

    lines = [
        "# VIO Drift Reduction — Model Comparison Report",
        "",
        "## Executive Summary",
        "",
        f"Five models were trained to predict EKF velocity residuals (Δv_N, Δv_E, Δv_D) from "
        f"GPS-denied-safe IMU features on UAV flight log data. "
        f"All models were evaluated on the **held-out test flight** `log_54` (57.6 s, 8,306 IMU rows).",
        "",
        f"> **Best overall model (lowest mean RMSE): `{best_rmse}`**",
        "",
        "---",
        "",
        "## Dataset Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Total rows | 38,967 |",
        "| IMU rate | ~200 Hz |",
        "| Training logs | log_51, log_53 |",
        "| Validation log | log_52 |",
        "| **Test log (unseen)** | **log_54** |",
        "| Feature count | 34 (GPS-denied-safe) |",
        "| Targets | Δv_N, Δv_E, Δv_D (m/s) |",
        "",
        "---",
        "",
        "## Results on Unseen Test Flight (log_54)",
        "",
    ]

    # Build table
    if not metrics_df.empty:
        avail_cols = [c for c in metric_cols if c in metrics_df.columns]
        header = "| Model | " + " | ".join(avail_cols) + " |"
        sep    = "|-------|" + "|".join(["------"] * len(avail_cols)) + "|"
        lines += [header, sep]
        for _, row in metrics_df.iterrows():
            vals = [f"{float(row[c]):.5f}" if c in row.index else "N/A" for c in avail_cols]
            lines.append(f"| {row['model']} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "---",
        "",
        "## Mathematical Intuitions",
        "",
        "### 1. Ridge Regression",
        "**Loss**: `L = ||y - Xw||² + α||w||²`  ",
        "**Solution**: `w* = (XᵀX + αI)⁻¹ Xᵀy`  ",
        "Adds L2 penalty to OLS, shrinking correlated coefficients. Closed-form, interpretable.",
        "Serves as a linear baseline — low variance, potentially high bias for non-linear IMU patterns.",
        "",
        "### 2. Random Forest",
        "**Prediction**: `ŷ = (1/B) Σ T_b(x)`  ",
        "**Split**: MSE reduction = `var(parent) − [n_L/n·var(L) + n_R/n·var(R)]`  ",
        "Bagging + random feature subsets decorrelate trees, reducing ensemble variance.",
        "No distributional assumptions; handles non-linear IMU-drift relationships natively.",
        "",
        "### 3. MLP",
        "**Forward pass**: `z⁽ˡ⁾ = σ(W⁽ˡ⁾z⁽ˡ⁻¹⁾ + b⁽ˡ⁾)` with ReLU, BatchNorm, Dropout  ",
        "**Loss**: MSE. **Optimiser**: Adam with ReduceLROnPlateau and early stopping.  ",
        "Universal approximator. Rolling features provide temporal context so a flat input suffices.",
        "",
        "### 4. 1D CNN",
        "**Convolution**: `(x ∗ k)[t] = Σ x[t+i]·k[i]`  ",
        "Learns local temporal patterns (e.g., vibration burst ≈ 35 ms receptive field).  ",
        "GlobalAvgPool aggregates across time → fixed vector → regression head.",
        "Windows of 50 timesteps (~0.25 s). Gap-crossing windows excluded.",
        "",
        "### 5. LSTM",
        "**Cell update**: `cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ tanh(Wcxₜ)`  ",
        "Additive cell update enables long-range gradient flow.  ",
        "Forget gate learns to erase stale IMU history; input gate writes new manoeuvre onset.",
        "Windows of 100 timesteps (~0.5 s). Stateless windowed mode (no cross-window memory).",
        "",
        "---",
        "",
        "## Training Choices",
        "",
        "| Choice | Rationale |",
        "|--------|-----------|",
        "| Train=log51+53, Val=log52, Test=log54 | Temporal integrity; test flight strictly unseen |",
        "| StandardScaler fit on train only | Prevents data leakage from val/test distributions |",
        "| Drop NaN rows (not impute) | NaNs from gap boundaries — imputation would corrupt derivatives |",
        "| Gap-aware windows (skip dt>0.1s) | Prevents CNN/LSTM from learning spurious gap transitions |",
        "| Multi-output (3 targets jointly) | Allows model to learn cross-axis velocity error correlations |",
        "| EarlyStopping + ReduceLROnPlateau | Prevents overfitting on 4-flight small dataset |",
        "| MSE loss | Penalises large errors more than MAE; appropriate for smooth velocity targets |",
        "| Gradient clipping (norm=1.0) | Stabilises LSTM training; prevents exploding gradients |",
        "",
        "---",
        "",
        "## Key Limitations",
        "",
        "- Only 4 flights (~274 s total) → generalisation claims are limited.",
        "- Targets are EKF innovations (GPS−EKF), not raw IMU drift — smaller magnitude than true dead-reckoning error.",
        "- All flights on same day, same location — environmental diversity is minimal.",
        "- Periodic ~2.3 s logging gaps affect sequence models (windows excluded at boundaries).",
        "",
        "---",
        "",
        "## Plots",
        "",
        "All plots saved to `results/plots/`:",
        "- `metrics_heatmap.png` — colour-coded performance across all models and metrics",
        "- `rmse_bar.png` — RMSE bar chart per axis",
        "- `timeseries_tabular.png` — predicted vs true time-series (Ridge, RF, MLP)",
        "- `residuals_violin.png` — absolute residual distribution per model",
        "- `scatter_pred_vs_true.png` — scatter plots",
        "- `training_curves.png` — loss curves (MLP, CNN, LSTM)",
        "- `r2_radar.png` — R² radar chart",
        "- `rf_feature_importance.png` — top-15 RF feature importances",
        "",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved → {report_path}")
    results_report = os.path.join(RESULTS_DIR, "model_comparison_report.md")
    with open(results_report, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved → {results_report}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  06: COMPARISON & VISUALISATION")
    print("=" * 60)

    df = load_data()
    _, _, test_df = split_data(df)

    print("\n[1/4] Loading models and running predictions...")

    # Load metrics already computed by individual scripts
    metrics_path = os.path.join(RESULTS_DIR, "metrics_table.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        print(f"  Found metrics for: {metrics_df['model'].tolist()}")
    else:
        print("  No metrics_table.csv found — run individual model scripts first.")
        metrics_df = pd.DataFrame()

    # Model classes already imported at module level (DriftMLP, DriftCNN1D, DriftLSTM)
    preds = {}

    # Ridge
    if os.path.exists(os.path.join(SAVED_DIR, "ridge_model.pkl")):
        with open(os.path.join(SAVED_DIR, "ridge_model.pkl"), "rb") as f:
            ridge_model = pickle.load(f)
        with open(os.path.join(SAVED_DIR, "ridge_scaler.pkl"), "rb") as f:
            ridge_scaler = pickle.load(f)
        X_te, y_te_tab, idx_tab = prepare_tabular(test_df)
        X_te_s = apply_scaler(ridge_scaler, X_te)
        preds["Ridge"] = {"y_pred": ridge_model.predict(X_te_s),
                           "y_true": y_te_tab, "idx": idx_tab}

    # RF
    if os.path.exists(os.path.join(SAVED_DIR, "rf_model.pkl")):
        with open(os.path.join(SAVED_DIR, "rf_model.pkl"), "rb") as f:
            rf_model = pickle.load(f)
        with open(os.path.join(SAVED_DIR, "rf_scaler.pkl"), "rb") as f:
            rf_scaler = pickle.load(f)
        X_te_s = apply_scaler(rf_scaler, X_te)
        preds["RandomForest"] = {"y_pred": rf_model.predict(X_te_s),
                                  "y_true": y_te_tab, "idx": idx_tab}

    # MLP
    if os.path.exists(os.path.join(SAVED_DIR, "mlp_model.pt")):
        with open(os.path.join(SAVED_DIR, "mlp_scaler.pkl"), "rb") as f:
            mlp_scaler = pickle.load(f)
        X_te_s = apply_scaler(mlp_scaler, X_te)
        mlp_model = DriftMLP(X_te_s.shape[1]).to(DEVICE)
        mlp_model.load_state_dict(torch.load(os.path.join(SAVED_DIR, "mlp_model.pt"),
                                              map_location=DEVICE))
        mlp_model.eval()
        with torch.no_grad():
            y_pred_mlp = mlp_model(torch.tensor(X_te_s).to(DEVICE)).cpu().numpy()
        preds["MLP"] = {"y_pred": y_pred_mlp, "y_true": y_te_tab, "idx": idx_tab}

    # CNN
    if os.path.exists(os.path.join(SAVED_DIR, "cnn_model.pt")):
        X_cnn, y_cnn, _ = build_windows(test_df, CNN_FEATURE_COLS, window_size=50, stride=5)
        with open(os.path.join(SAVED_DIR, "cnn_scaler.pkl"), "rb") as f:
            cnn_scaler = pickle.load(f)
        N, W, F = X_cnn.shape
        X_cnn_s = cnn_scaler.transform(X_cnn.reshape(-1, F)).reshape(N, W, F).astype(np.float32)
        cnn_model = DriftCNN1D(F).to(DEVICE)
        cnn_model.load_state_dict(torch.load(os.path.join(SAVED_DIR, "cnn_model.pt"),
                                              map_location=DEVICE))
        cnn_model.eval()
        with torch.no_grad():
            y_pred_cnn = cnn_model(torch.tensor(X_cnn_s).to(DEVICE)).cpu().numpy()
        preds["CNN1D"] = {"y_pred": y_pred_cnn, "y_true": y_cnn}

    # LSTM
    if os.path.exists(os.path.join(SAVED_DIR, "lstm_model.pt")):
        X_lstm, y_lstm, _ = build_windows(test_df, ALL_FEATURE_COLS, window_size=100, stride=5)
        with open(os.path.join(SAVED_DIR, "lstm_scaler.pkl"), "rb") as f:
            lstm_scaler = pickle.load(f)
        N2, W2, F2 = X_lstm.shape
        X_lstm_s = lstm_scaler.transform(X_lstm.reshape(-1, F2)).reshape(N2, W2, F2).astype(np.float32)
        lstm_model = DriftLSTM(F2).to(DEVICE)
        lstm_model.load_state_dict(torch.load(os.path.join(SAVED_DIR, "lstm_model.pt"),
                                               map_location=DEVICE))
        lstm_model.eval()
        with torch.no_grad():
            y_pred_lstm = lstm_model(torch.tensor(X_lstm_s).to(DEVICE)).cpu().numpy()
        preds["LSTM"] = {"y_pred": y_pred_lstm, "y_true": y_lstm}

    print(f"\n  Models with predictions: {list(preds.keys())}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Generating plots...")
    if not metrics_df.empty:
        plot_metrics_heatmap(metrics_df,
            os.path.join(PLOTS_DIR, "metrics_heatmap.png"))
        plot_rmse_bar(metrics_df,
            os.path.join(PLOTS_DIR, "rmse_bar.png"))
        plot_r2_radar(metrics_df,
            os.path.join(PLOTS_DIR, "r2_radar.png"))

    if preds:
        plot_residuals_violin(preds,
            os.path.join(PLOTS_DIR, "residuals_violin.png"))
        plot_scatter_pred_vs_true(preds,
            os.path.join(PLOTS_DIR, "scatter_pred_vs_true.png"))
        if "Ridge" in preds:
            plot_timeseries(preds, test_df,
                os.path.join(PLOTS_DIR, "timeseries_tabular.png"))

    plot_training_curves(os.path.join(PLOTS_DIR, "training_curves.png"))
    plot_feature_importance_rf(os.path.join(PLOTS_DIR, "rf_feature_importance.png"))

    # ── Final metrics recompute ────────────────────────────────────────────────
    print("\n[3/4] Recomputing and printing final metrics table...")
    all_metrics = []
    for name, data in preds.items():
        m = compute_metrics(data["y_true"], data["y_pred"])
        m["model"] = name
        all_metrics.append(m)

    if all_metrics:
        new_df = pd.DataFrame(all_metrics)
        if not metrics_df.empty:
            meta_cols = [c for c in metrics_df.columns if c not in new_df.columns and c != "model"]
            if meta_cols:
                final_df = new_df.merge(metrics_df[["model"] + meta_cols], on="model", how="left")
            else:
                final_df = new_df
        else:
            final_df = new_df
        final_df.to_csv(os.path.join(RESULTS_DIR, "metrics_table.csv"), index=False)
        print("\n" + final_df[["model", "MAE_mean", "RMSE_mean", "R2_mean"]].to_string(index=False))
        metrics_df = final_df

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n[4/4] Generating comparison report...")
    generate_report(metrics_df, all_metrics)

    print("\n✓ Comparison complete. See results/ and model_comparison_report.md\n")


if __name__ == "__main__":
    main()
