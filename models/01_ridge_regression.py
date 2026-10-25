#!/usr/bin/env python3
"""
01_ridge_regression.py
======================
Ridge Regression baseline for VIO drift prediction.

Mathematical Summary
--------------------
Ridge solves: min_w  ||y - Xw||^2 + alpha * ||w||^2
Closed-form:  w* = (X^T X + alpha*I)^{-1} X^T y

Key properties:
  - L2 penalty shrinks correlated weights toward zero.
  - Closed-form solution — no gradient descent, no epochs.
  - Assumes linear feature-target relationship.
  - StandardScaler required (L2 penalty is scale-sensitive).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import time
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from utils import (
    load_data, split_data, prepare_tabular, fit_scaler, apply_scaler,
    compute_metrics, print_metrics, save_sklearn_model, save_metrics,
    load_config, ALL_FEATURE_COLS, TARGET_COLS
)

MODEL_NAME = "Ridge"


def main():
    print("=" * 60)
    print("  MODEL 1: RIDGE REGRESSION")
    print("=" * 60)

    cfg = load_config("ridge")

    # ── Load & split ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    # ── Prepare tabular arrays ────────────────────────────────────────────────
    print("\n[2/5] Preparing features...")
    X_tr, y_tr, _ = prepare_tabular(train_df)
    X_v,  y_v,  _ = prepare_tabular(val_df)
    X_te, y_te, _ = prepare_tabular(test_df)

    scaler = fit_scaler(X_tr)
    X_tr_s = apply_scaler(scaler, X_tr)
    X_v_s  = apply_scaler(scaler, X_v)
    X_te_s = apply_scaler(scaler, X_te)
    print(f"  Feature set: {X_tr_s.shape[1]} features")

    # ── Hyperparameter search ─────────────────────────────────────────────────
    print("\n[3/5] Alpha search via GridSearchCV (3-fold TimeSeriesSplit)...")
    alphas = cfg["hyperparameters"]["alpha_search_space"]
    tscv = TimeSeriesSplit(n_splits=3)

    # For multi-output, we search alpha on the concatenated training data.
    # Ridge handles multi-output natively; MultiOutputRegressor wraps it.
    best_alpha, best_val_rmse = None, np.inf
    for alpha in alphas:
        model = MultiOutputRegressor(Ridge(alpha=alpha, fit_intercept=True))
        model.fit(X_tr_s, y_tr)
        pred_v = model.predict(X_v_s)
        val_rmse = np.sqrt(np.mean((y_v - pred_v) ** 2))
        print(f"  alpha={alpha:8.4f}  val_RMSE={val_rmse:.6f}")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_alpha = alpha

    print(f"\n  ★ Best alpha = {best_alpha}  (val RMSE = {best_val_rmse:.6f})")

    # ── Final training ────────────────────────────────────────────────────────
    print("\n[4/5] Training final model on train+val...")
    X_full_s = apply_scaler(scaler, np.vstack([X_tr, X_v]))
    y_full   = np.vstack([y_tr, y_v])

    model = MultiOutputRegressor(Ridge(alpha=best_alpha, fit_intercept=True))
    t0 = time.time()
    model.fit(X_full_s, y_full)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.2f}s")

    # ── Coefficient analysis ──────────────────────────────────────────────────
    print("\n  Ridge coefficients (top 5 by |coef| per axis):")
    axis_names = ["N", "E", "D"]
    coef_rows = []
    for i, (est, ax) in enumerate(zip(model.estimators_, axis_names)):
        coefs = est.coef_
        top_idx = np.argsort(np.abs(coefs))[::-1][:5]
        print(f"    Δv_{ax}:")
        for idx in top_idx:
            print(f"      {ALL_FEATURE_COLS[idx]:30s}: {coefs[idx]:+.4f}")
        coef_rows.append({"axis": ax, **{ALL_FEATURE_COLS[j]: coefs[j]
                                          for j in range(len(ALL_FEATURE_COLS))}})
    pd.DataFrame(coef_rows).to_csv(
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "results", "ridge_coefficients.csv"), index=False)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on TEST flight (log_54, unseen)...")
    t0 = time.time()
    y_pred = model.predict(X_te_s)
    infer_ms = 1000 * (time.time() - t0) / len(X_te_s) * 1000
    print(f"  Inference: {infer_ms:.3f} ms per 1000 samples")

    metrics = compute_metrics(y_te, y_pred)
    metrics["alpha"] = best_alpha
    metrics["train_time_s"] = round(train_time, 3)
    metrics["infer_ms_per_1k"] = round(infer_ms, 3)
    metrics["n_features"] = X_tr_s.shape[1]
    print_metrics(metrics, MODEL_NAME)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_sklearn_model(model, scaler, "ridge")
    save_metrics(metrics, MODEL_NAME)

    print(f"\n✓ Ridge Regression complete.\n")
    return y_pred, y_te


if __name__ == "__main__":
    main()
