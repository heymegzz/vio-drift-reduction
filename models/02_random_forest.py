#!/usr/bin/env python3
"""
02_random_forest.py
===================
Random Forest for VIO drift prediction.

Mathematical Summary
--------------------
RF = ensemble of B CART decision trees, each trained on bootstrap sample.
Prediction: y_hat = (1/B) * sum_{b=1}^{B} T_b(x)

Split criterion (regression): MSE reduction
  gain = var(parent) - [n_L/n * var(left) + n_R/n * var(right)]

Variance reduction via averaging:
  Var[mean(T_b)] ≈ rho * sigma^2 + (1-rho)*sigma^2/B
  where rho = pairwise tree correlation, sigma^2 = single-tree variance.
  Random feature subset (sqrt(p) per split) minimises rho.

Assumptions:
  - No distributional assumptions on residuals.
  - Non-linear relationships between IMU features and velocity error.
  - Rolling-stat features provide sufficient temporal context per sample.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV

from utils import (
    load_data, split_data, prepare_tabular, fit_scaler, apply_scaler,
    compute_metrics, print_metrics, save_sklearn_model, save_metrics,
    load_config, ALL_FEATURE_COLS, TARGET_COLS
)

MODEL_NAME = "RandomForest"


def main():
    print("=" * 60)
    print("  MODEL 2: RANDOM FOREST")
    print("=" * 60)

    cfg = load_config("rf")
    hp  = cfg["hyperparameters"]

    # ── Load & split ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    # ── Prepare tabular arrays ────────────────────────────────────────────────
    print("\n[2/5] Preparing features...")
    X_tr, y_tr, _ = prepare_tabular(train_df)
    X_v,  y_v,  _ = prepare_tabular(val_df)
    X_te, y_te, _ = prepare_tabular(test_df)

    # RF is scale-invariant but we scale for consistency
    scaler = fit_scaler(X_tr)
    X_tr_s = apply_scaler(scaler, X_tr)
    X_v_s  = apply_scaler(scaler, X_v)
    X_te_s = apply_scaler(scaler, X_te)
    print(f"  Feature set: {X_tr_s.shape[1]} features, {X_tr_s.shape[0]} training rows")

    # ── Hyperparameter search ─────────────────────────────────────────────────
    print("\n[3/5] RandomizedSearchCV on training set (20 iter, 3-fold)...")
    param_dist = {
        "estimator__max_depth":        hp["max_depth_search"],
        "estimator__max_features":     hp["max_features_search"],
        "estimator__min_samples_leaf": hp["min_samples_leaf_search"],
    }

    base_rf = RandomForestRegressor(
        n_estimators=hp["n_estimators"],
        random_state=hp["random_state"],
        n_jobs=-1
    )
    multi_rf = MultiOutputRegressor(base_rf, n_jobs=1)

    search = RandomizedSearchCV(
        multi_rf,
        param_distributions=param_dist,
        n_iter=hp["n_iter_random_search"],
        cv=3,
        scoring="neg_mean_squared_error",
        random_state=hp["random_state"],
        verbose=1,
        n_jobs=1,
    )
    search.fit(X_tr_s, y_tr)
    best_params = search.best_params_
    print(f"\n  ★ Best params: {best_params}")
    print(f"  ★ Best CV MSE: {-search.best_score_:.6f}")

    # Evaluate best params on val set
    best_model_cv = search.best_estimator_
    pred_v = best_model_cv.predict(X_v_s)
    val_rmse = np.sqrt(np.mean((y_v - pred_v) ** 2))
    print(f"  Val RMSE: {val_rmse:.6f}")

    # ── Final training ────────────────────────────────────────────────────────
    print("\n[4/5] Training final model on train+val...")
    X_full_s = apply_scaler(scaler, np.vstack([X_tr, X_v]))
    y_full   = np.vstack([y_tr, y_v])

    # Extract best params for individual estimator
    depth   = best_params.get("estimator__max_depth")
    mf      = best_params.get("estimator__max_features", "sqrt")
    msl     = best_params.get("estimator__min_samples_leaf", 1)

    final_rf = RandomForestRegressor(
        n_estimators=hp["n_estimators"],
        max_depth=depth,
        max_features=mf,
        min_samples_leaf=msl,
        random_state=hp["random_state"],
        n_jobs=-1
    )
    model = MultiOutputRegressor(final_rf, n_jobs=1)
    t0 = time.time()
    model.fit(X_full_s, y_full)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # ── Feature importances ───────────────────────────────────────────────────
    print("\n  Feature importances (top 10, averaged across axes):")
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    top_idx = np.argsort(importances)[::-1][:10]
    imp_rows = []
    for rank, idx in enumerate(top_idx):
        print(f"    {rank+1:2d}. {ALL_FEATURE_COLS[idx]:35s}: {importances[idx]:.4f}")
        imp_rows.append({"rank": rank+1, "feature": ALL_FEATURE_COLS[idx], "importance": importances[idx]})
    pd.DataFrame(imp_rows).to_csv(
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "results", "rf_feature_importances.csv"), index=False)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on TEST flight (log_54, unseen)...")
    t0 = time.time()
    y_pred = model.predict(X_te_s)
    infer_ms = 1000 * (time.time() - t0) / len(X_te_s) * 1000
    print(f"  Inference: {infer_ms:.3f} ms per 1000 samples")

    metrics = compute_metrics(y_te, y_pred)
    metrics["best_max_depth"]        = str(depth)
    metrics["best_max_features"]     = str(mf)
    metrics["best_min_samples_leaf"] = str(msl)
    metrics["train_time_s"]          = round(train_time, 1)
    metrics["infer_ms_per_1k"]       = round(infer_ms, 3)
    metrics["n_features"]            = X_tr_s.shape[1]
    print_metrics(metrics, MODEL_NAME)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_sklearn_model(model, scaler, "rf")
    save_metrics(metrics, MODEL_NAME)

    print(f"\n✓ Random Forest complete.\n")
    return y_pred, y_te


if __name__ == "__main__":
    main()
