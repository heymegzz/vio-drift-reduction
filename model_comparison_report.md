# VIO Drift Reduction — Model Comparison Report

## Executive Summary

Five models were trained to predict EKF velocity residuals (Δv_N, Δv_E, Δv_D) from GPS-denied-safe IMU features on UAV flight log data. All models were evaluated on the **held-out test flight** `log_54` (57.6 s, 8,306 IMU rows).

> **Best overall model (lowest mean RMSE): `CNN1D`**

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Total rows | 38,967 |
| IMU rate | ~200 Hz |
| Training logs | log_51, log_53 |
| Validation log | log_52 |
| **Test log (unseen)** | **log_54** |
| Feature count | 34 (GPS-denied-safe) |
| Targets | Δv_N, Δv_E, Δv_D (m/s) |

---

## Results on Unseen Test Flight (log_54)

| Model | MAE_N | MAE_E | MAE_D | RMSE_N | RMSE_E | RMSE_D | R2_N | R2_E | R2_D | RMSE_mean | MAE_mean | R2_mean |
|-------|------|------|------|------|------|------|------|------|------|------|------|------|
| Ridge | 0.07862 | 0.07935 | 0.19759 | 0.10221 | 0.10846 | 0.27416 | -0.01650 | -0.11772 | -0.35193 | 0.16161 | 0.11852 | -0.16205 |
| RandomForest | 0.07373 | 0.07629 | 0.17995 | 0.09917 | 0.10334 | 0.25821 | 0.04298 | -0.01467 | -0.19921 | 0.15357 | 0.10999 | -0.05697 |
| MLP | 0.07882 | 0.08106 | 0.22542 | 0.10898 | 0.11100 | 0.30149 | -0.15558 | -0.17064 | -0.63490 | 0.17382 | 0.12843 | -0.32037 |
| CNN1D | 0.07606 | 0.07162 | 0.16378 | 0.10519 | 0.09959 | 0.23320 | -0.08682 | -0.02539 | 0.01348 | 0.14600 | 0.10382 | -0.03291 |
| LSTM | 0.08645 | 0.07578 | 0.21664 | 0.11350 | 0.10256 | 0.27949 | -0.28788 | -0.09262 | -0.66957 | 0.16518 | 0.12629 | -0.35003 |

---

## Mathematical Intuitions

### 1. Ridge Regression
**Loss**: `L = ||y - Xw||² + α||w||²`  
**Solution**: `w* = (XᵀX + αI)⁻¹ Xᵀy`  
Adds L2 penalty to OLS, shrinking correlated coefficients. Closed-form, interpretable.
Serves as a linear baseline — low variance, potentially high bias for non-linear IMU patterns.

### 2. Random Forest
**Prediction**: `ŷ = (1/B) Σ T_b(x)`  
**Split**: MSE reduction = `var(parent) − [n_L/n·var(L) + n_R/n·var(R)]`  
Bagging + random feature subsets decorrelate trees, reducing ensemble variance.
No distributional assumptions; handles non-linear IMU-drift relationships natively.

### 3. MLP
**Forward pass**: `z⁽ˡ⁾ = σ(W⁽ˡ⁾z⁽ˡ⁻¹⁾ + b⁽ˡ⁾)` with ReLU, BatchNorm, Dropout  
**Loss**: MSE. **Optimiser**: Adam with ReduceLROnPlateau and early stopping.  
Universal approximator. Rolling features provide temporal context so a flat input suffices.

### 4. 1D CNN
**Convolution**: `(x ∗ k)[t] = Σ x[t+i]·k[i]`  
Learns local temporal patterns (e.g., vibration burst ≈ 35 ms receptive field).  
GlobalAvgPool aggregates across time → fixed vector → regression head.
Windows of 50 timesteps (~0.25 s). Gap-crossing windows excluded.

### 5. LSTM
**Cell update**: `cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ tanh(Wcxₜ)`  
Additive cell update enables long-range gradient flow.  
Forget gate learns to erase stale IMU history; input gate writes new manoeuvre onset.
Windows of 100 timesteps (~0.5 s). Stateless windowed mode (no cross-window memory).

---

## Training Choices

| Choice | Rationale |
|--------|-----------|
| Train=log51+53, Val=log52, Test=log54 | Temporal integrity; test flight strictly unseen |
| StandardScaler fit on train only | Prevents data leakage from val/test distributions |
| Drop NaN rows (not impute) | NaNs from gap boundaries — imputation would corrupt derivatives |
| Gap-aware windows (skip dt>0.1s) | Prevents CNN/LSTM from learning spurious gap transitions |
| Multi-output (3 targets jointly) | Allows model to learn cross-axis velocity error correlations |
| EarlyStopping + ReduceLROnPlateau | Prevents overfitting on 4-flight small dataset |
| MSE loss | Penalises large errors more than MAE; appropriate for smooth velocity targets |
| Gradient clipping (norm=1.0) | Stabilises LSTM training; prevents exploding gradients |

---

## Key Limitations

- Only 4 flights (~274 s total) → generalisation claims are limited.
- Targets are EKF innovations (GPS−EKF), not raw IMU drift — smaller magnitude than true dead-reckoning error.
- All flights on same day, same location — environmental diversity is minimal.
- Periodic ~2.3 s logging gaps affect sequence models (windows excluded at boundaries).

---

## Plots

All plots saved to `results/plots/`:
- `metrics_heatmap.png` — colour-coded performance across all models and metrics
- `rmse_bar.png` — RMSE bar chart per axis
- `timeseries_tabular.png` — predicted vs true time-series (Ridge, RF, MLP)
- `residuals_violin.png` — absolute residual distribution per model
- `scatter_pred_vs_true.png` — scatter plots
- `training_curves.png` — loss curves (MLP, CNN, LSTM)
- `r2_radar.png` — R² radar chart
- `rf_feature_importance.png` — top-15 RF feature importances
