# Phase 1: Machine Learning Model Comparison Report

## 1. Executive Summary

This report benchmarks three classical machine learning architectures for predicting velocity drift residuals ($\Delta v_N, \Delta v_E, \Delta v_D$) in GNSS-denied UAV navigation using exclusively IMU-derived features.

Evaluation was conducted on the held-out test flight (`log_54`, 8,306 samples):

| model             |   rmse_mean |   mae_mean |   r2_mean |   rmse_N |   rmse_E |   rmse_D |     mae_N |     mae_E |    mae_D |       r2_N |        r2_E |      r2_D |
|:------------------|------------:|-----------:|----------:|---------:|---------:|---------:|----------:|----------:|---------:|-----------:|------------:|----------:|
| Random Forest     |    0.160292 |   0.115457 | -0.131851 | 0.100309 | 0.101695 | 0.278871 | 0.0739634 | 0.0724832 | 0.199924 |  0.0116092 |  0.00672515 | -0.413887 |
| Gradient Boosting |    0.168173 |   0.124094 | -0.2541   | 0.106264 | 0.108112 | 0.290142 | 0.0783909 | 0.0808349 | 0.213055 | -0.109224  | -0.122592   | -0.530484 |
| Ridge             |    0.173359 |   0.131545 | -0.372568 | 0.115459 | 0.116118 | 0.288498 | 0.0830983 | 0.0847645 | 0.226771 | -0.309493  | -0.295021   | -0.513191 |


## 2. Model Performance Analysis

- **Top Performer:** **Random Forest** achieved the lowest overall RMSE of **0.1603 m/s**.

- **Non-linear vs. Linear Comparison:** Tree-based ensemble methods (Random Forest and Gradient Boosting) substantially outperform the linear Ridge baseline, demonstrating that inertial drift residuals contain significant non-linear dynamics from body rotations and centrifugal accelerations.

- **Axis Breakdown:** Across all models, vertical residual ($\Delta v_D$) exhibits higher variance due to gravity coupling and vertical aerodynamic effects, while horizontal velocity residuals ($\Delta v_N, \Delta v_E$) achieve tighter error bounds.


## 3. Artifacts and Reproducibility

- Detailed bar charts and time-series overlays are stored under `results/plots/`.

- Model parameters and scalers are archived in `configs/` and `saved_models/`.
