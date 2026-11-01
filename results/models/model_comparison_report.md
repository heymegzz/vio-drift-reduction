# Phase 1: Model Comparison Report

## 1. Executive Summary

This report benchmarks baseline and classical ML models for predicting PX4–GPS velocity residuals (Δv_N, Δv_E, Δv_D) using IMU-derived features.

> **Important caveat:** The PX4 velocity estimates were recorded from the PX4 EKF2 estimator, which may have used GPS aiding during the flight. Removing GPS *features* from the ML model inputs does not remove GPS dependence from the recorded PX4 estimator output. This experiment should be described as *PX4–GPS velocity residual prediction*, not as validated GNSS-denied or IMU-only drift reduction. See `docs/target_definition.md` for details.

Evaluation on the held-out test flight (`log_54`, 8,306 samples):

| model | rmse_mean | mae_mean | r2_mean | bias_mean | rmse_N | rmse_E | rmse_D | mae_N | mae_E | mae_D | r2_N | r2_E | r2_D | bias_N | bias_E | bias_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero Correction | 0.147245 | 0.106464 | -0.012706 | 0.016879 | 0.100908 | 0.102152 | 0.238673 | 0.077365 | 0.072394 | 0.169633 | -0.000229 | -0.002238 | -0.035653 | 0.001527 | 0.004827 | 0.044284 |
| Random Forest | 0.162646 | 0.118935 | -0.170882 | 0.058072 | 0.102422 | 0.104258 | 0.281260 | 0.077179 | 0.075440 | 0.204187 | -0.030461 | -0.043973 | -0.438211 | -0.007082 | 0.005736 | 0.175564 |
| Gradient Boosting | 0.168448 | 0.125173 | -0.255250 | 0.063572 | 0.104705 | 0.109137 | 0.291503 | 0.077377 | 0.081122 | 0.217019 | -0.076903 | -0.143972 | -0.544874 | -0.007037 | 0.011008 | 0.186745 |
| Mean Predictor | 0.169940 | 0.126798 | -0.238542 | 0.068923 | 0.100898 | 0.102441 | 0.306480 | 0.077207 | 0.073021 | 0.230167 | -0.000015 | -0.007912 | -0.707697 | 0.000396 | 0.009076 | 0.197297 |
| Ridge | 0.174600 | 0.131908 | -0.386885 | 0.070271 | 0.116506 | 0.115333 | 0.291962 | 0.083740 | 0.084812 | 0.227171 | -0.333351 | -0.277563 | -0.549740 | -0.003422 | 0.022672 | 0.191562 |


## 2. Baseline Comparison

- **No ML model achieves lower mean RMSE than zero correction** (0.1472 m/s). This means that on this test flight, applying any of the trained corrections **worsens** the overall velocity estimate compared to leaving PX4 output uncorrected.

- This is a common outcome when the residuals are small, noisy, and weakly correlated with the available IMU features. The result is reported honestly as a negative finding.


## 3. Velocity Correction Evaluation

The actual correction is: `corrected_velocity = px4_velocity - predicted_residual`.

Comparing corrected vs. original PX4 velocity against GPS reference:

| model | original_rmse_mean | corrected_rmse_mean | improvement_pct_mean | worsens_overall | original_rmse_N | corrected_rmse_N | improvement_pct_N | original_rmse_E | corrected_rmse_E | improvement_pct_E | original_rmse_D | corrected_rmse_D | improvement_pct_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero Correction | 0.1472 | 0.1472 | 0.0000 | No | 0.1009 | 0.1009 | 0.0000 | 0.1022 | 0.1022 | 0.0000 | 0.2387 | 0.2387 | 0.0000 |
| Mean Predictor | 0.1472 | 0.1699 | -15.4131 | Yes | 0.1009 | 0.1009 | 0.0107 | 0.1022 | 0.1024 | -0.2827 | 0.2387 | 0.3065 | -28.4099 |
| Ridge | 0.1472 | 0.1746 | -18.5785 | Yes | 0.1009 | 0.1165 | -15.4576 | 0.1022 | 0.1153 | -12.9031 | 0.2387 | 0.2920 | -22.3270 |
| Random Forest | 0.1472 | 0.1626 | -10.4600 | Yes | 0.1009 | 0.1024 | -1.5000 | 0.1022 | 0.1043 | -2.0609 | 0.2387 | 0.2813 | -17.8431 |
| Gradient Boosting | 0.1472 | 0.1684 | -14.4002 | Yes | 0.1009 | 0.1047 | -3.7621 | 0.1022 | 0.1091 | -6.8371 | 0.2387 | 0.2915 | -22.1348 |


## 4. Methodology Notes

- **mean(axis RMSE)** is reported — the mean of per-axis RMSEs, not pooled RMSE across all axes.

- **Bias** is mean(predicted − true): positive = model over-predicts.

- Feature importance for Random Forest uses impurity-based importance (variance reduction for regression), not Gini importance (which applies to classification).

- Temporal features (rolling windows, lags, jerk) are reset at segment boundaries to avoid crossing temporal gaps > 50 ms.

- Jerk is computed as da/dt using actual elapsed time between samples.


## 5. Artifacts and Reproducibility

- Comparison plots and time-series overlays are in `results/plots/`.

- Model parameters and scalers are in `configs/` and `saved_models/`.

- Test flight `log_54` has already been inspected during development; it is no longer a pristine unseen test. A new independent recording is recommended for stronger confirmation.
