# Phase 1 Project Presentation & Viva Voce Guide

## Title: ML-Assisted IMU Drift Reduction for GNSS-Denied UAV Navigation
**Course:** CSAI Project — Phase 1 Evaluation  
**Team Authors:** Meghna Nair, Pranjal Shukla, Shreya Narayani  
**Framework:** Open-source Python (Scikit-Learn, Pandas, NumPy, Matplotlib)

---

## Slide 1: Problem Statement & Motivation
- **Context:** Autonomous UAV navigation relies on IMU measurements (accelerometer + gyroscope) fused by PX4's EKF2.
- **The Challenge:** In GNSS-denied or degraded environments (tunnels, urban canyons, RF jamming), unaided dead reckoning drifts rapidly due to double integration of bias and noise.
- **Objective:** Predict velocity residuals ($\mathbf{v}^{\text{PX4}} - \mathbf{v}^{\text{GPS}}$) using *only* IMU-derived features to synthesize corrective pseudo-measurements for EKF2 during GNSS blackouts.
- **Key Insight:** Treat residual prediction as a regression problem using IMU features. Note: this is PX4–GPS velocity residual prediction; validated GNSS-denied drift reduction requires additional evidence (see target_definition.md).

---

## Slide 2: Dataset Characteristics & Quality Audit
- **Data Source:** High-rate flight logs (PX4 `.ulg` telemetry) totaling **38,967 samples** at ~200 Hz cadence.
- **Integrity:** Zero missing values, zero infinities, verified monotonic timestamps.
- **Quality Findings:**
  - Raw accelerometers retain gravitational component ($a_z \approx -9.78\text{ m/s}^2$ in FRD body frame).
  - Target variables verified with machine precision ($\Delta v = v^{\text{PX4}} - v^{\text{GPS}}$, error $< 10^{-15}$).
  - Critical leakage audit identified and quarantined all GPS velocities, satellite counts, EPH/EPV, and status flags.

---

## Slide 3: Feature Engineering & Leakage Protection
- **Expansion:** 8 raw IMU channels $\to$ **142 engineered features**.
- **Temporal Transformations:**
  - *Multi-scale rolling windows:* (10, 50, 100 samples $\approx$ 50ms, 250ms, 500ms) for local moving average, standard deviation, and dynamic range.
  - *Temporal derivatives:* First differences (rate of change) and second differences (jerk $d^2a/dt^2$).
  - *Lag features:* Short-term historical states (t-1, t-5, t-10).
  - *Kinematic interactions:* Planar angular orientations ($\text{atan2}$) and acceleration-to-angular velocity norm ratios.
- **Strict Boundary Rule:** All temporal statistics are grouped and computed strictly *within* individual flight logs to guarantee zero inter-log contamination.

---

## Slide 4: Splitting Strategy & Autocorrelation Rationale
- **Why Random K-Fold Fails:** EDA autocorrelation analysis demonstrated persistent correlation across hundreds of samples (>1 second). Random row-wise sampling results in severe temporal leakage and over-optimistic test scores.
- **Log-Based Partition:**
  - **Train:** `log_51` + `log_53` (22,004 samples, 56.5%)
  - **Validation:** `log_52` (8,657 samples, 22.2%) — for tuning and stopping
  - **Test (Unseen Flight):** `log_54` (8,306 samples, 21.3%) — final evaluation

---

## Slide 5: Model Architectures & Methodological Rigor
1. **Multi-Output Ridge Regression:**
   - Linear baseline with $L_2$ regularization ($\alpha=1000.0$ selected via Leave-One-Log-Out CV).
   - Solves multicollinearity across dense rolling features.
2. **Random Forest Ensemble:**
   - 150 bagged trees, `max_features='sqrt'`, `max_depth=16`.
   - Captures non-linear cross-axis couplings and rotational kinematics.
3. **Histogram Gradient Boosting:**
   - Gradient-boosted decision trees with shrinkage ($\eta=0.08$) and $L_2$ leaf regularization.
   - Discretized 256-bin histograms provide fast residual learning.

---

## Slide 6: Benchmark Results & Cross-Model Comparison
*(See results/models/metrics_table.csv for full values)*

| Model | Test RMSE Mean (m/s) | Test MAE Mean (m/s) | Test R² Mean | Bias (m/s) |
|---|---|---|---|---|
| **Zero Correction** | **0.1472** | **0.1065** | **-0.013** | 0.017 |
| Random Forest | 0.1626 | 0.1189 | -0.171 | 0.058 |
| Gradient Boosting | 0.1684 | 0.1252 | -0.255 | 0.064 |
| Mean Predictor | 0.1699 | 0.1268 | -0.239 | 0.069 |
| Ridge Regression | 0.1746 | 0.1319 | -0.387 | 0.070 |

- **Key Finding:** No ML model achieves lower RMSE than zero correction on this test flight. Applying any trained correction **worsens** the velocity estimate.
- All R² values are negative, meaning every model performs worse than predicting the target mean.
- This is a negative result, reported honestly as a reproducible benchmark.
- Vertical drift (Δv_D) remains the hardest to predict due to aerodynamic downwash and gravity coupling.

---

## Slide 7: Viva Voce — Anticipated Questions & Answers

### Q1: Why did you predict velocity residuals instead of absolute position or velocity?
**Answer:** Predicting absolute velocity or position in an open-loop manner inevitably accumulates integration drift. By predicting the *residual error* of the existing PX4 EKF2 filter, our model acts as a closed-loop correction term, preserving the robust high-frequency physical integration of the Kalman filter while learning its systematic biases.

### Q2: Why is the R² value negative for all models?
**Answer:** A negative R² on an unseen test log means the model performs worse than a simple constant prediction equal to the target mean. This occurs because the drift residuals are small, noisy, and weakly correlated with the available IMU features on this flight. The residuals are driven by non-linear dynamics and GPS–PX4 synchronization effects that the current feature set does not capture well. This is an honest negative result that motivates Phase 2 investigations (additional sensors, richer features, or alternative target formulations).

### Q2b: Why does zero correction beat all ML models?
**Answer:** Zero correction (predicting residual = 0) has RMSE equal to the RMS of the actual targets. When the targets are small and noisy, any model that introduces prediction error adds noise rather than reducing it. The ML models' prediction errors exceed the targets' inherent magnitude, making "do nothing" the safer strategy on this particular test flight.

### Q3: How did you ensure there is no data leakage?
**Answer:** We implemented a multi-tier leakage prevention protocol:
1. *Feature isolation:* Quarantined all GPS columns, velocity truths, and estimator status flags.
2. *Temporal isolation:* All rolling windows, lags, and differences are computed per continuous segment (gap threshold 50 ms). Entire flights are separated into train, validation, and test sets.
3. *Causal features:* Lag startup rows use forward-fill (past values only), not back-fill.

### Q4: How would this run in real time on an actual drone?
**Answer:** Feature computation requires a sliding buffer of 100 IMU samples (0.5 seconds at 200 Hz). The Random Forest model inference latency is under 1.5 ms per sample, which easily fits within the 5 ms budget of a 200 Hz onboard companion computer (e.g., Raspberry Pi 4 or NVIDIA Jetson).

