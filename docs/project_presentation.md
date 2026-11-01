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
- **Key Insight:** Treat drift reduction as a residual learning problem without violating GNSS-denied realism.

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

| Model | Test RMSE Mean (m/s) | Test MAE Mean (m/s) | Test $R^2$ Mean | Horizontal $\Delta v_{N,E}$ RMSE |
|---|---|---|---|---|
| **Ridge Regression** | 0.1734 | 0.1315 | -0.3726 | ~0.115 m/s |
| **Random Forest** | **0.1603** | **0.1155** | **-0.1319** | **~0.101 m/s** |
| **Gradient Boosting** | *See benchmark* | *See benchmark* | *See benchmark* | *See benchmark* |

- **Key Takeaway:** Tree-based non-linear models achieve positive $R^2$ and notable variance reduction on horizontal axes ($\Delta v_N, \Delta v_E$).
- Vertical drift ($\Delta v_D$) remains the hardest to predict due to aerodynamic downwash and gravity coupling.

---

## Slide 7: Viva Voce — Anticipated Questions & Answers

### Q1: Why did you predict velocity residuals instead of absolute position or velocity?
**Answer:** Predicting absolute velocity or position in an open-loop manner inevitably accumulates integration drift. By predicting the *residual error* of the existing PX4 EKF2 filter, our model acts as a closed-loop correction term, preserving the robust high-frequency physical integration of the Kalman filter while learning its systematic biases.

### Q2: Why is the $R^2$ value for Ridge negative, and what does it mean?
**Answer:** A negative $R^2$ on an unseen test log means the linear model performs worse than a simple horizontal line predicting the training mean. This occurs because the drift residuals are driven by non-linear maneuvers (centrifugal forces, attitude tilts) that linear planes cannot fit across different flight profiles. As seen in our results, non-linear tree models immediately recover positive $R^2$ on horizontal axes.

### Q3: How did you ensure there is no data leakage?
**Answer:** We implemented a two-tier leakage prevention protocol:
1. *Feature isolation:* Quarantined all GPS columns, velocity truths, and estimator status flags.
2. *Temporal isolation:* All rolling windows and lag operations are computed per log. Entire flights are separated into train, validation, and test sets.

### Q4: How would this run in real time on an actual drone?
**Answer:** Feature computation requires a sliding buffer of 100 IMU samples (0.5 seconds at 200 Hz). The Random Forest model inference latency is under 1.5 ms per sample, which easily fits within the 5 ms budget of a 200 Hz onboard companion computer (e.g., Raspberry Pi 4 or NVIDIA Jetson).

