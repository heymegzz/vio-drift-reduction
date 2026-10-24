# Data Quality & Exploratory Data Analysis Report

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| File | `imu_drift_feature_dataset.csv` |
| Rows | 38,967 |
| Columns (original) | 31 |
| File size | ~12.7 MB |
| Missing values | 0 (none in any column) |
| Duplicate rows | 0 |
| Duplicate (log_id, timestamp) | 0 |
| Flight segments | 4 |
| Total recording duration | ~274 seconds |

---

## 2. Flight Segments

All four flights were recorded on the same day (2026-09-08), separated by 1–2 minutes.

| log_id | Rows | Duration (s) | Start Time |
|--------|------|-------------|------------|
| `log_51_2026-9-8-18-47-00` | 10,277 | 70.4 | 18:47:00 |
| `log_52_2026-9-8-18-48-50` | 8,657 | 64.0 | 18:48:50 |
| `log_53_2026-9-8-18-50-18` | 11,727 | 82.2 | 18:50:18 |
| `log_54_2026-9-8-18-51-28` | 8,306 | 57.6 | 18:51:28 |

---

## 3. Sampling Characteristics

### IMU Sampling (~200 Hz)

| Metric | Value |
|--------|-------|
| Median dt | 5.0 ms |
| Mean dt | 7.0 ms (inflated by gaps) |
| Std dt | 67.2 ms |
| Non-positive diffs | 0 |

The median confirms a consistent ~200 Hz IMU rate. The elevated mean and standard deviation are caused by periodic large gaps.

### Large Time Gaps

| Gap Size | Count | Pattern |
|---------|-------|---------|
| > 20 ms | 40 | Mostly clustered as ~2.3s gaps every ~7s |
| > 0.1 s | 37 | Same large gaps |
| > 2.0 s | ~32 | Periodic ~2.3s interruptions |

These gaps occur at regular ~7-second intervals and are consistent across all four logs. They likely represent **logging buffer flushes or telemetry interruptions**, not breaks in flight. The IMU data before and after each gap continues smoothly.

### GPS Update Rate (~5 Hz)

GPS velocity measurements update at a median interval of 0.200s (5 Hz) with zero-order hold between updates. The GPS columns (`gps_vel_n/e/d_m_s`, `gps_speed_m_s`, `gps_eph_m`, `gps_epv_m`) remain constant between updates.

### PX4 EKF Update Rate (~7–8 Hz)

PX4 velocity estimates update at ~130–140 ms intervals, also held constant between updates.

---

## 4. Constant / Zero-Information Columns

Six columns have exactly 1 unique value and carry no predictive information:

| Column | Value | Interpretation |
|--------|-------|---------------|
| `vel_ned_valid` | 1.0 | NED velocity always valid |
| `v_xy_valid` | 1.0 | XY velocity always valid |
| `v_z_valid` | 1.0 | Z velocity always valid |
| `xy_valid` | 1.0 | XY position always valid |
| `z_valid` | 1.0 | Z position always valid |
| `dead_reckoning` | 0.0 | Never in dead-reckoning mode |

**Action taken**: Dropped from the engineered dataset. These indicate that the UAV maintained GPS lock throughout all recordings — there are no GPS-denied intervals in this dataset.

---

## 5. IMU Measurement Statistics

### Accelerometer

| Statistic | accel_x (m/s²) | accel_y (m/s²) | accel_z (m/s²) | magnitude (m/s²) |
|-----------|---------------|---------------|---------------|-----------------|
| Mean | 0.017 | -0.186 | -10.66 | 11.87 |
| Std | 2.86 | 3.66 | 3.70 | 4.47 |
| Min | -19.50 | -29.93 | -54.01 | 0.34 |
| Max | 24.62 | 24.22 | 9.08 | 109.19 |

- **accel_z mean ≈ −9.8 m/s²**: Consistent with gravity along the Z-axis in a body-frame convention where Z points up when level. The negative sign indicates the accelerometer measures reactive force (standard IMU convention).
- **Magnitude mean ≈ 11.87**: Exceeds g ≈ 9.81, indicating frequent dynamic manoeuvring.
- **Magnitude range**: Min=0.34 (near-weightless moment), Max=109.19 m/s² (high-g manoeuvre or vibration spike — ~11g). The 109.19 value is notable but could be genuine in an aggressive UAV manoeuvre or a vibration transient; **not automatically classified as an error**.

### Gyroscope

| Statistic | gyro_x (rad/s) | gyro_y (rad/s) | gyro_z (rad/s) | magnitude (rad/s) |
|-----------|---------------|---------------|---------------|------------------|
| Mean | 0.032 | 0.009 | -0.007 | 0.538 |
| Std | 0.39 | 0.36 | 0.34 | 0.359 |
| Min | -2.68 | -3.97 | -2.72 | 0.0006 |
| Max | 2.85 | 2.71 | 2.08 | 5.98 |

- Maximum angular velocity ~5.98 rad/s (~343°/s) — represents aggressive rotation but within typical MEMS gyro range.
- Near-zero means suggest roughly balanced flight with no persistent spin bias in the data.

---

## 6. GPS Quality Assessment

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| gps_eph_m | 0.616 | 0.114 | 0.498 | 1.075 |
| gps_epv_m | 1.312 | 0.321 | 0.967 | 2.496 |
| gps_satellites_used | 27.2 | 1.5 | 23 | 31 |
| gps_speed_m_s | 0.717 | 0.811 | 0.006 | 5.440 |

- **High satellite count** (23–31): Excellent GPS environment with no obstructions.
- **Low eph** (< 1.1m): Horizontal GPS accuracy consistently good.
- **fix_type**: 3.0 (3D fix) for log_51 only; 4.0 (3D + DGPS) for logs 52–54. Log 51 has slightly lower GPS quality.
- **Low speeds** (mean 0.72 m/s): The UAV was moving slowly or hovering for much of the flight.

---

## 7. Target Variable Analysis

### Statistics

| Target | Mean | Std | Min | Max | |Abs Mean| |
|--------|------|-----|-----|-----|----------|
| Δv_N (m/s) | -0.004 | 0.129 | -1.307 | 0.990 | 0.089 |
| Δv_E (m/s) | 0.003 | 0.113 | -0.929 | 0.753 | 0.080 |
| Δv_D (m/s) | 0.064 | 0.266 | -1.582 | 1.042 | 0.202 |

### Key Observations

1. **Down axis has largest errors**: Δv_D has 2× the standard deviation and 2.5× the absolute mean of the horizontal axes. Vertical velocity estimation is harder — this is expected due to weaker vertical GPS geometry and barometric effects.

2. **Near-zero horizontal means**: Δv_N and Δv_E are approximately zero-mean, suggesting the EKF horizontal estimates are roughly unbiased on average.

3. **Positive Δv_D mean (0.064)**: Systematic downward velocity overestimation by the EKF (or GPS underestimation), suggesting a persistent vertical bias.

4. **Low inter-target correlation**: N-E correlation = 0.18, N-D = -0.05, E-D = 0.006. The three axes are largely independent prediction problems.

5. **Zero-order hold pattern**: Targets are piecewise-constant, changing at ~10–15 Hz (when either GPS or PX4 updates). Many consecutive IMU rows share identical target values.

---

## 8. Potential Concerns

### 8.1 Extreme Accelerometer Values
The maximum accelerometer magnitude of 109.19 m/s² (~11.1g) is unusually high. This could represent:
- Genuine high-frequency vibration (common in multirotor UAVs)
- A brief sensor saturation event
- An actual extreme manoeuvre

**No action taken** — the value is not removed. It should be monitored during modelling; if it causes problems, investigate further.

### 8.2 Target as EKF Innovation (Not Pure IMU Drift)
The PX4 EKF already fuses GPS. Therefore, `target = gps_vel - px4_v` is the EKF innovation (measurement residual), not the error from dead-reckoning IMU integration alone. This means:
- The target magnitude is likely **smaller** than pure IMU drift would be
- The ML model learns to predict residual filter error, not raw navigation drift
- This is still a valid and useful task, but the distinction matters for academic claims

### 8.3 Short Flight Durations
Flights span 57–82 seconds each. Long-term drift behaviour (minutes to hours) is not captured. The model may not generalise to longer missions.

### 8.4 Only 4 Flights
With only 4 flight segments, leave-one-flight-out cross-validation provides only 4 folds. Generalisation claims will be limited.

### 8.5 Periodic ~2.3s Gaps
These gaps appear at ~7-second intervals in every log. During these gaps, no IMU samples are recorded, meaning any integration-based features would accumulate error differently. The feature engineering script correctly handles these by invalidating derivatives at gap boundaries.

---

## 9. Correlation Highlights

### Raw IMU → Target Correlations (from inspection)

Individual IMU axes have **very weak linear correlation** with targets (|r| < 0.08 for all). This is expected — the relationship between instantaneous IMU readings and velocity error is non-linear and depends on temporal context. This motivates:
- Rolling statistics (capturing recent dynamics, not single-sample values)
- Derivatives (capturing rate of change)
- Potential for non-linear ML models

### Strongest Linear Predictors of Δv_D

| Feature | Correlation |
|---------|-------------|
| fix_type | -0.477 |
| px4_vz_m_s | 0.393 |
| timestamp | -0.360 |
| gps_satellites_used | -0.291 |

These are all GPS-related or temporal (excluded from GPS-denied predictors), explaining why the prediction task from IMU alone will be challenging.
