# Exploratory Data Analysis Report

**Project**: ML-Assisted IMU Drift Reduction for GNSS-Denied UAV Navigation  
**Date**: November 2026  
**Scope**: Dataset understanding, quality assessment, target verification, EDA

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| File | `data/imu_drift_feature_dataset.csv` |
| Rows | 38,967 |
| Columns | 31 (1 string, 30 numeric) |
| Size | ~12 MB |
| Recording segments | 4 (`log_51` through `log_54`) |
| Time span per log | 56–82 seconds |
| Missing values | None |
| Infinite values | None |
| Duplicate rows | None |

The dataset was extracted from PX4 `.ulg` flight logs recorded on
2026-09-08. All four logs have timestamps from the same session. Whether
they represent independent flights, system restarts, or segments of one
flight is unknown.

## 2. Quality Findings

### 2.1 Completeness

- **No missing values** in any column across all 38,967 rows.
- **No infinite values** (positive or negative) in any numeric column.
- **No exact duplicate rows**.
- **No duplicate timestamps** within any individual log.

### 2.2 Constant columns

The following columns have only a single unique value across the entire
dataset and carry no discriminative information:

| Column | Value |
|--------|-------|
| `vel_ned_valid` | 1.0 |
| `v_xy_valid` | 1.0 |
| `v_z_valid` | 1.0 |
| `xy_valid` | 1.0 |
| `z_valid` | 1.0 |
| `dead_reckoning` | 0.0 |

These flags indicate all rows report valid velocity and position estimates
with no dead-reckoning state. While constant in this dataset, they may vary
in datasets with actual GNSS outages.

### 2.3 Data types

- `log_id`: string (recording identifier)
- All other columns: numeric (int64 for `timestamp`, float64 for rest)
- No unexpected non-numeric columns

## 3. Sampling and Gap Observations

### 3.1 Row cadence

- Nominal row interval: **~5,000 µs (5 ms)**, corresponding to ~200 Hz
  row cadence.
- This is the CSV row rate, not necessarily the native sensor sampling rate.
  The IMU may sample at a higher rate with rows representing merged or
  downsampled output.

### 3.2 Timestamp characteristics

- `timestamp` is strictly monotonically increasing within each log (no
  repeated or decreasing values).
- `time_s` is numerically consistent with `(timestamp − timestamp₀) / 1e6`
  to within ~1e-6 seconds.

### 3.3 Gaps

- Gaps exceeding 10 ms are present in all logs but are infrequent.
- The largest gaps are typically 10–15 ms (2–3× the nominal interval).
- These may reflect USB logging delays, topic synchronisation, or brief
  sensor dropouts.
- Continuous lines should not be drawn across large gaps in time-series
  plots.

### 3.4 Per-log summary

| Log | Rows | Time span (s) | Monotonic | Gaps > 10 ms |
|-----|------|---------------|-----------|--------------|
| log_51 | ~10,700 | ~53 s | Yes | Few |
| log_52 | ~11,200 | ~56 s | Yes | Few |
| log_53 | ~7,600 | ~38 s | Yes | Few |
| log_54 | ~9,400 | ~47 s | Yes | Few |

(Exact counts are in `results/eda/tables/log_summary.csv`.)

## 4. Sensor and Reference Observations

### 4.1 Accelerometer

- Mean Z-axis reading ≈ −9.78 m/s², consistent with gravity in a
  front-right-down (FRD) body frame. Data appears to be **raw** (not
  gravity-compensated).
- X and Y axes are centred near zero with symmetric spread, suggesting
  varied manoeuvres.
- Magnitude range: 0.34 to 109.2 m/s². The maximum exceeds 11 g and
  likely corresponds to high-vibration or aggressive-rotation events.

### 4.2 Gyroscope

- All three axes are approximately zero-mean, consistent with varied
  rotational manoeuvres.
- Magnitude range: 0.0006 to 5.98 rad/s (up to ~343 °/s).
- High angular rates may correspond to aggressive turns or vibration.

### 4.3 Magnitude verification

Both `accel_magnitude_m_s2` and `gyro_magnitude_rad_s` match their
recomputed Euclidean norms to machine precision (max error < 1e-10).

### 4.4 GPS

- GPS velocity updates at a lower rate than the row cadence; many
  consecutive rows share identical GPS values.
- `fix_type` is either 3 (3D fix) or 4 (DGPS/RTK), indicating GPS was
  available throughout.
- Satellite count ranges from 23 to 31 — consistently high.
- EPH (horizontal precision) ranges from ~0.5 to ~1.1 m.

### 4.5 PX4 estimator

- PX4 velocity also shows periods of repeated values, updating at an
  intermediate cadence between IMU and GPS.
- All validity flags are 1.0 and `dead_reckoning` is always 0.0.
- **Important**: This does not prove that PX4 velocity is GPS-independent.
  The EKF2 likely fuses GPS when available, so `px4_v*` may already contain
  GPS corrections.

### 4.6 Absent data categories

Not present in this dataset:
- GPS position (lat, lon, alt)
- PX4 position estimates
- Attitude (quaternion / Euler angles)
- Magnetometer data
- Barometer data

## 5. Target Definition — Verified

The targets are velocity residuals: `target = px4_velocity − gps_velocity`.

This was verified numerically with maximum absolute error < 1e-15 across
all 38,967 rows for all three NED components.

See `docs/target_definition.md` for detailed interpretation, correction
sign convention, and limitations.

**Key caveat**: These residuals represent the difference between a
potentially GPS-aided estimator and GPS itself. They are not automatically
equal to pure IMU integration drift.

## 6. Main Visual Findings

### 6.1 Time series

- Accelerometer and gyroscope signals show periods of relative calm
  interspersed with high-activity bursts, suggesting varied manoeuvres.
- GPS and PX4 velocities track each other closely in all logs, with
  PX4 showing smoother behaviour (expected from Kalman filtering).
- Target residuals are small (typically within ±0.3 m/s) with occasional
  larger excursions.

### 6.2 Distributions

- IMU distributions are approximately symmetric and peaked near zero for
  X/Y accelerometers and all gyroscope axes.
- Z-axis accelerometer is centred near −9.8 m/s² (gravity).
- Target distributions are roughly symmetric and zero-centred, with
  heavier tails in the down (Z) component.
- Per-log distributions show broadly similar shapes, though the spread
  and peak locations vary — indicating different motion profiles across
  recordings.

### 6.3 Correlations

- Strong correlations exist between PX4 and GPS velocity components
  (expected: they measure similar quantities).
- Moderate correlations between some IMU channels and targets.
- **Caution**: Correlations involving targets and PX4/GPS velocities are
  algebraically induced (`target = px4 − gps`) and do not prove causal
  relationships.

### 6.4 Autocorrelation

- Target residuals show **strong autocorrelation** persisting for hundreds
  of lags (~200+ rows ≈ 1+ seconds).
- This indicates high temporal dependence: consecutive rows carry heavily
  correlated information.
- **Implication**: Random row-wise train/test splitting would leak temporal
  information and produce artificially optimistic evaluation metrics.
  Time-based or log-based splitting is essential.

## 7. Leakage Risks

| Risk | Columns | Severity |
|------|---------|----------|
| GPS velocity as input feature | `gps_vel_n/e/d_m_s` | **Critical** — trivially reconstructs target |
| GPS quality as input | `gps_eph_m`, `gps_epv_m`, `gps_satellites_used`, `fix_type` | **High** — unavailable in GNSS-denied scenario |
| GPS speed as input | `gps_speed_m_s` | **High** — derived from GPS velocity |
| GPS validity flag | `vel_ned_valid` | **Moderate** — always 1.0 here but GPS-dependent |
| PX4 validity flags | `v_xy_valid`, `v_z_valid`, `xy_valid`, `z_valid` | **Moderate** — may depend on GPS lock status |
| Dead reckoning flag | `dead_reckoning` | **High** — directly indicates GPS availability |
| Temporal leakage | Row-wise splitting | **Critical** — high autocorrelation |

See `docs/data_dictionary.md` for a detailed discussion of each column's
leakage potential.

## 8. Limitations

1. **No provenance trail**: The original `.ulg` files, extraction script,
   and PX4 configuration are not available. Column semantics are inferred
   from naming conventions.

2. **Unknown flight conditions**: No metadata describes vehicle type,
   location, weather, pilot inputs, or flight mode. Differences between
   logs cannot be attributed to specific conditions.

3. **GPS always available**: `dead_reckoning` is always 0 and GPS quality
   is consistently high. The dataset does not contain actual GNSS outage
   periods, limiting direct validation of GNSS-denied performance.

4. **Limited diversity**: Four logs from a single date. Model
   generalization to other vehicles, locations, or conditions is untested.

5. **Unknown estimator configuration**: Whether PX4 EKF2 was actively
   fusing GPS during these recordings cannot be confirmed from the CSV
   alone.

6. **Coordinate frame uncertainty**: Body frame (FRD) and NED assignments
   are assumed from PX4 conventions but not independently verified.

## 9. Questions to Resolve Before Modelling

1. **EKF2 fusion status**: Was GPS actively fused into the PX4 velocity
   estimate? If so, what exactly do the residuals represent?

2. **Dataset representativeness**: Are four logs from one session sufficient
   for training a generalizable model?

3. **Feature set**: Which columns are permissible inputs in a GNSS-denied
   scenario? (Only IMU channels, magnitudes, and timestamps are clearly
   safe.)

4. **Splitting strategy**: Should evaluation use log-based splits (4 logs)
   or time-based splits within logs? With only 4 logs, cross-validation
   options are limited.

5. **Target stationarity**: Are the residual statistics stable across logs,
   or should models account for distributional shift?

6. **Extreme values**: Should the 11 g accelerometer readings be
   investigated further or accepted as physical extremes?

7. **GPS update alignment**: How were GPS and IMU rows time-aligned in the
   extraction process?

## 10. Recommendation on Dataset Regeneration

Dataset regeneration is **not required** based on the EDA findings:

- The data is complete, well-formatted, and internally consistent.
- Targets are correctly computed.
- No corruption or systematic errors were detected.

However, regeneration may be **conditionally recommended** if:

- The EKF2 fusion configuration needs to be controlled or verified.
- More recording diversity (different flights, conditions, vehicles) is
  desired.
- Actual GNSS outage periods are needed for validation.
- The time-alignment method between topics needs to be documented.

## 11. Next Steps Before Feature Engineering

1. Clarify PX4 EKF2 fusion configuration.
2. Decide on train/test splitting strategy (log-based recommended).
3. Define the permissible feature set (GPS-free inputs only).
4. Decide whether extreme IMU values require investigation or filtering.
5. Consider whether additional recordings or simulated GNSS outages are
   needed.
