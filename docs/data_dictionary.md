# Data Dictionary

## Dataset: `data/imu_drift_feature_dataset.csv`

This document describes every column in the dataset. Information about units
and frames is inferred from column naming conventions consistent with PX4
flight-controller data. Entries marked with **(assumed)** lack independent
confirmation.

---

### Identifiers and Time

| Column      | Dtype   | Description | Unit | Frame | Role | Notes |
|-------------|---------|-------------|------|-------|------|-------|
| `log_id`    | string  | Recording segment identifier (e.g., `log_51_2026-9-8-18-47-00`). Encodes a sequence number and date-time string. | — | — | Grouping key | Whether each log represents an independent flight is unconfirmed. All four logs have timestamps from the same date. |
| `timestamp` | int64   | Monotonic timestamp, likely in microseconds from PX4 system boot. | µs **(assumed)** | — | Primary time index | Strictly increasing within each log. Not a wall-clock or UTC time. |
| `time_s`    | float64 | Elapsed time in seconds since the start of each log, computed as `(timestamp − timestamp₀) / 1e6`. | s | — | Convenience time index | Numerically consistent with `timestamp` to within ~1e-6 s. |

### IMU — Accelerometer

| Column              | Dtype   | Description | Unit | Frame | Role | Notes |
|---------------------|---------|-------------|------|-------|------|-------|
| `accel_x_m_s2`      | float64 | Accelerometer X-axis reading. | m/s² **(assumed from name)** | Body **(assumed FRD)** | Input feature | Includes gravity. Mean Z ≈ −9.78 m/s² consistent with FRD gravity. |
| `accel_y_m_s2`      | float64 | Accelerometer Y-axis reading. | m/s² | Body | Input feature | |
| `accel_z_m_s2`      | float64 | Accelerometer Z-axis reading. | m/s² | Body | Input feature | |
| `accel_magnitude_m_s2` | float64 | Euclidean norm of the three accelerometer axes. | m/s² | — | Derived feature | Verified: matches `√(ax² + ay² + az²)` to machine precision. |

### IMU — Gyroscope

| Column              | Dtype   | Description | Unit | Frame | Role | Notes |
|---------------------|---------|-------------|------|-------|------|-------|
| `gyro_x_rad_s`      | float64 | Gyroscope X-axis reading. | rad/s **(assumed from name)** | Body **(assumed FRD)** | Input feature | Calibration status unknown. |
| `gyro_y_rad_s`      | float64 | Gyroscope Y-axis reading. | rad/s | Body | Input feature | |
| `gyro_z_rad_s`      | float64 | Gyroscope Z-axis reading. | rad/s | Body | Input feature | |
| `gyro_magnitude_rad_s` | float64 | Euclidean norm of the three gyroscope axes. | rad/s | — | Derived feature | Verified: matches `√(gx² + gy² + gz²)` to machine precision. |

### GPS Velocity and Quality

| Column               | Dtype   | Description | Unit | Frame | Role | Notes |
|----------------------|---------|-------------|------|-------|------|-------|
| `gps_vel_n_m_s`      | float64 | GPS velocity, North component. | m/s | NED **(assumed)** | Reference | Updates at lower rate than IMU; many consecutive identical values. |
| `gps_vel_e_m_s`      | float64 | GPS velocity, East component. | m/s | NED | Reference | |
| `gps_vel_d_m_s`      | float64 | GPS velocity, Down component. | m/s | NED | Reference | |
| `vel_ned_valid`      | float64 | GPS NED velocity validity flag. | boolean (0/1) | — | Quality flag | Always 1.0 in this dataset. |
| `fix_type`           | float64 | GPS fix type. 3 = 3D fix, 4 = 3D fix with DGPS or RTK. | enum | — | Quality indicator | Values are 3.0 and 4.0. |
| `gps_speed_m_s`      | float64 | GPS ground speed. | m/s | — | Derived / reference | Likely `√(vel_n² + vel_e²)` but not verified. |
| `gps_eph_m`          | float64 | GPS horizontal position accuracy estimate (EPH). | m | — | Quality indicator | Smaller values indicate better precision. |
| `gps_epv_m`          | float64 | GPS vertical position accuracy estimate (EPV). | m | — | Quality indicator | |
| `gps_satellites_used`| float64 | Number of GPS satellites used in the solution. | count | — | Quality indicator | Range: 23–31. |

### PX4 Estimator Velocity and Validity

| Column           | Dtype   | Description | Unit | Frame | Role | Notes |
|------------------|---------|-------------|------|-------|------|-------|
| `px4_vx_m_s`     | float64 | PX4 EKF2 velocity estimate, X (North). | m/s | NED **(assumed)** | Estimator output | May be GPS-aided; provenance requires EKF2 config review. |
| `px4_vy_m_s`     | float64 | PX4 EKF2 velocity estimate, Y (East). | m/s | NED | Estimator output | |
| `px4_vz_m_s`     | float64 | PX4 EKF2 velocity estimate, Z (Down). | m/s | NED | Estimator output | |
| `v_xy_valid`     | float64 | PX4 horizontal velocity validity flag. | boolean (0/1) | — | Quality flag | Always 1.0 in this dataset. |
| `v_z_valid`      | float64 | PX4 vertical velocity validity flag. | boolean (0/1) | — | Quality flag | Always 1.0 in this dataset. |
| `xy_valid`       | float64 | PX4 horizontal position validity flag. | boolean (0/1) | — | Quality flag | Always 1.0 in this dataset. |
| `z_valid`        | float64 | PX4 vertical position validity flag. | boolean (0/1) | — | Quality flag | Always 1.0 in this dataset. |
| `dead_reckoning` | float64 | PX4 dead-reckoning indicator. | boolean (0/1) | — | Quality flag | Always 0.0 — estimator reports no dead-reckoning state. Cannot conclude GPS was actively fused from this flag alone. |

### Target Variables

| Column                 | Dtype   | Description | Unit | Frame | Role | Notes |
|------------------------|---------|-------------|------|-------|------|-------|
| `target_delta_v_n_m_s` | float64 | Velocity residual, North: `px4_vx − gps_vel_n`. | m/s | NED | **Target** | See `docs/target_definition.md`. |
| `target_delta_v_e_m_s` | float64 | Velocity residual, East: `px4_vy − gps_vel_e`. | m/s | NED | **Target** | |
| `target_delta_v_d_m_s` | float64 | Velocity residual, Down: `px4_vz − gps_vel_d`. | m/s | NED | **Target** | |

---

## Potential Leakage Sources

1. **GPS velocity columns**: If the modelling goal is GNSS-denied correction,
   GPS velocity (`gps_vel_*`) must not be used as input features at inference
   time. However, the GPS velocity is part of the target computation, so
   including it as a feature would trivially reconstruct the target.

2. **GPS quality columns**: `gps_eph_m`, `gps_epv_m`, `gps_satellites_used`,
   `fix_type`, `vel_ned_valid` — these reflect GPS availability and would not
   be available in a true GNSS-denied scenario.

3. **PX4 validity flags**: If these flags depend on GPS lock, they may leak
   GPS status information.

4. **`dead_reckoning`**: Directly indicates GPS-denied state. Using it as a
   feature is circular if the goal is to operate during dead reckoning.

5. **`gps_speed_m_s`**: Derived from GPS velocity.

6. **Target components as features**: Including any target column (or its
   algebraic equivalent) as a feature would constitute direct leakage.

## Fact vs Interpretation

| Claim | Status |
|-------|--------|
| Column units match PX4 conventions | **Fact** (from naming; no source code confirmation) |
| IMU data is in body FRD frame | **Interpretation** (consistent with PX4 defaults) |
| GPS and PX4 velocities are in NED frame | **Interpretation** (from PX4 convention) |
| Accelerometer data includes gravity | **Interpretation** (from mean Z ≈ −9.81) |
| `dead_reckoning = 0` means GPS is fused | **Uncertain** — flag semantics require EKF2 source review |

## Missing Provenance Materials

- Original `.ulg` log files.
- Dataset extraction script (Python / pyulog code).
- Time-synchronization method between GPS and IMU topics.
- PX4 EKF2 fusion configuration parameters.
- Recording-session metadata (vehicle, location, weather, flight mode).
- Calibration records for IMU sensors.
