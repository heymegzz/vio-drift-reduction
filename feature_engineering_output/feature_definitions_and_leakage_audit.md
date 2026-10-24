# Feature Definitions & Leakage Audit

## Column Role Classification

### Identifiers (not used as features)
| Column | Purpose |
|--------|---------|
| `log_id` | Flight segment identifier (4 flights); used for grouping/splitting only |

### Temporal Columns
| Column | Purpose | GPS-Denied Safe |
|--------|---------|:---:|
| `timestamp` | Microsecond-scale timestamp (likely µs from boot) | ✅ |
| `time_s` | Elapsed time in seconds, resets per log | ✅ |

### GPS-Denied Safe Predictors (34 features)

#### Raw IMU (8 features — original, unmodified)
| Feature | Units | Formula | Rationale |
|---------|-------|---------|-----------|
| `accel_x_m_s2` | m/s² | Direct measurement | Specific force, X-axis |
| `accel_y_m_s2` | m/s² | Direct measurement | Specific force, Y-axis |
| `accel_z_m_s2` | m/s² | Direct measurement | Specific force, Z-axis |
| `gyro_x_rad_s` | rad/s | Direct measurement | Angular velocity, X-axis |
| `gyro_y_rad_s` | rad/s | Direct measurement | Angular velocity, Y-axis |
| `gyro_z_rad_s` | rad/s | Direct measurement | Angular velocity, Z-axis |
| `accel_magnitude_m_s2` | m/s² | √(ax² + ay² + az²) | Total specific force; pre-existing, verified |
| `gyro_magnitude_rad_s` | rad/s | √(gx² + gy² + gz²) | Total angular velocity; pre-existing, verified |

#### Temporal Features (2 new features)
| Feature | Units | Formula | Rationale | NaN Pattern |
|---------|-------|---------|-----------|-------------|
| `dt_s` | s | time_s[i] − time_s[i−1] | Captures sampling irregularity; irregular dt affects numerical integration error | NaN at first row of each log (4 rows total) |
| `time_since_gap_s` | s | time_s[i] − time_s[last gap start] | Segments continuous recording intervals; resets at gaps > 0.1s | No NaN |

#### Derivatives (8 new features)
| Feature | Units | Formula | Rationale | NaN Pattern |
|---------|-------|---------|-----------|-------------|
| `jerk_x_m_s3` | m/s³ | Δ(accel_x)/Δt | Rate of force change, X-axis; rapid jerk increases integration error | 41 NaN (log boundaries + large gaps) |
| `jerk_y_m_s3` | m/s³ | Δ(accel_y)/Δt | Same, Y-axis | 41 NaN |
| `jerk_z_m_s3` | m/s³ | Δ(accel_z)/Δt | Same, Z-axis | 41 NaN |
| `jerk_magnitude_m_s3` | m/s³ | √(jx² + jy² + jz²) | Total jerk magnitude | 41 NaN |
| `gyro_dot_x_rad_s2` | rad/s² | Δ(gyro_x)/Δt | Angular acceleration, X-axis; captures manoeuvre onset | 41 NaN |
| `gyro_dot_y_rad_s2` | rad/s² | Δ(gyro_y)/Δt | Same, Y-axis | 41 NaN |
| `gyro_dot_z_rad_s2` | rad/s² | Δ(gyro_z)/Δt | Same, Z-axis | 41 NaN |
| `gyro_dot_magnitude_rad_s2` | rad/s² | √(gdx² + gdy² + gdz²) | Total angular acceleration magnitude | 41 NaN |

> **Note on derivative NaN**: 4 NaN from log boundaries (first row of each flight) + 37 NaN from large time gaps (>0.1s) where finite differences are unreliable. Total: 41 out of 38,967 rows (0.1%).

#### Rolling Statistics (12 new features, window = 100 samples ≈ 0.5s)
| Feature | Units | Formula | Rationale | NaN Pattern |
|---------|-------|---------|-----------|-------------|
| `accel_roll_mean_x` | m/s² | rolling_mean(accel_x, w=100) | Smoothed force bias, X-axis | 396 NaN |
| `accel_roll_mean_y` | m/s² | rolling_mean(accel_y, w=100) | Same, Y-axis | 396 NaN |
| `accel_roll_mean_z` | m/s² | rolling_mean(accel_z, w=100) | Same, Z-axis | 396 NaN |
| `accel_roll_std_x` | m/s² | rolling_std(accel_x, w=100) | Vibration/turbulence intensity, X | 396 NaN |
| `accel_roll_std_y` | m/s² | rolling_std(accel_y, w=100) | Same, Y | 396 NaN |
| `accel_roll_std_z` | m/s² | rolling_std(accel_z, w=100) | Same, Z | 396 NaN |
| `gyro_roll_mean_x` | rad/s | rolling_mean(gyro_x, w=100) | Smoothed rotation rate, X-axis | 396 NaN |
| `gyro_roll_mean_y` | rad/s | rolling_mean(gyro_y, w=100) | Same, Y | 396 NaN |
| `gyro_roll_mean_z` | rad/s | rolling_mean(gyro_z, w=100) | Same, Z | 396 NaN |
| `gyro_roll_std_x` | rad/s | rolling_std(gyro_x, w=100) | Rotation variability, X | 396 NaN |
| `gyro_roll_std_y` | rad/s | rolling_std(gyro_y, w=100) | Same, Y | 396 NaN |
| `gyro_roll_std_z` | rad/s | rolling_std(gyro_z, w=100) | Same, Z | 396 NaN |

> **Note on rolling NaN**: 99 NaN per log × 4 logs = 396 rows where the window is incomplete (first 99 rows of each flight). These are **not filled or imputed**; they represent genuinely unavailable data.

> **Window justification**: 100 samples at ~200 Hz ≈ 0.5 seconds. Spans 2-3 GPS update epochs. Short enough for manoeuvre transitions, long enough to smooth IMU noise. Past-only (right-aligned) windows prevent future leakage.

#### Cross-Axis Coupling (4 new features)
| Feature | Units | Formula | Rationale | NaN Pattern |
|---------|-------|---------|-----------|-------------|
| `accel_xy_product` | m²/s⁴ | accel_x × accel_y | Lateral force coupling; non-zero during coordinated turns | No NaN |
| `accel_xz_product` | m²/s⁴ | accel_x × accel_z | Vertical-lateral coupling; pitch manoeuvres | No NaN |
| `gyro_xy_product` | rad²/s² | gyro_x × gyro_y | Roll-pitch coupling; complex rotations | No NaN |
| `gyro_xz_product` | rad²/s² | gyro_x × gyro_z | Roll-yaw coupling; coordinated turns | No NaN |

---

### Excluded from Predictors (GPS-denied unsafe)

#### GPS Reference (excluded from predictors — used for target construction only)
| Column | Reason for Exclusion |
|--------|---------------------|
| `gps_vel_n_m_s` | GPS velocity — unavailable in GPS-denied; directly constructs target |
| `gps_vel_e_m_s` | Same |
| `gps_vel_d_m_s` | Same |
| `gps_speed_m_s` | GPS ground speed — unavailable in GPS-denied |

#### GPS Quality Metadata (excluded from predictors)
| Column | Reason for Exclusion |
|--------|---------------------|
| `fix_type` | GPS receiver status — no GPS fix in denial scenario |
| `gps_eph_m` | GPS horizontal accuracy — unavailable without GPS |
| `gps_epv_m` | GPS vertical accuracy — same |
| `gps_satellites_used` | Satellite count — same |

#### PX4 EKF Velocity (excluded from predictors — indirect GPS leakage)
| Column | Reason for Exclusion |
|--------|---------------------|
| `px4_vx_m_s` | PX4 EKF fuses GPS; velocity contains GPS information |
| `px4_vy_m_s` | Same |
| `px4_vz_m_s` | Same |

#### Dropped Columns (zero information)
| Column | Reason |
|--------|--------|
| `vel_ned_valid` | Constant = 1.0 across all 38,967 rows |
| `v_xy_valid` | Constant = 1.0 |
| `v_z_valid` | Constant = 1.0 |
| `xy_valid` | Constant = 1.0 |
| `z_valid` | Constant = 1.0 |
| `dead_reckoning` | Constant = 0.0 |

### Targets (supervision only — never used as predictors)
| Column | Definition | Interpretation |
|--------|-----------|----------------|
| `target_delta_v_n_m_s` | gps_vel_n − px4_vx | North velocity discrepancy (m/s) |
| `target_delta_v_e_m_s` | gps_vel_e − px4_vy | East velocity discrepancy (m/s) |
| `target_delta_v_d_m_s` | gps_vel_d − px4_vz | Down velocity discrepancy (m/s) |

---

## Leakage Audit Summary

| Leakage Type | Mitigation |
|-------------|------------|
| **Target leakage** (GPS velocity in predictors) | GPS velocity columns excluded from predictor set |
| **Indirect leakage** (PX4 EKF fuses GPS) | PX4 velocity excluded from predictors |
| **Future leakage** (rolling windows using future data) | All rolling windows are right-aligned (past-only) |
| **Cross-flight leakage** (rolling windows crossing logs) | All rolling/derivative computations reset at `log_id` boundaries |
| **Gap leakage** (derivatives across large gaps) | Derivatives set to NaN at gaps > 0.1s |
| **Target as predictor** | Target columns clearly labelled; never included in predictor set |
