# Target Variable Definition

## Overview

The dataset contains three target columns that represent velocity residuals
between the PX4 estimator output and GPS reference measurements:

| Target Column            | Definition                            |
|--------------------------|---------------------------------------|
| `target_delta_v_n_m_s`   | `px4_vx_m_s − gps_vel_n_m_s`         |
| `target_delta_v_e_m_s`   | `px4_vy_m_s − gps_vel_e_m_s`         |
| `target_delta_v_d_m_s`   | `px4_vz_m_s − gps_vel_d_m_s`         |

## Verified Computation

The target values were numerically verified against the formula
`target = PX4_velocity − GPS_velocity` for all three NED components.
Maximum absolute discrepancy across all rows is below machine epsilon
(~1e-15), confirming the targets are computed exactly as defined.

## Interpretation

### What the targets represent

These values are the **instantaneous velocity difference** between the PX4
state estimator output and the GPS velocity measurement at each row.

- A positive residual means PX4 estimates a higher velocity than GPS reports.
- A negative residual means PX4 estimates a lower velocity than GPS reports.

### What the targets do NOT represent

1. **Not pure IMU drift**: The PX4 Extended Kalman Filter (EKF2) fuses
   multiple sensors, potentially including GPS itself. Without verifying the
   estimator fusion configuration and the `dead_reckoning` flag behaviour,
   `px4_v*` may already incorporate GPS corrections. The residual therefore
   reflects the *combined estimator error*, not isolated IMU drift.

2. **Not sensor bias**: Sensor bias is a slowly varying offset in raw IMU
   readings. The velocity residual is downstream of integration and filtering
   and cannot be directly equated with accelerometer or gyroscope bias.

3. **Not position error**: These are velocity-domain residuals (m/s), not
   position-domain errors (m). Velocity residuals do not directly measure
   position drift, though they are related through integration.

4. **Not successive-time velocity increments**: Each target value is a
   single-epoch difference between two velocity estimates, not the change
   in velocity between consecutive time steps.

## Correction Sign

If a model learns to predict `target_delta_v = px4_v − gps_v`, the
corrected velocity estimate would be:

```
v_corrected = px4_v − predicted_delta_v
```

This subtracts the predicted residual from the PX4 estimate, pushing it
toward the GPS reference value.

**Important**: This correction has not been implemented or validated in
this EDA stage. The formula above documents the intended sign convention
only.

## GPS as Reference

GPS velocity is used as the reference measurement, but it is **not perfect
truth**. GPS velocity has its own uncertainty governed by:

- Satellite geometry (DOP).
- Number of visible satellites (`gps_satellites_used`).
- Horizontal and vertical precision estimates (`gps_eph_m`, `gps_epv_m`).
- Multipath and atmospheric effects.

The target residuals therefore include both PX4 estimator errors and GPS
measurement noise.

## Open Questions

1. Does `px4_v*` incorporate GPS corrections via the EKF2 fusion?
2. What is the actual `dead_reckoning` flag behaviour in this dataset
   (it is always 0.0)?
3. Are rows with lower GPS quality (`fix_type=3`) affecting target
   reliability?
4. What is the time alignment between GPS and PX4 velocity samples?

## GPS Feature Removal ≠ GPS Independence

Removing GPS-derived columns (velocity, DOP, satellite count) from the
ML model's **input features** does not remove GPS dependence from the
**recorded PX4 estimator output**.  During the flight, the PX4 EKF2
likely fused GPS into its velocity estimate.  The targets
(`px4_v − gps_v`) were computed from that GPS-aided estimate.

Consequently:

- The ML model predicts the residual between a **GPS-aided** estimator
  and the GPS reference, not the residual of a pure IMU integration.
- Claiming that this system validates "GNSS-denied drift reduction" is
  **not supported** by the available data.  The correct description is
  *PX4–GPS velocity residual prediction using IMU-derived features*.
- To validate genuine GNSS-denied performance, one would need either:
  1. PX4 logs recorded with GPS fusion explicitly disabled; or
  2. A simulated GPS outage using dead-reckoning segments where
     `dead_reckoning = 1`, which does not occur in this dataset.

## Phase 1 Correction Results

In Phase 1, the velocity correction `v_corrected = px4_v − predicted_delta_v`
was evaluated on the test flight (log_54).  Whether the correction improves or
worsens the velocity estimate compared to zero correction (leaving PX4
uncorrected) is reported in `results/models/correction_evaluation.csv` and
`results/models/model_comparison_report.md`.  No model is guaranteed to beat
zero correction on this test flight.
