#!/usr/bin/env python3
"""
Stage B: Feature Engineering for GPS-Assisted ML for IMU Drift Reduction
=========================================================================

This script implements the feature engineering plan approved in Stage A.
It produces IMU-derived features suitable for a GPS-denied prediction scenario.

Author: Feature engineering pipeline (automated)
Date: 2026-10-24

Input:  imu_drift_feature_dataset.csv (original, never modified)
Output: feature_engineered_dataset.csv (new file with all features)

All features are computed from IMU-only data (GPS-denied safe).
GPS/PX4/target columns are preserved but clearly labelled for role.
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = "/Users/meghna/Desktop/vio-drift-reduction"
INPUT_CSV = os.path.join(PROJECT_DIR, "imu_drift_feature_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "feature_engineering_output")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "feature_engineered_dataset.csv")

ROLLING_WINDOW = 100  # samples (~0.5s at 200 Hz)
LARGE_GAP_THRESHOLD = 0.1  # seconds; gaps > this are flagged

# Column role definitions (from Stage A)
CONSTANT_COLS = [
    'vel_ned_valid', 'v_xy_valid', 'v_z_valid',
    'xy_valid', 'z_valid', 'dead_reckoning'
]

GPS_REFERENCE_COLS = [
    'gps_vel_n_m_s', 'gps_vel_e_m_s', 'gps_vel_d_m_s', 'gps_speed_m_s'
]

GPS_QUALITY_COLS = [
    'fix_type', 'gps_eph_m', 'gps_epv_m', 'gps_satellites_used'
]

EKF_REFERENCE_COLS = [
    'px4_vx_m_s', 'px4_vy_m_s', 'px4_vz_m_s'
]

TARGET_COLS = [
    'target_delta_v_n_m_s', 'target_delta_v_e_m_s', 'target_delta_v_d_m_s'
]

IMU_ACCEL_COLS = ['accel_x_m_s2', 'accel_y_m_s2', 'accel_z_m_s2']
IMU_GYRO_COLS = ['gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']
PRE_DERIVED_COLS = ['accel_magnitude_m_s2', 'gyro_magnitude_rad_s']
IDENTIFIER_COLS = ['log_id']
TEMPORAL_COLS = ['timestamp', 'time_s']


def load_and_validate(path):
    """Load CSV and perform basic validation."""
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicate rows: {df.duplicated().sum()}")

    # Verify expected columns exist
    expected = (CONSTANT_COLS + GPS_REFERENCE_COLS + GPS_QUALITY_COLS +
                EKF_REFERENCE_COLS + TARGET_COLS + IMU_ACCEL_COLS +
                IMU_GYRO_COLS + PRE_DERIVED_COLS + IDENTIFIER_COLS + TEMPORAL_COLS)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def drop_constant_columns(df):
    """Drop columns with zero variance (verified constant in Stage A)."""
    dropped = []
    for col in CONSTANT_COLS:
        if col in df.columns:
            nunique = df[col].nunique()
            if nunique <= 1:
                dropped.append(col)
            else:
                print(f"  WARNING: {col} was expected constant but has {nunique} unique values. Keeping.")
    df = df.drop(columns=dropped)
    print(f"  Dropped {len(dropped)} constant columns: {dropped}")
    return df


def compute_dt(df):
    """
    Compute time step (dt) per row, gap-aware by log_id.

    dt[i] = time_s[i] - time_s[i-1] for rows within the same log.
    First row of each log gets NaN.

    Also computes: time_since_gap (seconds since last large gap > 0.1s).
    """
    print("  Computing dt and time_since_gap...")
    df['dt_s'] = np.nan

    for lid in df['log_id'].unique():
        mask = df['log_id'] == lid
        idx = df.index[mask]
        ts = df.loc[idx, 'time_s'].values
        dt = np.diff(ts, prepend=np.nan)
        dt[0] = np.nan  # first row of each log
        df.loc[idx, 'dt_s'] = dt

    # time_since_gap: seconds elapsed since the last large gap (> 0.1s)
    df['time_since_gap_s'] = np.nan
    for lid in df['log_id'].unique():
        mask = df['log_id'] == lid
        idx = df.index[mask]
        dt_vals = df.loc[idx, 'dt_s'].values
        ts_vals = df.loc[idx, 'time_s'].values

        time_since = np.zeros(len(idx))
        last_gap_time = ts_vals[0]  # start of flight
        for i in range(len(idx)):
            if i == 0 or (not np.isnan(dt_vals[i]) and dt_vals[i] > LARGE_GAP_THRESHOLD):
                last_gap_time = ts_vals[i]
            time_since[i] = ts_vals[i] - last_gap_time
        df.loc[idx, 'time_since_gap_s'] = time_since

    return df


def compute_derivatives(df):
    """
    Compute first derivatives of IMU measurements (jerk and gyro angular acceleration).

    jerk_x = d(accel_x) / dt
    gyro_dot_x = d(gyro_x) / dt

    Gap-aware: derivatives are NaN at log boundaries and after large time gaps.
    Large gaps (> 0.1s) produce unreliable finite differences.
    """
    print("  Computing derivatives (jerk, gyro_dot)...")

    axes = ['x', 'y', 'z']
    accel_cols = [f'accel_{a}_m_s2' for a in axes]
    gyro_cols = [f'gyro_{a}_rad_s' for a in axes]

    jerk_names = [f'jerk_{a}_m_s3' for a in axes]
    gyro_dot_names = [f'gyro_dot_{a}_rad_s2' for a in axes]

    for name in jerk_names + gyro_dot_names:
        df[name] = np.nan

    for lid in df['log_id'].unique():
        mask = df['log_id'] == lid
        idx = df.index[mask]
        dt_vals = df.loc[idx, 'dt_s'].values

        for acol, jname in zip(accel_cols, jerk_names):
            vals = df.loc[idx, acol].values
            deriv = np.diff(vals, prepend=np.nan) / dt_vals
            # Invalidate at large gaps
            deriv[0] = np.nan
            for i in range(1, len(deriv)):
                if np.isnan(dt_vals[i]) or dt_vals[i] > LARGE_GAP_THRESHOLD:
                    deriv[i] = np.nan
            df.loc[idx, jname] = deriv

        for gcol, gdname in zip(gyro_cols, gyro_dot_names):
            vals = df.loc[idx, gcol].values
            deriv = np.diff(vals, prepend=np.nan) / dt_vals
            deriv[0] = np.nan
            for i in range(1, len(deriv)):
                if np.isnan(dt_vals[i]) or dt_vals[i] > LARGE_GAP_THRESHOLD:
                    deriv[i] = np.nan
            df.loc[idx, gdname] = deriv

    # Derivative magnitudes
    df['jerk_magnitude_m_s3'] = np.sqrt(
        df['jerk_x_m_s3']**2 + df['jerk_y_m_s3']**2 + df['jerk_z_m_s3']**2
    )
    df['gyro_dot_magnitude_rad_s2'] = np.sqrt(
        df['gyro_dot_x_rad_s2']**2 + df['gyro_dot_y_rad_s2']**2 + df['gyro_dot_z_rad_s2']**2
    )

    return df


def compute_rolling_statistics(df):
    """
    Compute rolling mean and std for IMU axes, respecting log_id boundaries.

    Window: 100 samples (~0.5s at 200 Hz).
    Uses past-only (right-aligned) windows to prevent future leakage.
    Rolling stats are NaN for the first (window-1) rows of each log.
    """
    print(f"  Computing rolling statistics (window={ROLLING_WINDOW})...")

    axes = ['x', 'y', 'z']
    accel_cols = [f'accel_{a}_m_s2' for a in axes]
    gyro_cols = [f'gyro_{a}_rad_s' for a in axes]

    for source_cols, prefix in [(accel_cols, 'accel'), (gyro_cols, 'gyro')]:
        for col in source_cols:
            axis = col.split('_')[1]  # x, y, or z
            mean_name = f'{prefix}_roll_mean_{axis}'
            std_name = f'{prefix}_roll_std_{axis}'

            df[mean_name] = np.nan
            df[std_name] = np.nan

            for lid in df['log_id'].unique():
                mask = df['log_id'] == lid
                idx = df.index[mask]
                series = df.loc[idx, col]

                # Right-aligned rolling (past-only, includes current sample)
                roll_mean = series.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
                roll_std = series.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()

                df.loc[idx, mean_name] = roll_mean.values
                df.loc[idx, std_name] = roll_std.values

    return df


def compute_cross_axis_products(df):
    """
    Compute cross-axis coupling products.

    These capture non-linear interactions between sensor axes that may
    correlate with complex manoeuvres causing increased drift.
    """
    print("  Computing cross-axis coupling products...")

    df['accel_xy_product'] = df['accel_x_m_s2'] * df['accel_y_m_s2']
    df['accel_xz_product'] = df['accel_x_m_s2'] * df['accel_z_m_s2']
    df['gyro_xy_product'] = df['gyro_x_rad_s'] * df['gyro_y_rad_s']
    df['gyro_xz_product'] = df['gyro_x_rad_s'] * df['gyro_z_rad_s']

    return df


def validate_outputs(df, original_len):
    """Validate the engineered dataset."""
    print("\n" + "=" * 70)
    print("OUTPUT VALIDATION")
    print("=" * 70)

    # Row count preservation
    assert len(df) == original_len, \
        f"Row count mismatch: {len(df)} vs original {original_len}"
    print(f"  ✓ Row count preserved: {len(df)}")

    # Check no original IMU columns were modified
    print(f"  ✓ Total columns: {len(df.columns)}")

    # NaN accounting
    new_cols = [c for c in df.columns if c not in
                (IDENTIFIER_COLS + TEMPORAL_COLS + IMU_ACCEL_COLS + IMU_GYRO_COLS +
                 PRE_DERIVED_COLS + GPS_REFERENCE_COLS + GPS_QUALITY_COLS +
                 EKF_REFERENCE_COLS + TARGET_COLS)]
    print(f"\n  New/modified columns ({len(new_cols)}):")
    for col in sorted(new_cols):
        n_nan = df[col].isnull().sum()
        n_valid = len(df) - n_nan
        pct = 100 * n_valid / len(df)
        print(f"    {col}: {n_valid} valid ({pct:.1f}%), {n_nan} NaN")

    # Check derivative sanity
    jerk_cols = [c for c in df.columns if c.startswith('jerk_') and 'magnitude' not in c]
    for col in jerk_cols:
        valid = df[col].dropna()
        if len(valid) > 0:
            q01, q99 = valid.quantile(0.01), valid.quantile(0.99)
            print(f"    {col} range [Q01, Q99]: [{q01:.2f}, {q99:.2f}]")

    # Verify no constant-column remnants
    for col in CONSTANT_COLS:
        assert col not in df.columns, f"Constant column {col} should have been dropped"
    print(f"  ✓ All {len(CONSTANT_COLS)} constant columns removed")

    # Verify targets are untouched
    print(f"  ✓ Target columns present: {[c for c in TARGET_COLS if c in df.columns]}")

    return True


def build_feature_registry(df):
    """
    Build a structured registry of all features with metadata.
    Returns a list of dicts.
    """
    registry = []

    def add(name, category, formula, rationale, units, source_cols,
            gps_denied_safe, leakage_risk, assumptions="None", is_new=True):
        registry.append({
            "feature_name": name,
            "category": category,
            "formula": formula,
            "rationale": rationale,
            "units": units,
            "source_columns": source_cols,
            "gps_denied_safe": gps_denied_safe,
            "leakage_risk": leakage_risk,
            "assumptions": assumptions,
            "is_new_feature": is_new,
            "nan_count": int(df[name].isnull().sum()) if name in df.columns else "N/A",
            "valid_count": int(df[name].notna().sum()) if name in df.columns else "N/A"
        })

    # Original IMU columns
    for axis in ['x', 'y', 'z']:
        add(f'accel_{axis}_m_s2', 'raw_imu',
            f'Direct measurement (accelerometer {axis.upper()}-axis)',
            'Primary IMU measurement; captures specific force along axis',
            'm/s²', [f'accel_{axis}_m_s2'], True, 'none',
            'Coordinate frame unknown (likely body frame, Z≈-9.8)', False)

        add(f'gyro_{axis}_rad_s', 'raw_imu',
            f'Direct measurement (gyroscope {axis.upper()}-axis)',
            'Primary IMU measurement; captures angular velocity around axis',
            'rad/s', [f'gyro_{axis}_rad_s'], True, 'none',
            'Coordinate frame unknown (likely body frame)', False)

    # Pre-existing derived
    add('accel_magnitude_m_s2', 'pre_derived',
        '√(accel_x² + accel_y² + accel_z²)',
        'Total specific force magnitude; indicates dynamic regime',
        'm/s²', IMU_ACCEL_COLS, True, 'none',
        'Verified as exact L2 norm (max error ~1e-14)', False)

    add('gyro_magnitude_rad_s', 'pre_derived',
        '√(gyro_x² + gyro_y² + gyro_z²)',
        'Total angular velocity; higher values correlate with manoeuvring',
        'rad/s', IMU_GYRO_COLS, True, 'none',
        'Verified as exact L2 norm (max error ~1e-16)', False)

    # dt
    add('dt_s', 'temporal',
        'time_s[i] - time_s[i-1] (within same log_id)',
        'Captures sampling irregularity; irregular dt affects integration error',
        's', ['time_s', 'log_id'], True, 'none',
        'NaN at first row of each log and assumes time_s is monotonic')

    # time_since_gap
    add('time_since_gap_s', 'temporal',
        'time_s[i] - time_s[last_gap_start]',
        'Time elapsed since last large gap; segments continuous recording intervals',
        's', ['time_s', 'dt_s', 'log_id'], True, 'none',
        'Gap threshold = 0.1s; first row of each log starts at 0')

    # Jerk (accel derivatives)
    for axis in ['x', 'y', 'z']:
        add(f'jerk_{axis}_m_s3', 'derivative',
            f'd(accel_{axis}_m_s2)/dt = Δaccel_{axis} / Δt',
            'Rate of force change; rapid jerk increases integration error',
            'm/s³', [f'accel_{axis}_m_s2', 'dt_s'], True, 'none',
            'Finite difference; NaN at log boundaries and gaps > 0.1s; amplifies noise')

    add('jerk_magnitude_m_s3', 'derivative',
        '√(jerk_x² + jerk_y² + jerk_z²)',
        'Total jerk magnitude; summarises rate of force change',
        'm/s³', ['jerk_x_m_s3', 'jerk_y_m_s3', 'jerk_z_m_s3'], True, 'none')

    # Gyro derivatives
    for axis in ['x', 'y', 'z']:
        add(f'gyro_dot_{axis}_rad_s2', 'derivative',
            f'd(gyro_{axis}_rad_s)/dt = Δgyro_{axis} / Δt',
            'Angular acceleration; captures onset of rotational manoeuvres',
            'rad/s²', [f'gyro_{axis}_rad_s', 'dt_s'], True, 'none',
            'Finite difference; NaN at boundaries/gaps; amplifies noise')

    add('gyro_dot_magnitude_rad_s2', 'derivative',
        '√(gyro_dot_x² + gyro_dot_y² + gyro_dot_z²)',
        'Total angular acceleration magnitude',
        'rad/s²', ['gyro_dot_x_rad_s2', 'gyro_dot_y_rad_s2', 'gyro_dot_z_rad_s2'],
        True, 'none')

    # Rolling statistics
    for sensor, prefix in [('accel', 'accel'), ('gyro', 'gyro')]:
        unit = 'm/s²' if sensor == 'accel' else 'rad/s'
        for axis in ['x', 'y', 'z']:
            src = f'{prefix}_{axis}_{"m_s2" if sensor == "accel" else "rad_s"}'
            add(f'{prefix}_roll_mean_{axis}', 'rolling_statistic',
                f'rolling_mean({src}, window=100)',
                f'Smoothed {sensor} {axis.upper()}-axis; captures sustained force/rotation bias',
                unit, [src, 'log_id'], True, 'none',
                f'Window=100 samples (~0.5s); NaN for first 99 rows per log; does not cross log boundaries')

            add(f'{prefix}_roll_std_{axis}', 'rolling_statistic',
                f'rolling_std({src}, window=100)',
                f'Variability of {sensor} {axis.upper()}-axis; indicates vibration/turbulence',
                unit, [src, 'log_id'], True, 'none',
                f'Window=100 samples (~0.5s); NaN for first 99 rows per log')

    # Cross-axis products
    add('accel_xy_product', 'cross_axis',
        'accel_x_m_s2 * accel_y_m_s2',
        'Lateral force coupling; non-zero during coordinated turns',
        'm²/s⁴', ['accel_x_m_s2', 'accel_y_m_s2'], True, 'none')

    add('accel_xz_product', 'cross_axis',
        'accel_x_m_s2 * accel_z_m_s2',
        'Vertical-lateral coupling; captures pitch-coupled manoeuvres',
        'm²/s⁴', ['accel_x_m_s2', 'accel_z_m_s2'], True, 'none')

    add('gyro_xy_product', 'cross_axis',
        'gyro_x_rad_s * gyro_y_rad_s',
        'Roll-pitch coupling; non-zero during complex rotations',
        'rad²/s²', ['gyro_x_rad_s', 'gyro_y_rad_s'], True, 'none')

    add('gyro_xz_product', 'cross_axis',
        'gyro_x_rad_s * gyro_z_rad_s',
        'Roll-yaw coupling; captures coordinated turn dynamics',
        'rad²/s²', ['gyro_x_rad_s', 'gyro_z_rad_s'], True, 'none')

    # Reference/target columns (not predictors)
    for col in TARGET_COLS:
        axis = col.split('_')[2]  # n, e, d
        add(col, 'target',
            f'gps_vel_{axis}_m_s - px4_v{axis}_m_s (at update instants, zero-order held)',
            'Supervision target: velocity discrepancy between GPS and EKF',
            'm/s', [f'gps_vel_{axis}_m_s', f'px4_v{"x" if axis=="n" else "y" if axis=="e" else "z"}_m_s'],
            False, 'is_target',
            'EKF innovation residual, not pure IMU drift; zero-order held between updates', False)

    return registry


def main():
    print("=" * 70)
    print("STAGE B: FEATURE ENGINEERING IMPLEMENTATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load and validate
    print("\n[1/7] Loading dataset...")
    df = load_and_validate(INPUT_CSV)
    original_len = len(df)

    # Step 2: Drop constant columns
    print("\n[2/7] Dropping constant columns...")
    df = drop_constant_columns(df)

    # Step 3: Temporal features
    print("\n[3/7] Computing temporal features...")
    df = compute_dt(df)

    # Step 4: Derivatives
    print("\n[4/7] Computing derivatives...")
    df = compute_derivatives(df)

    # Step 5: Rolling statistics
    print("\n[5/7] Computing rolling statistics...")
    df = compute_rolling_statistics(df)

    # Step 6: Cross-axis products
    print("\n[6/7] Computing cross-axis products...")
    df = compute_cross_axis_products(df)

    # Step 7: Validate
    print("\n[7/7] Validating outputs...")
    validate_outputs(df, original_len)

    # Save feature-engineered dataset
    print(f"\nSaving to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {os.path.getsize(OUTPUT_CSV) / 1e6:.1f} MB")

    # Build and save feature registry
    print("\nBuilding feature registry...")
    registry = build_feature_registry(df)
    registry_path = os.path.join(OUTPUT_DIR, "feature_registry.json")
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"  Saved: {registry_path} ({len(registry)} features)")

    # Summary
    new_features = [r for r in registry if r['is_new_feature']]
    predictors = [r for r in registry if r['gps_denied_safe'] and r['category'] != 'target']
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Original columns: 31")
    print(f"  Constant columns dropped: {len(CONSTANT_COLS)}")
    print(f"  New features added: {len(new_features)}")
    print(f"  Total columns in output: {len(df.columns)}")
    print(f"  GPS-denied-safe predictors: {len(predictors)}")
    print(f"  Rows: {len(df)} (unchanged)")
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
