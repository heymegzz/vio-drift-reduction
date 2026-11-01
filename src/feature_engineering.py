"""
Feature Engineering Pipeline for VIO Drift Reduction
=====================================================

Transforms the raw IMU dataset into a model-ready feature matrix by:
  1. Removing GPS/PX4 leaky columns (GNSS-denied scenario)
  2. Removing constant-value columns (no discriminative information)
  3. Engineering temporal features (rolling stats, differences, lags)
  4. Preserving log-boundary integrity (no cross-log contamination)

Usage:
    python feature_engineering.py                      # uses defaults
    python feature_engineering.py --input data.csv --output features.csv

Output:
    - feature_engineered_dataset.csv   (full featured dataset)
    - Console summary of new features
"""

import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message=".*fragmented.*")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from src.utils import IMU_FEATURES, LEAKY_COLUMNS, TARGET_COLUMNS


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

RAW_IMU_CHANNELS = [
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
]

MAGNITUDES = ["accel_magnitude_m_s2", "gyro_magnitude_rad_s"]

ROLLING_WINDOWS = [10, 50, 100]          # ~50ms, 250ms, 500ms at 200 Hz
LAG_STEPS = [1, 5, 10]                   # ~5ms, 25ms, 50ms

# Columns that are constant across the entire dataset (identified in EDA)
CONSTANT_COLUMNS = [
    "vel_ned_valid", "v_xy_valid", "v_z_valid",
    "xy_valid", "z_valid", "dead_reckoning",
]


# ──────────────────────────────────────────────────────────────────────
# Feature engineering functions
# ──────────────────────────────────────────────────────────────────────

def add_rolling_features(group: pd.DataFrame, columns: list,
                         windows: list) -> pd.DataFrame:
    """
    Add rolling mean and std for each column at each window size.

    Parameters
    ----------
    group : DataFrame for a single log (sorted by time)
    columns : sensor columns to compute rolling stats for
    windows : list of window sizes (in rows)

    Notes
    -----
    Uses min_periods=1 so early rows get partial-window stats rather
    than NaN.  Rolling is computed per-log to avoid cross-log leakage.
    """
    for col in columns:
        for w in windows:
            group[f"{col}_rmean_{w}"] = (
                group[col].rolling(window=w, min_periods=1).mean()
            )
            group[f"{col}_rstd_{w}"] = (
                group[col].rolling(window=w, min_periods=1).std().fillna(0)
            )
    return group


def add_difference_features(group: pd.DataFrame,
                            columns: list) -> pd.DataFrame:
    """
    Add first-order differences (rate of change) for each column.

    For accelerometer channels this approximates *jerk* (da/dt).
    For gyroscope channels this approximates angular acceleration.
    """
    for col in columns:
        group[f"{col}_diff1"] = group[col].diff().fillna(0)
    return group


def add_lag_features(group: pd.DataFrame, columns: list,
                     lags: list) -> pd.DataFrame:
    """
    Add lagged values of each column.

    Lags are computed per-log to prevent cross-log leakage.
    """
    for col in columns:
        for lag in lags:
            group[f"{col}_lag{lag}"] = group[col].shift(lag).bfill()
    return group


def add_jerk_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Add jerk (second-order differences of acceleration).

    jerk = d²a/dt²  ≈  diff(diff(accel))
    """
    accel_cols = ["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]
    for col in accel_cols:
        group[f"{col}_jerk"] = group[col].diff().diff().fillna(0)
    return group


def add_cross_axis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between sensor axes.

    - Accel XY angle (atan2)
    - Gyro XY angle (atan2)
    - Accel-Gyro magnitude ratio
    """
    df["accel_xy_angle"] = np.arctan2(
        df["accel_y_m_s2"], df["accel_x_m_s2"]
    )
    df["gyro_xy_angle"] = np.arctan2(
        df["gyro_y_rad_s"], df["gyro_x_rad_s"]
    )
    # Ratio of accel magnitude to gyro magnitude (add epsilon to avoid div/0)
    df["accel_gyro_mag_ratio"] = (
        df["accel_magnitude_m_s2"] / (df["gyro_magnitude_rad_s"] + 1e-8)
    )
    return df


def add_rolling_range_features(group: pd.DataFrame, columns: list,
                                windows: list) -> pd.DataFrame:
    """Add rolling min, max, and range (max-min) features."""
    for col in columns:
        for w in windows:
            rmin = group[col].rolling(window=w, min_periods=1).min()
            rmax = group[col].rolling(window=w, min_periods=1).max()
            group[f"{col}_rmin_{w}"] = rmin
            group[f"{col}_rmax_{w}"] = rmax
            group[f"{col}_rrange_{w}"] = rmax - rmin
    return group


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    df : raw dataset DataFrame
    verbose : print progress

    Returns
    -------
    DataFrame with engineered features, sorted by log_id and time.
    """
    n_original = len(df.columns)
    if verbose:
        print(f"Starting feature engineering ({len(df):,} rows, "
              f"{n_original} columns)")

    # 1. Sort by log and time
    df = df.sort_values(["log_id", "timestamp"]).reset_index(drop=True)

    # 2. Remove leaky and constant columns
    cols_to_drop = [c for c in LEAKY_COLUMNS + CONSTANT_COLUMNS
                    if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    if verbose:
        print(f"  Dropped {len(cols_to_drop)} leaky/constant columns")

    # 3. Per-log temporal features (to prevent cross-log leakage)
    sensor_cols = RAW_IMU_CHANNELS + MAGNITUDES

    # Apply all per-log transforms in a single groupby pass to avoid
    # repeated groupby overhead and column-drop issues.
    def _per_log_transforms(group):
        group = add_rolling_features(group, sensor_cols, ROLLING_WINDOWS)
        group = add_rolling_range_features(group, RAW_IMU_CHANNELS,
                                           ROLLING_WINDOWS)
        group = add_difference_features(group, sensor_cols)
        group = add_lag_features(group, RAW_IMU_CHANNELS, LAG_STEPS)
        group = add_jerk_features(group)
        return group

    if verbose:
        print("  Computing per-log temporal features "
              "(rolling, range, diff, lag, jerk)...")
    # Save log_id before groupby since include_groups=False drops it
    log_id_series = df["log_id"].copy()
    df = df.groupby("log_id", group_keys=False).apply(
        _per_log_transforms, include_groups=False
    )
    # Restore log_id column
    df.insert(0, "log_id", log_id_series)
    # Defragment the dataframe after many column insertions
    df = df.copy()

    # 4. Cross-axis features (row-wise, no temporal leakage risk)
    if verbose:
        print("  Computing cross-axis features...")
    df = add_cross_axis_features(df)

    # 5. Final checks
    n_new = len(df.columns)
    n_engineered = n_new - (n_original - len(cols_to_drop))
    assert df.isna().sum().sum() == 0, "NaN values found after engineering!"

    if verbose:
        print(f"\nFeature engineering complete:")
        print(f"  Original columns:    {n_original}")
        print(f"  Dropped columns:     {len(cols_to_drop)}")
        print(f"  Engineered features: {n_engineered}")
        print(f"  Final columns:       {n_new}")
        # List feature groups
        feature_cols = [c for c in df.columns
                        if c not in ["log_id", "timestamp", "time_s"]
                        + TARGET_COLUMNS]
        print(f"  Total input features: {len(feature_cols)}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of input feature column names (excludes meta & targets)."""
    exclude = {"log_id", "timestamp", "time_s"} | set(TARGET_COLUMNS)
    return [c for c in df.columns if c not in exclude]


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feature engineering for VIO drift reduction"
    )
    parser.add_argument(
        "--input", default="data/imu_drift_feature_dataset.csv",
        help="Path to raw CSV dataset"
    )
    parser.add_argument(
        "--output", default="feature_engineered_dataset.csv",
        help="Path to save the engineered dataset"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = engineer_features(df)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # Print feature summary
    feature_cols = get_feature_columns(df)
    print(f"\nFeature summary ({len(feature_cols)} features):")
    groups = {}
    for c in feature_cols:
        if "_rmean_" in c or "_rstd_" in c:
            groups.setdefault("Rolling mean/std", []).append(c)
        elif "_rmin_" in c or "_rmax_" in c or "_rrange_" in c:
            groups.setdefault("Rolling range", []).append(c)
        elif "_diff1" in c:
            groups.setdefault("First differences", []).append(c)
        elif "_lag" in c:
            groups.setdefault("Lag features", []).append(c)
        elif "_jerk" in c:
            groups.setdefault("Jerk", []).append(c)
        elif c in IMU_FEATURES:
            groups.setdefault("Raw IMU", []).append(c)
        else:
            groups.setdefault("Cross-axis / other", []).append(c)
    for group_name, cols in groups.items():
        print(f"  {group_name}: {len(cols)}")


if __name__ == "__main__":
    main()
