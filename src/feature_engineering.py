"""
Feature Engineering Pipeline for VIO Drift Reduction
=====================================================

Transforms the raw IMU dataset into a model-ready feature matrix by:
  1. Removing GPS/PX4 leaky columns (GNSS-denied scenario)
  2. Removing constant-value columns (no discriminative information)
  3. Detecting temporal gaps and segmenting each log into continuous runs
  4. Engineering temporal features (rolling stats, differences, lags, jerk)
     with proper segment-boundary resets
  5. Preserving log-boundary integrity (no cross-log contamination)

Gap handling:
  Timestamps are in microseconds (µs).  The nominal cadence is ~200 Hz
  (median dt ≈ 5000 µs).  A GAP_THRESHOLD_US of 50 000 µs (50 ms, ~10×
  the median interval) is used to split each log into continuous segments.
  Rolling windows, lags, differences, and jerk are reset at segment
  boundaries so that no feature crosses a temporal discontinuity.

Window convention:
  Rolling windows are specified in *samples* (not milliseconds).  At the
  nominal 200 Hz cadence: window=10 ≈ 50 ms, window=50 ≈ 250 ms,
  window=100 ≈ 500 ms.  Actual duration depends on the true sampling
  interval within each segment.

Jerk computation:
  jerk = da/dt, computed as diff(accel) / diff(timestamp), using the
  actual elapsed time between consecutive samples.

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

ROLLING_WINDOWS = [10, 50, 100]          # samples (~50ms, 250ms, 500ms at 200 Hz)
LAG_STEPS = [1, 5, 10]                   # samples (~5ms, 25ms, 50ms)

# Gap detection: intervals > 50 ms are treated as temporal discontinuities.
# Derived from training-data analysis: nominal dt ≈ 5 000 µs (200 Hz) with
# occasional ~2.3 s gaps between GPS update bursts.  50 ms ≈ 10× median dt.
GAP_THRESHOLD_US = 50_000  # microseconds

# Columns that are constant across the entire dataset (identified in EDA)
CONSTANT_COLUMNS = [
    "vel_ned_valid", "v_xy_valid", "v_z_valid",
    "xy_valid", "z_valid", "dead_reckoning",
]


# ──────────────────────────────────────────────────────────────────────
# Segment assignment
# ──────────────────────────────────────────────────────────────────────

def assign_segments(group: pd.DataFrame,
                    gap_threshold_us: float = GAP_THRESHOLD_US) -> pd.DataFrame:
    """
    Assign a segment ID to each row within a single log.

    A new segment starts whenever the timestamp gap exceeds
    gap_threshold_us.  The segment column is used downstream to
    reset rolling windows, lags, and differences at boundaries.
    """
    dt = group["timestamp"].diff()
    # First row of each log always starts segment 0
    is_gap = (dt > gap_threshold_us).fillna(False)
    group["_segment"] = is_gap.cumsum()
    return group


# ──────────────────────────────────────────────────────────────────────
# Feature engineering functions
# ──────────────────────────────────────────────────────────────────────

def add_rolling_features(segment: pd.DataFrame, columns: list,
                         windows: list) -> pd.DataFrame:
    """
    Add rolling mean and std for each column at each window size.

    Parameters
    ----------
    segment : DataFrame for a single continuous segment (sorted by time)
    columns : sensor columns to compute rolling stats for
    windows : list of window sizes (in samples)

    Notes
    -----
    Uses min_periods=1 so early rows within a segment get partial-window
    stats rather than NaN.  Because this is called per-segment, no
    rolling window crosses a temporal gap.
    """
    for col in columns:
        for w in windows:
            segment[f"{col}_rmean_{w}"] = (
                segment[col].rolling(window=w, min_periods=1).mean()
            )
            segment[f"{col}_rstd_{w}"] = (
                segment[col].rolling(window=w, min_periods=1).std().fillna(0)
            )
    return segment


def add_difference_features(segment: pd.DataFrame,
                            columns: list) -> pd.DataFrame:
    """
    Add first-order differences (rate of change) for each column.

    The first row of each segment receives 0 (no prior sample to
    difference against within this continuous run).
    """
    for col in columns:
        segment[f"{col}_diff1"] = segment[col].diff().fillna(0)
    return segment


def add_lag_features(segment: pd.DataFrame, columns: list,
                     lags: list) -> pd.DataFrame:
    """
    Add lagged values of each column.

    Lags are computed per-segment so they never reach across temporal
    gaps.  Startup rows where the lag is unavailable are forward-filled
    (using the most recent available past value, which is causal), then
    any remaining leading NaN (no past exists) are filled with the
    current value (equivalent to lag=0 at the segment start).
    """
    for col in columns:
        for lag in lags:
            lagged = segment[col].shift(lag)
            # ffill is causal: propagate last known past value forward
            lagged = lagged.ffill()
            # For the very first rows where no past exists, use current value
            lagged = lagged.fillna(segment[col])
            segment[f"{col}_lag{lag}"] = lagged
    return segment


def add_jerk_features(segment: pd.DataFrame) -> pd.DataFrame:
    """
    Add jerk = da/dt for each accelerometer axis.

    Uses actual elapsed time (dt in seconds) between consecutive
    timestamps, so the result has physical units of m/s³.  The first
    row of each segment receives 0 (no prior sample).
    """
    accel_cols = ["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]
    # Elapsed time in seconds between consecutive rows
    dt_s = segment["timestamp"].diff() / 1e6  # µs → s
    # Avoid division by zero for the first row
    dt_s = dt_s.replace(0, np.nan)
    for col in accel_cols:
        da = segment[col].diff()
        segment[f"{col}_jerk"] = (da / dt_s).fillna(0)
    return segment


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


def add_rolling_range_features(segment: pd.DataFrame, columns: list,
                                windows: list) -> pd.DataFrame:
    """Add rolling min, max, and range (max-min) features."""
    for col in columns:
        for w in windows:
            rmin = segment[col].rolling(window=w, min_periods=1).min()
            rmax = segment[col].rolling(window=w, min_periods=1).max()
            segment[f"{col}_rmin_{w}"] = rmin
            segment[f"{col}_rmax_{w}"] = rmax
            segment[f"{col}_rrange_{w}"] = rmax - rmin
    return segment


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

    # 3. Assign segments within each log (gap detection)
    #    Compute directly to avoid groupby column-dropping issues
    segment_ids = []
    for _, grp in df.groupby("log_id", sort=False):
        dt = grp["timestamp"].diff()
        is_gap = (dt > GAP_THRESHOLD_US).fillna(False)
        segment_ids.append(is_gap.cumsum())
    df["_segment"] = pd.concat(segment_ids)

    if verbose:
        n_segments = df.groupby("log_id")["_segment"].nunique().sum()
        print(f"  Detected {n_segments} continuous segments "
              f"across {df['log_id'].nunique()} logs "
              f"(gap threshold = {GAP_THRESHOLD_US/1000:.0f} ms)")

    # 4. Per-segment temporal features (to prevent cross-gap leakage)
    sensor_cols = RAW_IMU_CHANNELS + MAGNITUDES

    def _per_segment_transforms(segment):
        segment = add_rolling_features(segment, sensor_cols, ROLLING_WINDOWS)
        segment = add_rolling_range_features(segment, RAW_IMU_CHANNELS,
                                             ROLLING_WINDOWS)
        segment = add_difference_features(segment, sensor_cols)
        segment = add_lag_features(segment, RAW_IMU_CHANNELS, LAG_STEPS)
        segment = add_jerk_features(segment)
        return segment

    if verbose:
        print("  Computing per-segment temporal features "
              "(rolling, range, diff, lag, jerk)...")

    # Group by both log_id and _segment so features don't cross gaps
    # Save grouping columns before apply drops them
    log_id_series = df["log_id"].copy()
    segment_series = df["_segment"].copy()
    df = df.groupby(["log_id", "_segment"], group_keys=False).apply(
        _per_segment_transforms, include_groups=False
    )
    # Restore grouping columns
    df.insert(0, "log_id", log_id_series)

    # _segment is metadata — do not pass to models
    # (it was already consumed by the groupby above)

    # Defragment the dataframe after many column insertions
    df = df.copy()

    # 5. Cross-axis features (row-wise, no temporal leakage risk)
    if verbose:
        print("  Computing cross-axis features...")
    df = add_cross_axis_features(df)

    # 6. Final checks
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
            groups.setdefault("Jerk (da/dt)", []).append(c)
        elif c in IMU_FEATURES:
            groups.setdefault("Raw IMU", []).append(c)
        else:
            groups.setdefault("Cross-axis / other", []).append(c)
    for group_name, cols in groups.items():
        print(f"  {group_name}: {len(cols)}")


if __name__ == "__main__":
    main()
