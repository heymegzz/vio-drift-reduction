# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # IMU Drift Feature Dataset — Exploratory Data Analysis
#
# **Project**: ML-Assisted IMU Drift Reduction for GNSS-Denied UAV Navigation
#
# **Purpose**: Thoroughly characterize the dataset before any modelling:
# - Verify data integrity and provenance.
# - Audit schema, quality, and sampling.
# - Profile sensor, reference, and estimator signals.
# - Verify target variable definitions.
# - Visualize distributions, time-series behaviour, and inter-signal relationships.
#
# **Scope limitations**: This notebook performs EDA only. No model training,
# feature engineering, data cleaning, or outlier removal is applied.
# Diagnostic quantities (e.g., timestamp differences) are computed for
# analysis and are not part of a feature-engineering pipeline.

# %% [markdown]
# ---
# ## 1. Setup and Dataset Loading

# %%
import os
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid", context="notebook", palette="deep")

FIGURES_DIR = Path("../results/eda/figures")
TABLES_DIR  = Path("../results/eda/tables")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### 1.1 Locate and load the dataset

# %%
DATASET_PATH = Path("../imu_drift_feature_dataset.csv")
assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH.resolve()}"

file_size_bytes = DATASET_PATH.stat().st_size
file_size_mb = file_size_bytes / (1024 * 1024)

sha256 = hashlib.sha256()
with open(DATASET_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 16), b""):
        sha256.update(chunk)
dataset_checksum = sha256.hexdigest()

print(f"Dataset path : {DATASET_PATH.resolve()}")
print(f"File size    : {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")
print(f"SHA-256      : {dataset_checksum}")

# %%
df = pd.read_csv(DATASET_PATH)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# %%
df.head(3)

# %%
df.tail(3)

# %%
df.sample(3, random_state=42)

# %% [markdown]
# ---
# ## 2. Schema Inspection

# %%
schema_info = pd.DataFrame({
    "dtype": df.dtypes,
    "non_null": df.count(),
    "null_count": df.isnull().sum(),
    "n_unique": df.nunique(),
    "example_value": df.iloc[0],
})
schema_info["is_constant"] = schema_info["n_unique"] <= 1
print(schema_info.to_string())

# %%
numeric_cols = df.select_dtypes(include="number").columns.tolist()
nonnumeric_cols = df.select_dtypes(exclude="number").columns.tolist()
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Non-numeric columns ({len(nonnumeric_cols)}): {nonnumeric_cols}")

# %%
# Constant columns
constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
print(f"Constant columns: {constant_cols if constant_cols else 'None'}")

# %%
schema_info.to_csv(TABLES_DIR / "schema_summary.csv")
print(f"Schema summary saved to {TABLES_DIR / 'schema_summary.csv'}")

# %% [markdown]
# ---
# ## 3. Missing Values, Infinities, and Duplicates

# %%
# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0]
if missing_report.empty:
    print("No missing values found in any column.")
else:
    print(missing_report)

# %%
# Infinities (positive and negative) in numeric columns
inf_report = pd.DataFrame({
    "pos_inf": (df[numeric_cols] == np.inf).sum(),
    "neg_inf": (df[numeric_cols] == -np.inf).sum(),
})
inf_report["total_inf"] = inf_report["pos_inf"] + inf_report["neg_inf"]
inf_nonzero = inf_report[inf_report["total_inf"] > 0]
if inf_nonzero.empty:
    print("No infinite values found in any numeric column.")
else:
    print(inf_nonzero)

# %%
# Exact duplicate rows
n_dup_rows = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dup_rows}")
if n_dup_rows > 0:
    print("First 5 duplicated rows:")
    print(df[df.duplicated(keep=False)].head(5))

# %%
# Duplicate timestamps within each log
if "log_id" in df.columns and "timestamp" in df.columns:
    dup_ts = df.groupby("log_id")["timestamp"].apply(
        lambda s: s.duplicated().sum()
    )
    print("Duplicate timestamps within each log:")
    print(dup_ts.to_string())
    if dup_ts.sum() == 0:
        print("No duplicate timestamps within any log.")

# %%
quality_report = pd.DataFrame({
    "missing": df.isnull().sum(),
    "pos_inf": [(df[c] == np.inf).sum() if c in numeric_cols else 0 for c in df.columns],
    "neg_inf": [(df[c] == -np.inf).sum() if c in numeric_cols else 0 for c in df.columns],
    "n_unique": df.nunique(),
})
quality_report.to_csv(TABLES_DIR / "quality_audit.csv")
print(f"Quality audit saved to {TABLES_DIR / 'quality_audit.csv'}")

# %% [markdown]
# ---
# ## 4. Recording Groups and Timestamp Analysis

# %%
logs = df["log_id"].unique()
print(f"Number of logs: {len(logs)}")
for lid in logs:
    print(f"  {lid}")

# %%
log_summary_rows = []
for lid in logs:
    grp = df[df["log_id"] == lid].copy()
    ts = grp["timestamp"].values
    dt_us = np.diff(ts)  # microsecond differences

    time_s = grp["time_s"].values
    elapsed_ts = (ts - ts[0]) / 1e6  # convert µs to seconds
    elapsed_time_s = time_s - time_s[0]

    row = {
        "log_id": lid,
        "n_rows": len(grp),
        "timestamp_min": ts[0],
        "timestamp_max": ts[-1],
        "time_s_min": time_s[0],
        "time_s_max": time_s[-1],
        "time_span_s": (ts[-1] - ts[0]) / 1e6,
        "is_monotonic": bool(np.all(dt_us > 0)),
        "n_repeated_ts": int(np.sum(dt_us == 0)),
        "n_decreasing_ts": int(np.sum(dt_us < 0)),
        "dt_min_us": int(dt_us.min()) if len(dt_us) > 0 else None,
        "dt_median_us": int(np.median(dt_us)) if len(dt_us) > 0 else None,
        "dt_p90_us": int(np.percentile(dt_us, 90)) if len(dt_us) > 0 else None,
        "dt_p99_us": int(np.percentile(dt_us, 99)) if len(dt_us) > 0 else None,
        "dt_max_us": int(dt_us.max()) if len(dt_us) > 0 else None,
        "n_gaps_over_10ms": int(np.sum(dt_us > 10_000)),
        "max_elapsed_diff": float(np.max(np.abs(elapsed_ts - elapsed_time_s))),
    }
    log_summary_rows.append(row)

log_summary = pd.DataFrame(log_summary_rows)
print(log_summary.to_string(index=False))

# %%
log_summary.to_csv(TABLES_DIR / "log_summary.csv", index=False)
print(f"Log summary saved to {TABLES_DIR / 'log_summary.csv'}")

# %% [markdown]
# ### 4.1 Timestamp vs time_s consistency
#
# Both `timestamp` (integer, likely microseconds) and `time_s` (float, seconds)
# exist. We test whether `time_s` equals `(timestamp - timestamp_0) / 1e6`
# within each log.

# %%
for lid in logs:
    grp = df[df["log_id"] == lid]
    ts = grp["timestamp"].values.astype(np.float64)
    time_s = grp["time_s"].values
    elapsed_from_ts = (ts - ts[0]) / 1e6
    diff = np.abs(elapsed_from_ts - (time_s - time_s[0]))
    print(f"{lid}: max |elapsed_ts - elapsed_time_s| = {diff.max():.6e} s, "
          f"mean = {diff.mean():.6e} s")

# %% [markdown]
# ### 4.2 Sampling interval distributions
#
# The CSV row cadence (time between successive rows) is distinct from the
# native sensor sampling rate. The sensor may sample at a higher rate and be
# downsampled or merged during extraction. The nominal ~5 ms interval
# corresponds to ~200 Hz row cadence.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()
for i, lid in enumerate(logs):
    grp = df[df["log_id"] == lid]
    ts = grp["timestamp"].values
    dt_ms = np.diff(ts) / 1e3  # microseconds to milliseconds
    ax = axes[i]
    ax.hist(dt_ms, bins=100, edgecolor="k", alpha=0.7)
    ax.set_title(f"{lid}", fontsize=10)
    ax.set_xlabel("Interval (ms)")
    ax.set_ylabel("Count")
    ax.axvline(5.0, color="red", linestyle="--", alpha=0.7, label="5 ms")
    ax.legend(fontsize=8)
fig.suptitle("Sampling Interval Distributions per Log", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "sampling_intervals.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Locations of large gaps (> 10 ms)
for lid in logs:
    grp = df[df["log_id"] == lid].reset_index(drop=True)
    ts = grp["timestamp"].values
    dt_us = np.diff(ts)
    gap_idx = np.where(dt_us > 10_000)[0]
    if len(gap_idx) > 0:
        print(f"\n{lid}: {len(gap_idx)} gaps > 10 ms")
        for idx in gap_idx[:10]:  # show first 10
            print(f"  Row {idx}→{idx+1}: dt = {dt_us[idx]/1e3:.2f} ms, "
                  f"time_s = {grp.at[idx, 'time_s']:.4f}")
    else:
        print(f"{lid}: No gaps > 10 ms")

# %% [markdown]
# ---
# ## 5. IMU Channel Profiling

# %%
accel_cols = ["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]
gyro_cols = ["gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]
imu_cols = accel_cols + gyro_cols

imu_stats = df[imu_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
print(imu_stats.to_string())

# %%
imu_stats.to_csv(TABLES_DIR / "imu_statistics.csv")
print(f"IMU statistics saved to {TABLES_DIR / 'imu_statistics.csv'}")

# %% [markdown]
# ### 5.1 Accelerometer magnitude verification
#
# The column `accel_magnitude_m_s2` should equal
# `sqrt(accel_x² + accel_y² + accel_z²)`.

# %%
computed_accel_mag = np.sqrt(
    df["accel_x_m_s2"]**2 + df["accel_y_m_s2"]**2 + df["accel_z_m_s2"]**2
)
accel_mag_diff = np.abs(df["accel_magnitude_m_s2"] - computed_accel_mag)
print(f"Accelerometer magnitude verification:")
print(f"  Max |stored - computed|  = {accel_mag_diff.max():.2e}")
print(f"  Mean |stored - computed| = {accel_mag_diff.mean():.2e}")
print(f"  Match within 1e-6: {(accel_mag_diff < 1e-6).all()}")

# %% [markdown]
# ### 5.2 Gyroscope magnitude verification
#
# The column `gyro_magnitude_rad_s` should equal
# `sqrt(gyro_x² + gyro_y² + gyro_z²)`.

# %%
computed_gyro_mag = np.sqrt(
    df["gyro_x_rad_s"]**2 + df["gyro_y_rad_s"]**2 + df["gyro_z_rad_s"]**2
)
gyro_mag_diff = np.abs(df["gyro_magnitude_rad_s"] - computed_gyro_mag)
print(f"Gyroscope magnitude verification:")
print(f"  Max |stored - computed|  = {gyro_mag_diff.max():.2e}")
print(f"  Mean |stored - computed| = {gyro_mag_diff.mean():.2e}")
print(f"  Match within 1e-6: {(gyro_mag_diff < 1e-6).all()}")

# %% [markdown]
# ### 5.3 Physical interpretation notes
#
# - **Units**: Column names suggest `m/s²` for accelerometers and `rad/s` for
#   gyroscopes. These are consistent with PX4 `sensor_combined` topic units.
# - **Coordinate frame**: Unknown. PX4 typically uses FRD (front-right-down)
#   body frame for IMU data, but this cannot be confirmed from the CSV alone.
# - **Gravity**: At rest, we would expect `accel_z ≈ -9.81 m/s²` in FRD.
#   The mean of `accel_z_m_s2` is approximately -9.78, consistent with a
#   body-frame z-axis reading including gravity. This suggests raw
#   (not gravity-compensated) accelerometer data.
# - **Calibration**: No information about calibration status is available.

# %% [markdown]
# ---
# ## 6. GPS, PX4 Estimator, and Reference Fields

# %%
gps_vel_cols = ["gps_vel_n_m_s", "gps_vel_e_m_s", "gps_vel_d_m_s"]
gps_quality_cols = ["vel_ned_valid", "fix_type", "gps_speed_m_s",
                    "gps_eph_m", "gps_epv_m", "gps_satellites_used"]
px4_vel_cols = ["px4_vx_m_s", "px4_vy_m_s", "px4_vz_m_s"]
px4_validity_cols = ["v_xy_valid", "v_z_valid", "xy_valid", "z_valid"]
dr_cols = ["dead_reckoning"]

print("=== GPS Velocity Statistics ===")
print(df[gps_vel_cols].describe().T.to_string())

# %%
print("\n=== GPS Quality Fields ===")
print(df[gps_quality_cols].describe().T.to_string())

# %%
print("\n=== Unique values in GPS quality / validity fields ===")
for c in gps_quality_cols + ["vel_ned_valid"]:
    print(f"  {c}: {sorted(df[c].unique())}")

# %%
print("\n=== PX4 Velocity Statistics ===")
print(df[px4_vel_cols].describe().T.to_string())

# %%
print("\n=== PX4 Validity Flags ===")
for c in px4_validity_cols + dr_cols:
    print(f"  {c}: unique = {sorted(df[c].unique())}")

# %% [markdown]
# ### 6.1 Absent columns
#
# The following categories are **not present** in this dataset:
# - GPS position (lat, lon, alt).
# - PX4 position estimates.
# - Attitude (quaternion / Euler angles).
# - Magnetometer readings.
# - Barometer readings.

# %% [markdown]
# ### 6.2 Repeated consecutive values in GPS and PX4 velocity
#
# GPS receivers typically update at a lower rate than the IMU (e.g., 5–10 Hz
# vs 200 Hz). Between updates, the same GPS velocity may be repeated in merged
# rows. Similarly, the PX4 estimator may update at a different cadence.
#
# We count runs of consecutive identical values. **Note**: Observing value
# changes does not confirm a new sensor measurement arrived at that exact row;
# it only indicates the merged value changed.

# %%
for col_group, label in [(gps_vel_cols, "GPS velocity"), (px4_vel_cols, "PX4 velocity")]:
    print(f"\n=== Repeated consecutive values: {label} ===")
    for lid in logs:
        grp = df[df["log_id"] == lid]
        changes = {}
        for c in col_group:
            vals = grp[c].values
            n_changes = np.sum(vals[1:] != vals[:-1])
            changes[c] = n_changes
        total_rows = len(grp) - 1
        first_col = col_group[0]
        print(f"  {lid}: {changes[first_col]} value changes / {total_rows} intervals "
              f"({changes[first_col]/total_rows*100:.1f}%) for {first_col}")

# %% [markdown]
# ---
# ## 7. Target Definition Verification

# %%
target_cols = ["target_delta_v_n_m_s", "target_delta_v_e_m_s", "target_delta_v_d_m_s"]

# Verify: target = px4_velocity - gps_velocity
computed_n = df["px4_vx_m_s"] - df["gps_vel_n_m_s"]
computed_e = df["px4_vy_m_s"] - df["gps_vel_e_m_s"]
computed_d = df["px4_vz_m_s"] - df["gps_vel_d_m_s"]

diff_n = np.abs(df["target_delta_v_n_m_s"] - computed_n)
diff_e = np.abs(df["target_delta_v_e_m_s"] - computed_e)
diff_d = np.abs(df["target_delta_v_d_m_s"] - computed_d)

target_verif = pd.DataFrame({
    "component": ["north (n)", "east (e)", "down (d)"],
    "max_abs_diff": [diff_n.max(), diff_e.max(), diff_d.max()],
    "mean_abs_diff": [diff_n.mean(), diff_e.mean(), diff_d.mean()],
    "match_within_1e-10": [
        (diff_n < 1e-10).all(),
        (diff_e < 1e-10).all(),
        (diff_d < 1e-10).all(),
    ],
})
print("Target verification: target_delta_v = px4_v - gps_vel")
print(target_verif.to_string(index=False))

# %%
target_verif.to_csv(TABLES_DIR / "target_verification.csv", index=False)
print(f"Target verification saved to {TABLES_DIR / 'target_verification.csv'}")

# %%
print("\n=== Target statistics ===")
print(df[target_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.to_string())

# %% [markdown]
# ---
# ## 8. Distributions and Potential Outliers

# %%
# Global descriptive statistics for all numeric columns
desc_global = df[numeric_cols].describe(
    percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
).T
desc_global.to_csv(TABLES_DIR / "descriptive_statistics_global.csv")
print(f"Global statistics saved ({len(desc_global)} columns)")

# %%
# Per-log statistics for selected columns
selected_cols = imu_cols + target_cols + ["gps_eph_m", "gps_satellites_used"]
per_log_stats = df.groupby("log_id")[selected_cols].describe(
    percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
)
per_log_stats.to_csv(TABLES_DIR / "descriptive_statistics_per_log.csv")
print(f"Per-log statistics saved ({per_log_stats.shape})")

# %% [markdown]
# ### 8.1 Potential outliers using 1.5×IQR rule
#
# We use the classic 1.5×IQR rule as a diagnostic to flag values that lie
# far from the central distribution. These are flagged as **potential
# outliers requiring investigation**, not as confirmed faulty measurements.
# Extreme IMU readings can be physically real (e.g., during aggressive
# manoeuvres or vibration).

# %%
outlier_summary_rows = []
for c in imu_cols + target_cols:
    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[c] < lower) | (df[c] > upper)
    n_outliers = mask.sum()
    outlier_summary_rows.append({
        "column": c,
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_fence": lower, "upper_fence": upper,
        "n_flagged": n_outliers,
        "pct_flagged": round(n_outliers / len(df) * 100, 2),
    })

outlier_summary = pd.DataFrame(outlier_summary_rows)
print(outlier_summary.to_string(index=False))

# %%
outlier_summary.to_csv(TABLES_DIR / "outlier_flags_iqr.csv", index=False)
print(f"Outlier flags saved to {TABLES_DIR / 'outlier_flags_iqr.csv'}")

# %%
# Notable extremes with log_id and time_s
extreme_cols = ["accel_magnitude_m_s2", "gyro_magnitude_rad_s"] + target_cols
for c in extreme_cols:
    idx_max = df[c].idxmax()
    idx_min = df[c].idxmin()
    print(f"\n{c}:")
    print(f"  MAX = {df.at[idx_max, c]:.4f} at log={df.at[idx_max, 'log_id']}, "
          f"time_s={df.at[idx_max, 'time_s']:.4f}")
    print(f"  MIN = {df.at[idx_min, c]:.4f} at log={df.at[idx_min, 'log_id']}, "
          f"time_s={df.at[idx_min, 'time_s']:.4f}")

# %% [markdown]
# ---
# ## 9. Temporal Visualizations

# %% [markdown]
# ### 9.1 Accelerometer time series

# %%
fig, axes = plt.subplots(len(logs), 1, figsize=(16, 3.5 * len(logs)), sharex=False)
if len(logs) == 1:
    axes = [axes]
for i, lid in enumerate(logs):
    grp = df[df["log_id"] == lid]
    ax = axes[i]
    for c, color in zip(accel_cols, ["#e74c3c", "#27ae60", "#2980b9"]):
        ax.plot(grp["time_s"], grp[c], label=c.replace("accel_", "").replace("_m_s2", ""),
                alpha=0.7, linewidth=0.5, color=color)
    ax.set_title(f"Accelerometer — {lid}", fontsize=10)
    ax.set_ylabel("m/s²")
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time_s")
fig.suptitle("Accelerometer Axes per Log", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "accel_time_series.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 9.2 Gyroscope time series

# %%
fig, axes = plt.subplots(len(logs), 1, figsize=(16, 3.5 * len(logs)), sharex=False)
if len(logs) == 1:
    axes = [axes]
for i, lid in enumerate(logs):
    grp = df[df["log_id"] == lid]
    ax = axes[i]
    for c, color in zip(gyro_cols, ["#e74c3c", "#27ae60", "#2980b9"]):
        ax.plot(grp["time_s"], grp[c], label=c.replace("gyro_", "").replace("_rad_s", ""),
                alpha=0.7, linewidth=0.5, color=color)
    ax.set_title(f"Gyroscope — {lid}", fontsize=10)
    ax.set_ylabel("rad/s")
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time_s")
fig.suptitle("Gyroscope Axes per Log", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "gyro_time_series.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 9.3 GPS vs PX4 velocity

# %%
vel_pairs = [
    ("gps_vel_n_m_s", "px4_vx_m_s", "North / X"),
    ("gps_vel_e_m_s", "px4_vy_m_s", "East / Y"),
    ("gps_vel_d_m_s", "px4_vz_m_s", "Down / Z"),
]
fig, axes = plt.subplots(3, len(logs), figsize=(5 * len(logs), 10), sharex=False)
if len(logs) == 1:
    axes = axes.reshape(-1, 1)
for j, lid in enumerate(logs):
    grp = df[df["log_id"] == lid]
    for i, (gps_c, px4_c, label) in enumerate(vel_pairs):
        ax = axes[i, j]
        ax.plot(grp["time_s"], grp[gps_c], alpha=0.7, linewidth=0.5,
                label="GPS", color="#3498db")
        ax.plot(grp["time_s"], grp[px4_c], alpha=0.7, linewidth=0.5,
                label="PX4", color="#e67e22")
        if i == 0:
            ax.set_title(lid, fontsize=9)
        if j == 0:
            ax.set_ylabel(f"{label} (m/s)")
        ax.legend(fontsize=7, loc="upper right")
axes[-1, 0].set_xlabel("time_s")
fig.suptitle("GPS vs PX4 Velocity Components per Log", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "gps_vs_px4_velocity.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 9.4 Target residuals over time

# %%
fig, axes = plt.subplots(len(logs), 1, figsize=(16, 3.5 * len(logs)), sharex=False)
if len(logs) == 1:
    axes = [axes]
for i, lid in enumerate(logs):
    grp = df[df["log_id"] == lid]
    ax = axes[i]
    for c, color in zip(target_cols, ["#9b59b6", "#1abc9c", "#e74c3c"]):
        ax.plot(grp["time_s"], grp[c], alpha=0.7, linewidth=0.5,
                label=c.replace("target_delta_v_", "Δv_").replace("_m_s", ""),
                color=color)
    ax.set_title(f"Target Residuals — {lid}", fontsize=10)
    ax.set_ylabel("m/s")
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
axes[-1].set_xlabel("time_s")
fig.suptitle("Velocity Residuals (PX4 − GPS) per Log", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "target_residuals_time_series.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 9.5 Sampling intervals over time

# %%
fig, axes = plt.subplots(len(logs), 1, figsize=(16, 2.5 * len(logs)), sharex=False)
if len(logs) == 1:
    axes = [axes]
for i, lid in enumerate(logs):
    grp = df[df["log_id"] == lid]
    ts = grp["timestamp"].values
    dt_ms = np.diff(ts) / 1e3
    ax = axes[i]
    ax.plot(grp["time_s"].values[1:], dt_ms, linewidth=0.3, alpha=0.7, color="#2c3e50")
    ax.axhline(5.0, color="red", linestyle="--", alpha=0.5, label="5 ms")
    ax.set_title(f"Row Intervals — {lid}", fontsize=10)
    ax.set_ylabel("Interval (ms)")
    ax.legend(fontsize=8)
axes[-1].set_xlabel("time_s")
fig.suptitle("Sampling Intervals Over Time", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "sampling_intervals_over_time.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 10. Distribution and Recording Comparisons

# %% [markdown]
# ### 10.1 IMU channel distributions by log

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, c in enumerate(imu_cols):
    ax = axes[i // 3, i % 3]
    for lid in logs:
        grp = df[df["log_id"] == lid]
        ax.hist(grp[c], bins=80, alpha=0.5, label=lid[:6], density=True)
    ax.set_title(c, fontsize=10)
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
fig.suptitle("IMU Channel Distributions by Log", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "imu_distributions_by_log.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 10.2 Target distributions by log

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for i, c in enumerate(target_cols):
    ax = axes[i]
    for lid in logs:
        grp = df[df["log_id"] == lid]
        ax.hist(grp[c], bins=80, alpha=0.5, label=lid[:6], density=True)
    ax.set_title(c.replace("target_delta_v_", "Δv_"), fontsize=10)
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
fig.suptitle("Target Residual Distributions by Log", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "target_distributions_by_log.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 10.3 Box plots comparing logs

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, c in enumerate(target_cols):
    ax = axes[i]
    data_by_log = [df[df["log_id"] == lid][c].values for lid in logs]
    bp = ax.boxplot(data_by_log, labels=[lid[:6] for lid in logs], patch_artist=True)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(c.replace("target_delta_v_", "Δv_"), fontsize=10)
    ax.set_ylabel("m/s")
fig.suptitle("Target Residual Box Plots by Log", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "target_boxplots_by_log.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 10.4 Per-log summary comparison
#
# Differences across logs are described statistically. Without supporting
# metadata (e.g., flight mode, wind conditions, pilot inputs), we cannot
# attribute differences to specific flight conditions such as hovering
# or aggressive manoeuvres.

# %%
per_log_mean = df.groupby("log_id")[imu_cols + target_cols].mean()
print(per_log_mean.T.to_string())
per_log_mean.to_csv(TABLES_DIR / "per_log_means.csv")

