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

