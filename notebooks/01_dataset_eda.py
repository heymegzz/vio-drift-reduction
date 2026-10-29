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
