"""
Data Preparation — Log-Based Train / Validation / Test Splitting
=================================================================

This script:
  1. Loads the raw dataset
  2. Runs the feature engineering pipeline
  3. Splits by log_id (temporal split — prevents autocorrelation leakage)
  4. Verifies no data leakage between splits
  5. Saves split statistics

Split rationale (documented in docs/eda_report.md §6.4 and §7):
  - Target residuals exhibit strong autocorrelation (hundreds of lags)
  - Random row-wise splits would leak temporal information
  - Entire logs are assigned to train/val/test to ensure clean separation
  - Train: log_51 + log_53 (21,004 rows — largest combined)
  - Val:   log_52           (8,657 rows — tuning & early stopping)
  - Test:  log_54           (8,306 rows — final unseen evaluation)

Usage:
    python models/00_data_preparation.py
"""

import sys
import os
import numpy as np
import pandas as pd

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_dataset, log_based_split, TARGET_COLUMNS,
    TRAIN_LOGS, VAL_LOGS, TEST_LOGS, ensure_dirs,
)
from src.feature_engineering import engineer_features, get_feature_columns


def verify_no_leakage(df_train, df_val, df_test):
    """Verify that no log_id appears in more than one split."""
    train_logs = set(df_train["log_id"].unique())
    val_logs = set(df_val["log_id"].unique())
    test_logs = set(df_test["log_id"].unique())

    assert train_logs.isdisjoint(val_logs), \
        f"Leakage: train & val share logs: {train_logs & val_logs}"
    assert train_logs.isdisjoint(test_logs), \
        f"Leakage: train & test share logs: {train_logs & test_logs}"
    assert val_logs.isdisjoint(test_logs), \
        f"Leakage: val & test share logs: {val_logs & test_logs}"

    print("✓ No log-level leakage detected between splits")


def verify_temporal_separation(df_train, df_val, df_test):
    """Print timestamp ranges per split for manual verification."""
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        for log_id in sorted(df["log_id"].unique()):
            subset = df[df["log_id"] == log_id]
            print(f"  {name:>5} | {log_id} | "
                  f"t=[{subset['time_s'].min():.2f}, "
                  f"{subset['time_s'].max():.2f}] s | "
                  f"{len(subset):,} rows")


def main():
    ensure_dirs()

    # 1. Load raw data
    print("=" * 60)
    print("  Data Preparation Pipeline")
    print("=" * 60)
    print("\n1. Loading raw dataset...")
    df = load_dataset()
    print(f"   Loaded {len(df):,} rows × {len(df.columns)} columns")

    # 2. Feature engineering
    print("\n2. Running feature engineering pipeline...")
    df = engineer_features(df)

    # 3. Log-based split
    print("\n3. Applying log-based split...")
    df_train, df_val, df_test = log_based_split(df)

    # 4. Verify no leakage
    print("\n4. Verifying split integrity...")
    verify_no_leakage(df_train, df_val, df_test)
    verify_temporal_separation(df_train, df_val, df_test)

    # 5. Feature summary
    feature_cols = get_feature_columns(df)
    print(f"\n5. Feature summary:")
    print(f"   Input features:  {len(feature_cols)}")
    print(f"   Target columns:  {len(TARGET_COLUMNS)}")
    print(f"   Split ratios:    "
          f"train={len(df_train)/len(df):.1%}, "
          f"val={len(df_val)/len(df):.1%}, "
          f"test={len(df_test)/len(df):.1%}")

    # 6. Save split info
    split_info = pd.DataFrame({
        "split": ["train", "val", "test"],
        "logs": [", ".join(TRAIN_LOGS), ", ".join(VAL_LOGS),
                 ", ".join(TEST_LOGS)],
        "n_rows": [len(df_train), len(df_val), len(df_test)],
        "n_features": [len(feature_cols)] * 3,
    })
    split_info.to_csv("results/models/split_info.csv", index=False)
    print(f"\n   Saved split info to results/models/split_info.csv")

    # 7. Target statistics per split
    print("\n6. Target statistics per split:")
    for name, df_split in [("Train", df_train), ("Val", df_val),
                           ("Test", df_test)]:
        stats = df_split[TARGET_COLUMNS].describe().loc[["mean", "std"]]
        print(f"\n   {name}:")
        print(stats.to_string(float_format="{:.4f}".format).replace(
            "\n", "\n   "
        ))

    print(f"\n{'='*60}")
    print("  Data preparation complete. Ready for modelling.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

