#!/usr/bin/env python3
"""
00_data_preparation.py
======================
Validates dataset loading, split strategy, and NaN accounting.
Run this first to sanity-check everything before model training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import (
    load_data, split_data, prepare_tabular, build_windows,
    ALL_FEATURE_COLS, CNN_FEATURE_COLS, TARGET_COLS,
    TRAIN_LOGS, VAL_LOG, TEST_LOG
)


def main():
    print("=" * 60)
    print("DATA PREPARATION VALIDATION")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data()

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n[Split]")
    train_df, val_df, test_df = split_data(df)

    # ── Tabular split check ───────────────────────────────────────────────────
    print("\n[Tabular data — full 34-feature set]")
    print("  TRAIN:")
    X_tr, y_tr, _ = prepare_tabular(train_df, drop_nan=True)
    print(f"    X_train: {X_tr.shape}, y_train: {y_tr.shape}")

    print("  VAL:")
    X_v, y_v, _ = prepare_tabular(val_df, drop_nan=True)
    print(f"    X_val:   {X_v.shape}, y_val:   {y_v.shape}")

    print("  TEST:")
    X_te, y_te, _ = prepare_tabular(test_df, drop_nan=True)
    print(f"    X_test:  {X_te.shape}, y_test:  {X_te.shape}")

    # ── Target stats ──────────────────────────────────────────────────────────
    print("\n[Target statistics on TEST flight]")
    for i, tgt in enumerate(["Δv_N", "Δv_E", "Δv_D"]):
        vals = y_te[:, i]
        print(f"  {tgt}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")

    # ── Windowed data check ───────────────────────────────────────────────────
    print("\n[Windowed sequences — CNN features, window=50, stride=5]")
    X_cnn_tr, y_cnn_tr, _ = build_windows(train_df, CNN_FEATURE_COLS, window_size=50, stride=5)
    X_cnn_v,  y_cnn_v,  _ = build_windows(val_df,   CNN_FEATURE_COLS, window_size=50, stride=5)
    X_cnn_te, y_cnn_te, _ = build_windows(test_df,  CNN_FEATURE_COLS, window_size=50, stride=5)
    print(f"  CNN train: {X_cnn_tr.shape}, val: {X_cnn_v.shape}, test: {X_cnn_te.shape}")

    print("\n[Windowed sequences — LSTM features, window=100, stride=10]")
    X_lstm_tr, y_lstm_tr, _ = build_windows(train_df, ALL_FEATURE_COLS, window_size=100, stride=10)
    X_lstm_v,  y_lstm_v,  _ = build_windows(val_df,   ALL_FEATURE_COLS, window_size=100, stride=10)
    X_lstm_te, y_lstm_te, _ = build_windows(test_df,  ALL_FEATURE_COLS, window_size=100, stride=10)
    print(f"  LSTM train: {X_lstm_tr.shape}, val: {X_lstm_v.shape}, test: {X_lstm_te.shape}")

    # ── Leakage guard ─────────────────────────────────────────────────────────
    print("\n[Leakage guard]")
    gps_cols = [c for c in df.columns if "gps" in c.lower() and c not in ["gps_satellites_used"]]
    ekf_cols = [c for c in df.columns if c.startswith("px4_v")]
    for col in ALL_FEATURE_COLS:
        assert col not in gps_cols, f"GPS column {col} in feature set!"
        assert col not in ekf_cols, f"EKF column {col} in feature set!"
    for col in TARGET_COLS:
        assert col not in ALL_FEATURE_COLS, f"Target {col} in feature set!"
    print("  ✓ No GPS/EKF/target leakage in feature set")

    print("\n✓ Data preparation validation complete.\n")


if __name__ == "__main__":
    main()
