"""
Unit tests for feature engineering pipeline and data integrity.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import (
    load_dataset, log_based_split, IMU_FEATURES, LEAKY_COLUMNS,
    TARGET_COLUMNS, TRAIN_LOGS, VAL_LOGS, TEST_LOGS,
)
from feature_engineering import engineer_features, get_feature_columns


@pytest.fixture(scope="session")
def raw_dataset():
    """Load the raw dataset once for the test session."""
    assert os.path.exists("imu_drift_feature_dataset.csv"), "Dataset CSV not found!"
    return load_dataset("imu_drift_feature_dataset.csv")


def test_dataset_completeness(raw_dataset):
    """Verify that the raw dataset has expected columns and no NaNs."""
    assert len(raw_dataset) == 38967
    assert raw_dataset.isna().sum().sum() == 0, "Raw dataset contains unexpected missing values"
    for target in TARGET_COLUMNS:
        assert target in raw_dataset.columns


def test_leakage_column_quarantine(raw_dataset):
    """Verify that feature engineering excludes all GPS-derived and leaky columns."""
    sample_df = raw_dataset.head(500).copy()
    engineered = engineer_features(sample_df, verbose=False)
    feature_cols = get_feature_columns(engineered)

    for col in LEAKY_COLUMNS:
        assert col not in feature_cols, f"Leakage detected: {col} present in feature columns!"


def test_engineered_feature_count(raw_dataset):
    """Verify that feature engineering outputs the expected 142 input features."""
    sample_df = raw_dataset.head(500).copy()
    engineered = engineer_features(sample_df, verbose=False)
    feature_cols = get_feature_columns(engineered)
    assert len(feature_cols) == 142, f"Expected 142 features, got {len(feature_cols)}"


def test_log_splitting_disjoint(raw_dataset):
    """Verify that log-based split strictly isolates whole flights without overlap."""
    sample_df = raw_dataset.copy()
    df_train, df_val, df_test = log_based_split(sample_df)

    train_logs = set(df_train["log_id"].unique())
    val_logs = set(df_val["log_id"].unique())
    test_logs = set(df_test["log_id"].unique())

    assert train_logs.isdisjoint(val_logs), "Train and validation splits share logs"
    assert train_logs.isdisjoint(test_logs), "Train and test splits share logs"
    assert val_logs.isdisjoint(test_logs), "Validation and test splits share logs"


def test_no_nan_in_engineered_features(raw_dataset):
    """Ensure no NaN or Infinite values exist after temporal feature derivation."""
    sample_df = raw_dataset.head(1000).copy()
    engineered = engineer_features(sample_df, verbose=False)
    feature_cols = get_feature_columns(engineered)

    assert not engineered[feature_cols].isna().any().any(), "Engineered features contain NaN values"
    assert not np.isinf(engineered[feature_cols].values).any(), "Engineered features contain infinite values"

