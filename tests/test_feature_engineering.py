"""
Unit tests for feature engineering pipeline and data integrity using standard library unittest.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import (
    load_dataset, log_based_split, IMU_FEATURES, LEAKY_COLUMNS,
    TARGET_COLUMNS, TRAIN_LOGS, VAL_LOGS, TEST_LOGS,
)
from feature_engineering import engineer_features, get_feature_columns


class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_dataset = load_dataset("imu_drift_feature_dataset.csv")

    def test_dataset_completeness(self):
        """Verify that the raw dataset has expected columns and no NaNs."""
        self.assertEqual(len(self.raw_dataset), 38967)
        self.assertEqual(self.raw_dataset.isna().sum().sum(), 0)
        for target in TARGET_COLUMNS:
            self.assertIn(target, self.raw_dataset.columns)

    def test_leakage_column_quarantine(self):
        """Verify that feature engineering excludes all GPS-derived and leaky columns."""
        sample_df = self.raw_dataset.head(500).copy()
        engineered = engineer_features(sample_df, verbose=False)
        feature_cols = get_feature_columns(engineered)

        for col in LEAKY_COLUMNS:
            self.assertNotIn(col, feature_cols, f"Leakage detected: {col} present in feature columns!")

    def test_engineered_feature_count(self):
        """Verify that feature engineering outputs the expected 142 input features."""
        sample_df = self.raw_dataset.head(500).copy()
        engineered = engineer_features(sample_df, verbose=False)
        feature_cols = get_feature_columns(engineered)
        self.assertEqual(len(feature_cols), 142)

    def test_log_splitting_disjoint(self):
        """Verify that log-based split strictly isolates whole flights without overlap."""
        sample_df = self.raw_dataset.copy()
        df_train, df_val, df_test = log_based_split(sample_df)

        train_logs = set(df_train["log_id"].unique())
        val_logs = set(df_val["log_id"].unique())
        test_logs = set(df_test["log_id"].unique())

        self.assertTrue(train_logs.isdisjoint(val_logs))
        self.assertTrue(train_logs.isdisjoint(test_logs))
        self.assertTrue(val_logs.isdisjoint(test_logs))

    def test_no_nan_in_engineered_features(self):
        """Ensure no NaN or Infinite values exist after temporal feature derivation."""
        sample_df = self.raw_dataset.head(1000).copy()
        engineered = engineer_features(sample_df, verbose=False)
        feature_cols = get_feature_columns(engineered)

        self.assertFalse(engineered[feature_cols].isna().any().any())
        self.assertFalse(np.isinf(engineered[feature_cols].values).any())


if __name__ == "__main__":
    unittest.main()
