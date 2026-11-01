"""
Standard library unittest suite for CI and local verification without external dependencies.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import (
    load_dataset, log_based_split, LEAKY_COLUMNS,
    TARGET_COLUMNS, compute_metrics,
)
from feature_engineering import engineer_features, get_feature_columns


class TestVioDriftPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_df = load_dataset("imu_drift_feature_dataset.csv")

    def test_01_dataset_shape(self):
        self.assertEqual(len(self.raw_df), 38967)
        self.assertEqual(self.raw_df.isna().sum().sum(), 0)

    def test_02_leakage_safeguards(self):
        sample = self.raw_df.head(500).copy()
        engineered = engineer_features(sample, verbose=False)
        features = get_feature_columns(engineered)

        for col in LEAKY_COLUMNS:
            self.assertNotIn(col, features, f"Leaky column {col} found in features")

    def test_03_feature_count(self):
        sample = self.raw_df.head(500).copy()
        engineered = engineer_features(sample, verbose=False)
        features = get_feature_columns(engineered)
        self.assertEqual(len(features), 142)

    def test_04_log_splitting_integrity(self):
        df_train, df_val, df_test = log_based_split(self.raw_df)
        train_logs = set(df_train["log_id"].unique())
        val_logs = set(df_val["log_id"].unique())
        test_logs = set(df_test["log_id"].unique())

        self.assertTrue(train_logs.isdisjoint(val_logs))
        self.assertTrue(train_logs.isdisjoint(test_logs))
        self.assertTrue(val_logs.isdisjoint(test_logs))

    def test_05_metric_correctness(self):
        y_true = np.array([[0.5, -0.2, 0.1], [0.0, 0.4, -0.3]])
        y_pred = np.array([[0.5, -0.2, 0.1], [0.0, 0.4, -0.3]])
        metrics = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["rmse_mean"], 0.0)
        self.assertAlmostEqual(metrics["r2_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()

