"""
Unit tests for feature engineering pipeline and data integrity using standard library unittest.

Tests cover:
  - Raw dataset integrity
  - Leakage column exclusion
  - Feature count consistency
  - Log-based split disjointness
  - No NaN/Inf after engineering
  - Gap detection and segment handling
  - Jerk computed as da/dt (not second difference)
  - Causal lag features (no future leakage)
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_dataset, log_based_split, IMU_FEATURES, LEAKY_COLUMNS,
    TARGET_COLUMNS, TRAIN_LOGS, VAL_LOGS, TEST_LOGS,
)
from src.feature_engineering import (
    engineer_features, get_feature_columns, GAP_THRESHOLD_US,
    add_jerk_features, add_lag_features, add_difference_features,
)


class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_dataset = load_dataset()

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

    # ── Gap detection tests ──────────────────────────────────────────

    def test_gap_detection_creates_segments(self):
        """Verify that gaps > GAP_THRESHOLD_US create new segments."""
        # Create a synthetic mini-log with a clear gap
        t_base = 1_000_000  # 1s in µs
        dt_normal = 5_000   # 5 ms
        gap = GAP_THRESHOLD_US * 2  # double the threshold

        timestamps = []
        # Segment 0: 10 samples
        for i in range(10):
            timestamps.append(t_base + i * dt_normal)
        # Gap, then Segment 1: 10 more samples
        t_after_gap = timestamps[-1] + gap
        for i in range(10):
            timestamps.append(t_after_gap + i * dt_normal)

        df = pd.DataFrame({
            "log_id": "test_log",
            "timestamp": timestamps,
            "time_s": [(t - t_base) / 1e6 for t in timestamps],
            "accel_x_m_s2": np.random.randn(20),
            "accel_y_m_s2": np.random.randn(20),
            "accel_z_m_s2": np.random.randn(20),
            "gyro_x_rad_s": np.random.randn(20),
            "gyro_y_rad_s": np.random.randn(20),
            "gyro_z_rad_s": np.random.randn(20),
            "accel_magnitude_m_s2": np.abs(np.random.randn(20)),
            "gyro_magnitude_rad_s": np.abs(np.random.randn(20)),
            "target_delta_v_n_m_s": np.zeros(20),
            "target_delta_v_e_m_s": np.zeros(20),
            "target_delta_v_d_m_s": np.zeros(20),
        })

        engineered = engineer_features(df, verbose=False)
        # The diff1 feature at the boundary (row 10) should be 0,
        # not the difference across the gap
        diff_col = "accel_x_m_s2_diff1"
        self.assertIn(diff_col, engineered.columns)
        # Row 10 is the first row of segment 1 — its diff should be 0
        self.assertAlmostEqual(engineered[diff_col].iloc[10], 0.0,
                               msg="Diff feature should be reset at segment boundary")

    def test_jerk_uses_actual_dt(self):
        """Verify jerk = da/dt uses real elapsed time, not just diff(diff)."""
        # Create 5 samples with non-uniform dt
        timestamps = [0, 5000, 10000, 20000, 25000]  # µs
        accel = [0.0, 1.0, 2.0, 3.0, 4.0]

        segment = pd.DataFrame({
            "timestamp": timestamps,
            "accel_x_m_s2": accel,
            "accel_y_m_s2": [0.0] * 5,
            "accel_z_m_s2": [0.0] * 5,
        })

        result = add_jerk_features(segment)

        # Row 0: no prior → 0
        self.assertAlmostEqual(result["accel_x_m_s2_jerk"].iloc[0], 0.0)

        # Row 1: da=1.0, dt=0.005s → jerk=200 m/s³
        self.assertAlmostEqual(result["accel_x_m_s2_jerk"].iloc[1], 200.0, places=1)

        # Row 3: da=1.0, dt=0.010s → jerk=100 m/s³
        # (non-uniform dt means jerk differs even though da is the same)
        self.assertAlmostEqual(result["accel_x_m_s2_jerk"].iloc[3], 100.0, places=1)

    def test_lag_features_are_causal(self):
        """Verify lag features never use future values."""
        segment = pd.DataFrame({
            "accel_x_m_s2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        result = add_lag_features(segment, ["accel_x_m_s2"], [2])

        # Row 0 and 1: lag=2 has no past — should be filled causally
        # (ffill then fillna with current value)
        lag_col = "accel_x_m_s2_lag2"
        # Row 0: no past at all — should equal current value (10.0)
        self.assertAlmostEqual(result[lag_col].iloc[0], 10.0)
        # Row 2: lag=2 means value from row 0 = 10.0
        self.assertAlmostEqual(result[lag_col].iloc[2], 10.0)
        # Row 4: lag=2 means value from row 2 = 30.0
        self.assertAlmostEqual(result[lag_col].iloc[4], 30.0)

    def test_segment_metadata_excluded_from_features(self):
        """Verify _segment column is not in the final feature set."""
        sample_df = self.raw_dataset.head(500).copy()
        engineered = engineer_features(sample_df, verbose=False)
        feature_cols = get_feature_columns(engineered)
        self.assertNotIn("_segment", feature_cols)
        self.assertNotIn("_segment", engineered.columns)


if __name__ == "__main__":
    unittest.main()
