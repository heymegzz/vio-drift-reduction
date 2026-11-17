import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dl.data import PreparedData
from src.dl.evaluate import paired_block_interval
from src.dl.inference import ResidualPredictor
from src.dl.models import build_model
from src.dl.train import predict, seed_everything
from src.utils import IMU_FEATURES, compute_metrics


class TestEvaluation(unittest.TestCase):
    def test_correction_identity_and_sign(self):
        rng = np.random.default_rng(4)
        gps, px4, prediction = [rng.normal(size=(30, 3)) for _ in range(3)]
        y = px4 - gps
        np.testing.assert_allclose((px4 - prediction) - gps, y - prediction)
        self.assertAlmostEqual(compute_metrics(y, prediction)["rmse_mean"],
                               compute_metrics(gps, px4 - prediction)["rmse_mean"])

    def test_bootstrap_zero_and_perfect_predictor(self):
        y = np.arange(60).reshape(20, 3) / 100
        segments = np.repeat([0, 1], 10)
        zero = paired_block_interval(y, np.zeros_like(y), segments, 4, 100)
        self.assertEqual(zero["rmse_reduction_ci_low"], 0)
        self.assertEqual(zero["rmse_reduction_ci_high"], 0)
        perfect = paired_block_interval(y, y, segments, 4, 100)
        self.assertGreater(perfect["rmse_reduction_ci_low"], 0)


class TestStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            cls.data = PreparedData()

    def test_streaming_matches_offline_and_resets(self):
        seed_everything(3)
        data = self.data
        frame = data.parts["val"].frame.iloc[:125]
        for family in ["mlp", "raw_mlp", "tcn", "gru"]:
            config = {"model": family, "window": 100, "hidden": 8, "dropout": 0.0,
                      "use_dt": family in {"tcn", "gru"}}
            ds = data.dataset("val", config)
            inputs = ds[0][0].shape[-1]
            model = build_model(config, inputs).eval()
            torch.nn.init.normal_(model.head.weight, std=.01)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pt"
                torch.save({"state_dict": model.state_dict(), "config": config,
                            "inputs": inputs, "data": data.metadata()}, path)
                streaming = ResidualPredictor(path)
                observed = []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                    for _, row in frame.iterrows():
                        observed.append(streaming.update(int(row.timestamp), row[IMU_FEATURES].to_dict()))
                # Restrict dataset endpoints for efficient exact comparison.
                ds.endpoints = ds.endpoints[:len(frame)]
                np.testing.assert_allclose(observed, predict(model, ds), atol=2e-5, rtol=1e-4)
                imu = frame.iloc[-1][IMU_FEATURES].to_dict()
                new_timestamp = int(frame.timestamp.iloc[-1]) + 50001
                after_gap = streaming.update(new_timestamp, imu)
                fresh = ResidualPredictor(path).update(new_timestamp, imu)
                np.testing.assert_allclose(after_gap, fresh, atol=1e-6)
                with self.assertRaises(ValueError):
                    streaming.update(new_timestamp, imu)


if __name__ == "__main__":
    unittest.main()
