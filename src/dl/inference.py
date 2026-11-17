"""CPU streaming neural inference from current IMU values, without GPS/targets."""
from collections import deque

import numpy as np
import pandas as pd
import torch

from src.dl.data import CausalDataset, Scale
from src.dl.train import load_neural
from src.feature_engineering import engineer_features


class ResidualPredictor:
    def __init__(self, checkpoint):
        self.model, self.checkpoint = load_neural(checkpoint)
        self.config = self.checkpoint["config"]
        self.metadata = self.checkpoint["data"]
        self.raw_scale = Scale.from_dict(self.metadata["raw_scale"])
        self.feature_scale = Scale.from_dict(self.metadata["feature_scale"])
        self.dt_scale = Scale.from_dict(self.metadata["dt_scale"])
        self.buffer = deque(maxlen=max(100, self.config.get("window", 1)))
        self.last_timestamp = None

    def reset(self):
        self.buffer.clear()
        self.last_timestamp = None

    def update(self, timestamp, imu):
        """Return N/E/D predicted residual. Caller subtracts it from PX4 velocity.

        timestamp is integer microseconds; imu is a dict of six axes and two
        magnitudes in the checkpoint schema. Explicit reset is required at a
        new recording, even if its timestamp happens to increase.
        """
        if self.last_timestamp is not None:
            dt = int(timestamp) - self.last_timestamp
            if dt <= 0:
                raise ValueError("Non-increasing streaming timestamp")
            if dt > self.metadata["gap_threshold_us"]:
                self.reset()
        row = {name: float(imu[name]) for name in self.metadata["raw_names"]}
        if not np.isfinite(list(row.values())).all():
            raise ValueError("Nonfinite IMU input")
        row.update(timestamp=int(timestamp), log_id="stream", time_s=0.0)
        self.buffer.append(row)
        self.last_timestamp = int(timestamp)
        frame = pd.DataFrame(self.buffer)
        sequence = self.config["model"] in {"tcn", "gru"}
        if self.config["model"] == "mlp":
            featured = engineer_features(frame.copy(), verbose=False)
            x = self.feature_scale.transform(featured[self.metadata["feature_names"]].to_numpy())
        else:
            x = self.raw_scale.transform(frame[self.metadata["raw_names"]].to_numpy())
        if self.config.get("use_dt", False):
            dt = frame.timestamp.diff().fillna(0).to_numpy()[:, None] / 1e6
            x = np.column_stack([x, self.dt_scale.transform(dt)])
        dataset = CausalDataset(x, np.zeros((len(x), 3)), np.zeros(len(x)),
                                self.config.get("window", 1), [len(x) - 1], sequence)
        with torch.inference_mode():
            prediction = self.model(dataset[0][0].unsqueeze(0)).numpy()[0]
        return prediction
