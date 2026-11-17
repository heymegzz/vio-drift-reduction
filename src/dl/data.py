"""Validated row-aligned data and lazy causal windows; no target feedback."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.feature_engineering import GAP_THRESHOLD_US, engineer_features, get_feature_columns
from src.utils import IMU_FEATURES, TARGET_COLUMNS, TRAIN_LOGS, VAL_LOGS, TEST_LOGS

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {"train": TRAIN_LOGS, "val": VAL_LOGS, "test": TEST_LOGS}


def validate_raw(frame):
    required = ["log_id", "timestamp", "time_s"] + IMU_FEATURES + TARGET_COLUMNS
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or not np.isfinite(frame.select_dtypes(include="number")).all().all():
        raise ValueError("Empty or nonfinite dataset")
    expected = set(TRAIN_LOGS + VAL_LOGS + TEST_LOGS)
    if set(frame.log_id) != expected:
        raise ValueError("Unexpected or missing recording IDs; revise the explicit split manifest")
    for _, group in frame.groupby("log_id", sort=False):
        if not (np.diff(group.timestamp.to_numpy()) > 0).all():
            raise ValueError("Timestamps must be strictly increasing within each recording")
    return frame.sort_values(["log_id", "timestamp"]).reset_index(drop=True)


@dataclass
class Scale:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x):
        std = x.std(axis=0, dtype=np.float64)
        return cls(x.mean(axis=0, dtype=np.float64), np.where(std > 1e-8, std, 1.0))

    def transform(self, x):
        result = ((x - self.mean) / self.std).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite normalized input")
        return result

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, value):
        return cls(np.asarray(value["mean"]), np.asarray(value["std"]))


@dataclass
class Partition:
    frame: pd.DataFrame
    raw: np.ndarray
    engineered: np.ndarray
    y: np.ndarray
    starts: np.ndarray
    dt: np.ndarray

    def endpoints(self, stride=1):
        if stride < 1:
            raise ValueError("stride must be positive")
        indices = np.arange(len(self.y))
        return indices[(indices - self.starts) % stride == 0]


class PreparedData:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "data/imu_drift_feature_dataset.csv")
        self.checksum = sha256(self.path.read_bytes()).hexdigest()
        frame = validate_raw(pd.read_csv(self.path))
        engineered = engineer_features(frame.copy(), verbose=False)
        self.feature_names = get_feature_columns(engineered)
        expected = set(IMU_FEATURES)
        for column in self.feature_names:
            if column not in {"accel_xy_angle", "gyro_xy_angle", "accel_gyro_mag_ratio"} and not any(
                column == base or column.startswith(base + "_") for base in expected
            ):
                raise ValueError(f"Input not in IMU-derived allowlist: {column}")
        if len(self.feature_names) != 142:
            raise ValueError("Historical feature schema changed")
        if not engineered[["log_id", "timestamp"]].equals(frame[["log_id", "timestamp"]]):
            raise ValueError("Feature/label row alignment changed")
        self.parts = {}
        segment_rows = []
        for name, logs in SPLITS.items():
            mask = frame.log_id.isin(logs)
            part = frame.loc[mask].reset_index(drop=True).copy()
            boundary = part.log_id.ne(part.log_id.shift()) | part.timestamp.diff().gt(GAP_THRESHOLD_US)
            starts = np.maximum.accumulate(np.where(boundary, np.arange(len(part)), 0))
            part["segment_id"] = boundary.cumsum().to_numpy() - 1
            dt = part.timestamp.diff().to_numpy(dtype=np.float64) / 1e6
            dt[boundary.to_numpy()] = 0.0
            self.parts[name] = Partition(
                part, part[IMU_FEATURES].to_numpy(dtype=np.float64),
                engineered.loc[mask, self.feature_names].to_numpy(dtype=np.float64),
                part[TARGET_COLUMNS].to_numpy(dtype=np.float64), starts, dt,
            )
            for (log, segment), group in part.groupby(["log_id", "segment_id"]):
                segment_rows.append({"split": name, "log_id": log, "segment_id": int(segment),
                                     "rows": len(group), "duration_s": float((group.timestamp.iloc[-1] - group.timestamp.iloc[0]) / 1e6)})
        self.segments = pd.DataFrame(segment_rows)
        self.raw_scale = Scale.fit(self.parts["train"].raw)
        self.feature_scale = Scale.fit(self.parts["train"].engineered)
        self.dt_scale = Scale.fit(self.parts["train"].dt[:, None])

    def inputs(self, split, kind, use_dt=False):
        p = self.parts[split]
        x = self.feature_scale.transform(p.engineered) if kind == "mlp" else self.raw_scale.transform(p.raw)
        if use_dt:
            x = np.column_stack([x, self.dt_scale.transform(p.dt[:, None])])
        return np.ascontiguousarray(x, dtype=np.float32)

    def dataset(self, split, config, stride=1):
        p = self.parts[split]
        return CausalDataset(self.inputs(split, config["model"], config.get("use_dt", False)),
                             p.y, p.starts, config.get("window", 1), p.endpoints(stride),
                             config["model"] in {"tcn", "gru"})

    def metadata(self):
        return {"dataset_sha256": self.checksum, "feature_names": self.feature_names,
                "raw_names": IMU_FEATURES, "target_names": TARGET_COLUMNS,
                "gap_threshold_us": GAP_THRESHOLD_US, "splits": SPLITS,
                "raw_scale": self.raw_scale.to_dict(), "feature_scale": self.feature_scale.to_dict(),
                "dt_scale": self.dt_scale.to_dict()}


class CausalDataset(Dataset):
    def __init__(self, x, y, starts, window=1, endpoints=None, sequence=True):
        if window < 1 or len(x) != len(y) or len(starts) != len(x):
            raise ValueError("Invalid window or array lengths")
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.window, self.sequence = int(window), sequence
        self.endpoints = np.arange(len(x)) if endpoints is None else np.asarray(endpoints)

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        end = int(self.endpoints[index])
        if not self.sequence:
            return torch.from_numpy(self.x[end]), torch.from_numpy(self.y[end])
        start = max(int(self.starts[end]), end - self.window + 1)
        count = end - start + 1
        window = np.zeros((self.window, self.x.shape[1] + 1), dtype=np.float32)
        window[-count:, :-1] = self.x[start:end + 1]
        window[-count:, -1] = 1.0
        return torch.from_numpy(window), torch.from_numpy(self.y[end])
