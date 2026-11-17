"""Validation-only input sensitivity and complete local inference profiling."""
import argparse
import json

import numpy as np
import pandas as pd

from src.dl.data import ROOT, PreparedData
from src.dl.evaluate import profile_streaming
from src.dl.train import load_neural, predict, seed_everything
from src.utils import compute_metrics, IMU_FEATURES


def explain_validation(data, decision, results):
    rows = []
    for family, selection in decision["neural"].items():
        config = selection["best"]["config"]
        model, _ = load_neural(ROOT / selection["replicas"][0]["path"])
        dataset = data.dataset("val", config)
        y = data.parts["val"].y
        original = compute_metrics(y, predict(model, dataset))["rmse_mean"]
        names = data.feature_names if family == "mlp" else IMU_FEATURES.copy()
        if config.get("use_dt"):
            names = names + ["dt"]
        groups = {
            "accelerometer": [i for i, name in enumerate(names) if name.startswith("accel")],
            "gyroscope": [i for i, name in enumerate(names) if name.startswith("gyro")],
        }
        if config.get("use_dt"):
            groups["elapsed_time"] = [len(names) - 1]
        original_x = dataset.x.copy()
        for group, indices in groups.items():
            dataset.x = original_x.copy()
            dataset.x[:, indices] = 0.0  # Training mean in normalized coordinates.
            score = compute_metrics(y, predict(model, dataset))["rmse_mean"]
            rows.append({"model": family, "group": group, "validation_rmse": original,
                         "ablated_validation_rmse": score, "delta_rmse": score - original,
                         "replicate_seed": selection["replicas"][0]["seed"],
                         "interpretation": "training-mean replacement sensitivity, not causal importance"})
    pd.DataFrame(rows).to_csv(results / "input_sensitivity.csv", index=False)
    profile_streaming(data, decision, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="phase2_v2")
    args = parser.parse_args()
    seed_everything(11)
    decision = json.loads((ROOT / "saved_models/phase2" / args.run / "decision.json").read_text())
    data = PreparedData()
    if data.checksum != decision["identity"]["dataset_sha256"]:
        raise ValueError("Dataset differs from the frozen experiment")
    explain_validation(data, decision, ROOT / "results/phase2" / args.run)


if __name__ == "__main__":
    main()
