"""Aligned offline evaluation; selection decisions must already be frozen."""
import hashlib
import json
import time

import numpy as np
import pandas as pd

from src.dl.baselines import historical_predictions, predict_baseline
from src.dl.data import ROOT
from src.dl.train import load_neural, predict, save_json
from src.utils import compute_metrics, TARGET_COLUMNS, IMU_FEATURES


def paired_block_interval(y, prediction, segments, block_length=200, repeats=1000, seed=2026):
    """Conditional paired nonoverlapping-block bootstrap, stratified by segment.

    Positive reduction means model RMSE is below zero correction. Blocks never
    cross gaps. Tail blocks keep their actual sample count. Segments/rows are
    NOT claimed to be independent flights.
    """
    rng = np.random.default_rng(seed)
    total_zero = np.zeros((repeats, 3))
    total_model = np.zeros((repeats, 3))
    counts = np.zeros(repeats)
    for segment in np.unique(segments):
        indices = np.flatnonzero(segments == segment)
        blocks = [indices[i:i + block_length] for i in range(0, len(indices), block_length)]
        zero = np.array([(y[b] ** 2).sum(axis=0) for b in blocks])
        model = np.array([((y[b] - prediction[b]) ** 2).sum(axis=0) for b in blocks])
        lengths = np.array([len(b) for b in blocks])
        choices = rng.integers(0, len(blocks), size=(repeats, len(blocks)))
        total_zero += zero[choices].sum(axis=1)
        total_model += model[choices].sum(axis=1)
        counts += lengths[choices].sum(axis=1)
    reduction = np.sqrt(total_zero / counts[:, None]).mean(axis=1) - np.sqrt(total_model / counts[:, None]).mean(axis=1)
    low, high = np.quantile(reduction, [.025, .975])
    return {"block_length": block_length, "repeats": repeats,
            "rmse_reduction_ci_low": float(low), "rmse_reduction_ci_high": float(high)}


def evaluate_experiment(data, config, artifacts, results):
    from src.dl.experiment import training_fingerprint, environment, utc_now
    decision_path = artifacts / "decision.json"
    if not decision_path.exists():
        raise ValueError("Train and freeze validation decisions before evaluating")
    decision = json.loads(decision_path.read_text())
    identity = decision["identity"]
    if identity["config"] != config or identity["dataset_sha256"] != data.checksum:
        raise ValueError("Frozen config or dataset differs")
    if identity["training_code_sha256"] != training_fingerprint() or identity["packages"] != environment()["packages"]:
        raise ValueError("Training source/environment differs from frozen experiment")
    y = data.parts["test"].y
    p = data.parts["test"]
    predictions = {"zero": np.zeros_like(y), "train_mean": np.tile(data.parts["train"].y.mean(axis=0), (len(y), 1))}
    predictions.update(historical_predictions(data, "test"))
    timing = []
    for family, record in decision["classical"].items():
        started = time.perf_counter()
        predictions[family] = predict_baseline(record["best"]["path"], data.inputs("test", "mlp"))
        timing.append({"model": family, "batch_rows": len(y), "batch_seconds_including_load": time.perf_counter() - started})
    for family, record in decision["neural"].items():
        values = []
        started = time.perf_counter()
        for replica in record["replicas"]:
            model, _ = load_neural(ROOT / replica["path"])
            prediction = predict(model, data.dataset("test", record["best"]["config"]))
            predictions[f"{family}_seed{replica['seed']}"] = prediction
            values.append(prediction)
        predictions[family] = np.mean(values, axis=0)
        timing.append({"model": family + "_ensemble", "batch_rows": len(y),
                       "batch_seconds_including_load": time.perf_counter() - started})
    for name, gain in decision["gains"].items():
        predictions[name + "_gated"] = predictions[name] * gain["gain"]
    selected_name = decision["selected_model"] + "_gated"
    predictions["validation_selected"] = predictions[selected_name].copy()
    zero_rmse = compute_metrics(y, predictions["zero"])["rmse_mean"]
    rows, segment_rows, subset_rows = [], [], []
    masks = {f"history_at_least_{w}": np.arange(len(y)) - p.starts + 1 >= w for w in [100, 200, 400]}
    for name, prediction in predictions.items():
        if prediction.shape != y.shape or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid prediction array: {name}")
        metrics = compute_metrics(y, prediction)
        rows.append({"model": name, "rows": len(y), **metrics,
                     "improvement_pct_vs_zero": 100 * (zero_rmse - metrics["rmse_mean"]) / zero_rmse})
        for segment in p.frame.segment_id.unique():
            mask = p.frame.segment_id.to_numpy() == segment
            segment_rows.append({"model": name, "segment_id": int(segment), "rows": int(mask.sum()),
                                 **compute_metrics(y[mask], prediction[mask])})
        for subset, mask in masks.items():
            if mask.any():
                subset_rows.append({"model": name, "subset": subset, "rows": int(mask.sum()),
                                    **compute_metrics(y[mask], prediction[mask])})
    table = pd.DataFrame(rows)
    table.to_csv(results / "test_metrics.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(results / "segment_metrics.csv", index=False)
    pd.DataFrame(subset_rows).to_csv(results / "common_history_metrics.csv", index=False)
    pd.DataFrame(timing).to_csv(results / "batch_timing.csv", index=False)
    np.savez_compressed(results / "test_predictions.npz", **predictions)
    p.frame[["log_id", "timestamp", "time_s", "segment_id"] + TARGET_COLUMNS].to_csv(results / "test_rows.csv", index=False)
    corrected = p.frame[["log_id", "timestamp", "time_s", "segment_id"]].copy()
    for i, (axis, px4, gps) in enumerate(zip("NED", ["px4_vx_m_s", "px4_vy_m_s", "px4_vz_m_s"],
                                            ["gps_vel_n_m_s", "gps_vel_e_m_s", "gps_vel_d_m_s"])):
        corrected[f"px4_{axis}"] = p.frame[px4]
        corrected[f"gps_{axis}"] = p.frame[gps]
        corrected[f"predicted_residual_{axis}"] = predictions["validation_selected"][:, i]
        corrected[f"corrected_{axis}"] = p.frame[px4] - predictions["validation_selected"][:, i]
        np.testing.assert_allclose(corrected[f"corrected_{axis}"] - p.frame[gps],
                                   y[:, i] - predictions["validation_selected"][:, i], atol=1e-12)
    corrected.to_csv(results / "selected_correction.csv", index=False)
    intervals = []
    for name in ["ridge", "rf", "hgb", "mlp", "raw_mlp", "tcn", "gru", "validation_selected"]:
        for length in [100, 200, 400]:
            intervals.append({"model": name, **paired_block_interval(y, predictions[name],
                p.frame.segment_id.to_numpy(), length)})
    pd.DataFrame(intervals).to_csv(results / "conditional_block_intervals.csv", index=False)
    save_json(results / "evaluation_manifest.json", {
        "evaluated_at_utc": utc_now(), "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
        "dataset_sha256": data.checksum, "selected_model": decision["selected_model"],
        "selected_gain": decision["selected_gain"], "evaluation_rows": len(y),
        "scope": "historical held-out recording; no independent-flight inference"})
    print(table[~table.model.str.contains("_seed|_gated")][["model", "rmse_mean", "improvement_pct_vs_zero"]].to_string(index=False), flush=True)


def profile_streaming(data, decision, results, rows=160):
    """Sequential preprocessing + single-model inference, on validation IMU only."""
    from src.dl.inference import ResidualPredictor
    records = []
    frame = data.parts["val"].frame.iloc[:rows + 20]
    for family, record in decision["neural"].items():
        predictor = ResidualPredictor(ROOT / record["replicas"][0]["path"])
        durations, outputs = [], []
        for index, (_, row) in enumerate(frame.iterrows()):
            imu = {name: row[name] for name in IMU_FEATURES}
            started = time.perf_counter()
            outputs.append(predictor.update(int(row.timestamp), imu))
            elapsed = (time.perf_counter() - started) * 1000
            if index >= 20:
                durations.append(elapsed)
        model, _ = load_neural(ROOT / record["replicas"][0]["path"])
        batch = predict(model, data.dataset("val", record["best"]["config"]))[:len(frame)]
        max_error = float(np.max(np.abs(np.asarray(outputs) - batch)))
        np.testing.assert_allclose(outputs, batch, atol=2e-5, rtol=1e-4)
        records.append({"model": family, "rows": len(durations), "median_ms": float(np.median(durations)),
                        "p95_ms": float(np.quantile(durations, .95)), "p99_ms": float(np.quantile(durations, .99)),
                        "max_batch_stream_difference": max_error,
                        "scope": "one model, local CPU, preprocessing included; not onboard or ensemble timing"})
    pd.DataFrame(records).to_csv(results / "streaming_timing.csv", index=False)
