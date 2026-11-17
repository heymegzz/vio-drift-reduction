"""CLI: train without test scoring, freeze decisions, then evaluate once."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.dl.baselines import fit_baselines, historical_predictions, predict_baseline
from src.dl.data import PreparedData, ROOT
from src.dl.train import fit_neural, load_neural, predict, save_json, seed_everything
from src.utils import compute_metrics


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ["numpy", "pandas", "scikit-learn", "scipy", "torch", "matplotlib"]},
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "recorded_at_utc": utc_now()}


def training_fingerprint():
    paths = [ROOT / "src" / "dl" / f"{name}.py" for name in ["data", "models", "train", "baselines"]]
    paths += [ROOT / "src" / "feature_engineering.py", ROOT / "src" / "utils.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def train_experiment(data, config, artifacts, results):
    identity = {"config": config, "dataset_sha256": data.checksum, "training_code_sha256": training_fingerprint()}
    identity_path = artifacts / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Experiment identity changed. Use a new config name; do not overwrite an experiment.")
    save_json(identity_path, identity)
    save_json(results / "environment.json", environment())
    save_json(results / "data_contract.json", data.metadata())
    data.segments.to_csv(results / "segments.csv", index=False)
    coverage = []
    for split, part in data.parts.items():
        history = np.arange(len(part.y)) - part.starts + 1
        for window in [1, 100, 200, 400]:
            coverage.append({"split": split, "window": window, "rows": len(part.y),
                             "full_history_rows": int((history >= window).sum())})
    pd.DataFrame(coverage).to_csv(results / "window_coverage.csv", index=False)
    baseline_selection = fit_baselines(data, config["baselines"], artifacts / "classical")
    yv = data.parts["val"].y
    val_predictions = {"zero": np.zeros_like(yv),
                       "train_mean": np.tile(data.parts["train"].y.mean(axis=0), (len(yv), 1))}
    val_predictions.update(historical_predictions(data, "val"))
    for family, record in baseline_selection.items():
        val_predictions[family] = predict_baseline(record["best"]["path"], data.inputs("val", "mlp"))
    neural_selection = {}
    trial_rows = []
    for family, candidates in config["neural"].items():
        trials = []
        for index, candidate in enumerate(candidates):
            directory = artifacts / "search" / family / str(index)
            _, record = fit_neural(data, candidate, config["seeds"][0], directory, config["training"])
            record = dict(record, path=str((directory / "model.pt").relative_to(ROOT)), trial=index)
            trials.append(record)
            trial_rows.append({"family": family, "trial": index, "config": json.dumps(candidate, sort_keys=True),
                               "best_epoch": record["best_epoch"], "training_seconds": record["training_seconds"],
                               "parameters": record["parameters"], **record["validation"]})
            print(f"neural {family} {index + 1}/{len(candidates)} val={record['validation']['rmse_mean']:.6f} epoch={record['best_epoch']}", flush=True)
        best = min(trials, key=lambda r: (r["validation"]["rmse_mean"], r["parameters"]))
        replicas, predictions = [], []
        for seed in config["seeds"]:
            if seed == config["seeds"][0]:
                model, _ = load_neural(ROOT / best["path"])
                record, path = best, best["path"]
            else:
                directory = artifacts / "replicas" / family / str(seed)
                model, record = fit_neural(data, best["config"], seed, directory, config["training"])
                path = str((directory / "model.pt").relative_to(ROOT))
            replicas.append({"seed": seed, "path": path, "best_epoch": record["best_epoch"],
                             "validation": record["validation"]})
            prediction = predict(model, data.dataset("val", best["config"]))
            predictions.append(prediction)
            val_predictions[f"{family}_seed{seed}"] = prediction
        val_predictions[family] = np.mean(predictions, axis=0)
        neural_selection[family] = {"best": best, "replicas": replicas, "trials": trials}
        print(f"ensemble {family} val={compute_metrics(yv, val_predictions[family])['rmse_mean']:.6f}", flush=True)
    pd.DataFrame(trial_rows).to_csv(results / "neural_trials.csv", index=False)
    rows, gains = [], {}
    for name, prediction in val_predictions.items():
        rows.append({"model": name, **compute_metrics(yv, prediction)})
        if "_seed" not in name:
            scored = [(compute_metrics(yv, prediction * gain)["rmse_mean"], gain)
                      for gain in [0., .25, .5, .75, 1.]]
            score, gain = min(scored)
            gains[name] = {"gain": gain, "validation_rmse": float(score)}
    # Prefer literal zero in ties, then smaller model families by fixed name.
    selected = min(gains, key=lambda name: (gains[name]["validation_rmse"], name != "zero", name))
    decision = {"frozen_at_utc": utc_now(), "identity": identity,
                "classical": baseline_selection, "neural": neural_selection,
                "gains": gains, "selected_model": selected,
                "selection_metric": "mean of axis RMSE on log_52",
                "selected_gain": gains[selected]["gain"]}
    save_json(artifacts / "decision.json", decision)
    save_json(results / "decision_summary.json", {k: decision[k] for k in
              ["frozen_at_utc", "gains", "selected_model", "selected_gain", "selection_metric"]})
    pd.DataFrame(rows).to_csv(results / "validation_metrics.csv", index=False)
    np.savez_compressed(results / "validation_predictions.npz", **val_predictions)
    print(f"FROZEN: {selected}, gain={gains[selected]['gain']} (no test scoring used)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dl/benchmark.json")
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="train")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if args.smoke:
        config["name"] += "_smoke"
        config["seeds"] = config["seeds"][:1]
        config["training"].update(epochs=2, patience=2, train_stride=20, validation_stride=20)
        config["baselines"] = {k: v[:1] for k, v in config["baselines"].items()}
        config["neural"] = {k: v[:1] for k, v in config["neural"].items()}
    seed_everything(config["seeds"][0], config["training"]["threads"])
    artifacts = ROOT / "saved_models" / "phase2" / config["name"]
    results = ROOT / "results" / "phase2" / config["name"]
    results.mkdir(parents=True, exist_ok=True)
    data = PreparedData()
    if args.stage in {"train", "all"}:
        if (results / "test_metrics.csv").exists():
            raise ValueError("This run has test results already. Use evaluate to reproduce; never retune this run.")
        train_experiment(data, config, artifacts, results)
    if args.stage in {"evaluate", "all"}:
        from src.dl.evaluate import evaluate_experiment
        evaluate_experiment(data, config, artifacts, results)


if __name__ == "__main__":
    main()
