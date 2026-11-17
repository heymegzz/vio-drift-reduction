"""Classical selection on a whole held-out recording; original artifacts untouched."""
import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits

from src.dl.data import ROOT
from src.dl.train import save_json
from src.utils import compute_metrics


def make_estimator(family, params):
    if family == "ridge":
        return Ridge(**params)
    if family == "rf":
        return RandomForestRegressor(random_state=42, n_jobs=2, **params)
    if family == "hgb":
        # No random internal holdout: external log 52 selects explicit iterations.
        return MultiOutputRegressor(HistGradientBoostingRegressor(
            early_stopping=False, random_state=42, **params), n_jobs=1)
    raise ValueError(family)


def fit_baselines(data, config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xtr, xv = data.inputs("train", "mlp"), data.inputs("val", "mlp")
    result = {}
    for family, candidates in config.items():
        trials = []
        for index, params in enumerate(candidates):
            directory = out / family / str(index)
            model_path, score_path = directory / "model.joblib", directory / "training.json"
            if model_path.exists() and score_path.exists():
                record = json.loads(score_path.read_text())
                if record["params"] != params or record["dataset_sha256"] != data.checksum:
                    raise ValueError("Baseline run identity changed; use another run directory")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                model = make_estimator(family, params)
                with threadpool_limits(limits=2):
                    model.fit(xtr, data.parts["train"].y)
                    prediction = model.predict(xv)
                record = {"family": family, "params": params, "dataset_sha256": data.checksum,
                          "training_seconds": time.perf_counter() - started,
                          "validation": compute_metrics(data.parts["val"].y, prediction)}
                joblib.dump({"model": model, "data": data.metadata()}, model_path, compress=3)
                save_json(score_path, record)
            record["path"] = str(model_path)
            record["trial"] = index
            trials.append(record)
            print(f"baseline {family} {index + 1}/{len(candidates)} val={record['validation']['rmse_mean']:.6f}", flush=True)
        best = min(trials, key=lambda r: r["validation"]["rmse_mean"])
        result[family] = {"best": best, "trials": trials}
    save_json(out / "selection.json", result)
    return result


def historical_predictions(data, split):
    predictions = {}
    # Only trusted, user-supplied repository artifacts are loaded.
    for family in ["ridge", "rf", "gbm"]:
        with (ROOT / "saved_models" / f"{family}_model.pkl").open("rb") as stream:
            model = pickle.load(stream)
        with (ROOT / "saved_models" / f"{family}_scaler.pkl").open("rb") as stream:
            scaler = pickle.load(stream)
        with threadpool_limits(limits=2):
            predictions[f"historical_{family}"] = model.predict(scaler.transform(data.parts[split].engineered))
    return predictions


def predict_baseline(path, x):
    artifact = joblib.load(path)
    with threadpool_limits(limits=2):
        return artifact["model"].predict(x)
