"""Deterministic bounded neural training with validation-only checkpoint choice."""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dl.models import build_model
from src.utils import compute_metrics


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def predict(model, dataset, batch_size=512):
    model.eval()
    values = []
    with torch.inference_mode():
        for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            values.append(model(x).numpy())
    return np.concatenate(values).astype(np.float64)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def fit_neural(data, config, seed, directory, train_settings):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "model.pt"
    metadata_path = directory / "training.json"
    if checkpoint_path.exists() and metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old["config"] != config or old["seed"] != seed or old["dataset_sha256"] != data.checksum or old["settings"] != train_settings:
            raise ValueError(f"Run identity differs at {directory}; choose a new run directory")
        model, _ = load_neural(checkpoint_path)
        return model, old
    seed_everything(seed, train_settings.get("threads", 2))
    training = data.dataset("train", config, stride=train_settings["train_stride"])
    validation = data.dataset("val", config, stride=train_settings["validation_stride"])
    inputs = training[0][0].shape[-1]
    model = build_model(config, inputs)
    loader = DataLoader(training, batch_size=train_settings["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", .001),
                                 weight_decay=config.get("weight_decay", .0001))
    loss_fn = nn.MSELoss() if config.get("loss", "mse") == "mse" else nn.HuberLoss(delta=.1)
    val_y = data.parts["val"].y[validation.endpoints]
    initial_score = compute_metrics(val_y, predict(model, validation))["rmse_mean"]
    # Compare trained epochs here; zero correction remains a separate competitor.
    # Otherwise an epoch-zero checkpoint would conceal unsuccessful learned fits.
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, stale, history = 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, train_settings["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        val_rmse = compute_metrics(val_y, predict(model, validation))["rmse_mean"]
        history.append({"epoch": epoch, "train_loss": total / count, "val_rmse": float(val_rmse)})
        if val_rmse < best_score - 1e-6:
            best_score, best_epoch, stale = float(val_rmse), epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= train_settings["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    val_prediction = predict(model, data.dataset("val", config))
    details = {"config": config, "seed": seed, "settings": train_settings,
               "dataset_sha256": data.checksum, "best_epoch": best_epoch,
               "initial_zero_validation_rmse": float(initial_score),
               "epochs_run": len(history), "training_seconds": time.perf_counter() - started,
               "parameters": sum(p.numel() for p in model.parameters()),
               "validation": compute_metrics(data.parts["val"].y, val_prediction), "history": history}
    torch.save({"state_dict": model.state_dict(), "inputs": inputs, "config": config,
                "data": data.metadata(), "seed": seed}, checkpoint_path)
    save_json(metadata_path, details)
    return model, details


def load_neural(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"], checkpoint["inputs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
