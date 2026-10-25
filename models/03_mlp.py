#!/usr/bin/env python3
"""
03_mlp.py
=========
Multi-Layer Perceptron (MLP) for VIO drift prediction.

Mathematical Summary
--------------------
Forward pass: z^(l) = sigma(W^(l) z^(l-1) + b^(l))
  - sigma = ReLU for hidden layers (avoids vanishing gradient)
  - BatchNorm: normalises activation per mini-batch, stabilises training
  - Dropout: randomly zero p fraction of neurons — implicit ensemble regularisation

Loss: L = (1/N) sum_i ||y_i - y_hat_i||^2  [multi-output MSE]
Backprop: dL/dW via chain rule; Adam: m_t = beta1*m_{t-1} + (1-beta1)*grad
                                            v_t = beta2*v_{t-1} + (1-beta2)*grad^2
                                            w <- w - lr * m_hat / (sqrt(v_hat) + eps)

Architecture: 34 -> 256 -> BN -> ReLU -> Dropout(0.3) ->
                     128 -> BN -> ReLU -> Dropout(0.2) ->
                      64 -> ReLU -> 3
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    load_data, split_data, prepare_tabular, fit_scaler, apply_scaler,
    compute_metrics, print_metrics, save_metrics, load_config,
    ALL_FEATURE_COLS, TARGET_COLS, SAVED_DIR, RESULTS_DIR
)

MODEL_NAME = "MLP"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────────────────────────────────────

class DriftMLP(nn.Module):
    """
    Feedforward network for multi-output regression.
    Input: (batch, n_features)
    Output: (batch, 3)  — [Δv_N, Δv_E, Δv_D]
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * len(xb)
    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  MODEL 3: MLP   [device: {DEVICE}]")
    print("=" * 60)

    cfg = load_config("mlp")
    hp  = cfg["hyperparameters"]

    # ── Load & split ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    # ── Prepare ───────────────────────────────────────────────────────────────
    print("\n[2/5] Preparing features...")
    X_tr, y_tr, _ = prepare_tabular(train_df)
    X_v,  y_v,  _ = prepare_tabular(val_df)
    X_te, y_te, _ = prepare_tabular(test_df)

    scaler = fit_scaler(X_tr)
    X_tr_s = apply_scaler(scaler, X_tr)
    X_v_s  = apply_scaler(scaler, X_v)
    X_te_s = apply_scaler(scaler, X_te)
    n_features = X_tr_s.shape[1]
    print(f"  Features: {n_features}, Train rows: {X_tr_s.shape[0]}, Val: {X_v_s.shape[0]}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def make_loader(X, y, batch_size, shuffle=True):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)

    bs = hp["batch_size"]
    train_loader = make_loader(X_tr_s, y_tr, bs, shuffle=True)
    val_loader   = make_loader(X_v_s,  y_v,  bs, shuffle=False)

    # ── Model / Optimiser / Scheduler ─────────────────────────────────────────
    print("\n[3/5] Building model...")
    model = DriftMLP(n_features).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp["learning_rate"],
        betas=(hp["beta1"], hp["beta2"]),
        weight_decay=hp["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=hp["lr_factor"],
        patience=hp["lr_patience"]
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for up to {hp['max_epochs']} epochs...")
    best_val_loss = np.inf
    patience_counter = 0
    history = []
    t_start = time.time()

    for epoch in range(1, hp["max_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  clip_norm=hp["gradient_clip_norm"])
        val_loss   = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVED_DIR, "mlp_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= hp["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.6f})")
                break

    train_time = time.time() - t_start
    print(f"  Total training time: {train_time:.1f}s")

    # Save history
    pd.DataFrame(history).to_csv(
        os.path.join(RESULTS_DIR, "mlp_training_history.csv"), index=False)

    # ── Load best weights ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(SAVED_DIR, "mlp_best.pt"),
                                      map_location=DEVICE))

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on TEST flight (log_54, unseen)...")
    model.eval()
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(X_te_t).cpu().numpy()
    infer_ms = 1000 * (time.time() - t0) / len(X_te_s) * 1000

    metrics = compute_metrics(y_te, y_pred)
    metrics["best_val_loss"]    = round(float(best_val_loss), 8)
    metrics["epochs_trained"]   = len(history)
    metrics["train_time_s"]     = round(train_time, 1)
    metrics["infer_ms_per_1k"]  = round(infer_ms, 3)
    metrics["n_features"]       = n_features
    metrics["total_params"]     = total_params
    print_metrics(metrics, MODEL_NAME)

    # ── Save final model + scaler ─────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(SAVED_DIR, "mlp_model.pt"))
    with open(os.path.join(SAVED_DIR, "mlp_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved → saved_models/mlp_model.pt")
    save_metrics(metrics, MODEL_NAME)

    print(f"\n✓ MLP complete.\n")
    return y_pred, y_te


if __name__ == "__main__":
    main()
