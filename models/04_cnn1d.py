#!/usr/bin/env python3
"""
04_cnn1d.py
===========
1D Convolutional Neural Network for VIO drift prediction.

Mathematical Summary
--------------------
Convolution: (x * k)[t] = sum_{i=0}^{K-1} x[t+i] * k[i]  (1D cross-correlation)

Key properties:
  - Translation invariant: same filter weights at every temporal position.
  - Receptive field: Conv(k=5) + Conv(k=3) + Conv(k=3) = 9 timesteps (~45ms @ 200Hz).
  - Parameter sharing reduces parameters vs MLP on raw sequence.
  - GlobalAvgPool aggregates temporal representations to fixed-size vector.

Architecture: (batch, W=50, F) -> transpose to (batch, F, W)
  -> Conv1d(F->64, k=5) -> BN -> ReLU
  -> Conv1d(64->128, k=3) -> BN -> ReLU
  -> Conv1d(128->64, k=3) -> BN -> ReLU
  -> GlobalAvgPool -> Linear(64->64) -> ReLU -> Dropout(0.2) -> Linear(64->3)

Loss: MSE over 3 targets at the last timestep of each window.
Gap-aware: windows crossing logging gaps (dt > 0.1s) excluded.
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
    load_data, split_data, build_windows, scale_sequences,
    compute_metrics, print_metrics, save_metrics, load_config,
    CNN_FEATURE_COLS, TARGET_COLS, SAVED_DIR, RESULTS_DIR
)

MODEL_NAME = "CNN1D"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class DriftCNN1D(nn.Module):
    """
    1D CNN for multi-output regression on IMU sequences.
    Input:  (batch, window, n_features)   [batch-first]
    Output: (batch, 3)
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Block 3
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # Global average pooling (mean over time dim)
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        # x: (B, W, F) -> transpose to (B, F, W) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)            # (B, 64, W)
        x = x.mean(dim=-1)          # GlobalAvgPool -> (B, 64)
        return self.head(x)         # (B, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
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
            total_loss += criterion(model(xb), yb).item() * len(xb)
    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  MODEL 4: 1D CNN   [device: {DEVICE}]")
    print("=" * 60)

    cfg = load_config("cnn")
    hp  = cfg["hyperparameters"]
    W   = hp["window_size"]
    S   = hp["stride"]

    # ── Load & split ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = load_data()
    from utils import split_data
    train_df, val_df, test_df = split_data(df)

    # ── Build windows ─────────────────────────────────────────────────────────
    print(f"\n[2/5] Building windows (W={W}, stride={S})...")
    X_tr, y_tr, _ = build_windows(train_df, CNN_FEATURE_COLS, window_size=W, stride=S)
    X_v,  y_v,  _ = build_windows(val_df,   CNN_FEATURE_COLS, window_size=W, stride=S)
    X_te, y_te, _ = build_windows(test_df,  CNN_FEATURE_COLS, window_size=W, stride=5)
    print(f"  Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    X_tr_s, X_v_s, X_te_s, scaler = scale_sequences(X_tr, X_v, X_te)
    n_features = X_tr_s.shape[2]

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def make_loader(X, y, bs, shuffle):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=shuffle)

    bs = hp["batch_size"]
    train_loader = make_loader(X_tr_s, y_tr, bs, shuffle=True)
    val_loader   = make_loader(X_v_s,  y_v,  bs, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[3/5] Building CNN1D (n_features={n_features}, window={W})...")
    model = DriftCNN1D(n_features).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"],
                                  weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=hp["lr_factor"],
        patience=hp["lr_patience"]
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for up to {hp['max_epochs']} epochs...")
    best_val_loss = np.inf
    patience_counter = 0
    history = []
    t_start = time.time()

    for epoch in range(1, hp["max_epochs"] + 1):
        tr_loss  = train_epoch(model, train_loader, optimizer, criterion,
                                clip_norm=hp["gradient_clip_norm"])
        val_loss = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d} | train={tr_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVED_DIR, "cnn_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= hp["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t_start
    print(f"  Total training time: {train_time:.1f}s")
    pd.DataFrame(history).to_csv(
        os.path.join(RESULTS_DIR, "cnn_training_history.csv"), index=False)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(SAVED_DIR, "cnn_best.pt"),
                                      map_location=DEVICE))
    model.eval()
    print("\n[5/5] Evaluating on TEST flight (log_54, unseen)...")
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(X_te_t).cpu().numpy()
    infer_ms = 1000 * (time.time() - t0) / len(X_te_s) * 1000

    metrics = compute_metrics(y_te, y_pred)
    metrics["window_size"]      = W
    metrics["stride"]           = S
    metrics["best_val_loss"]    = round(float(best_val_loss), 8)
    metrics["epochs_trained"]   = len(history)
    metrics["train_time_s"]     = round(train_time, 1)
    metrics["infer_ms_per_1k"]  = round(infer_ms, 3)
    metrics["n_features"]       = n_features
    metrics["total_params"]     = total_params
    print_metrics(metrics, MODEL_NAME)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(SAVED_DIR, "cnn_model.pt"))
    with open(os.path.join(SAVED_DIR, "cnn_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved → saved_models/cnn_model.pt")
    save_metrics(metrics, MODEL_NAME)

    print(f"\n✓ 1D CNN complete.\n")
    return y_pred, y_te


if __name__ == "__main__":
    main()
