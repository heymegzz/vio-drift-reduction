#!/usr/bin/env python3
"""
05_lstm.py
==========
LSTM (Long Short-Term Memory) for VIO drift prediction.

Mathematical Summary
--------------------
At each timestep t:

  f_t = sigma(W_f [h_{t-1}, x_t] + b_f)          # forget gate
  i_t = sigma(W_i [h_{t-1}, x_t] + b_i)          # input gate
  g_t = tanh  (W_g [h_{t-1}, x_t] + b_g)         # cell candidate
  c_t = f_t * c_{t-1} + i_t * g_t                 # cell state update
  o_t = sigma(W_o [h_{t-1}, x_t] + b_o)          # output gate
  h_t = o_t * tanh(c_t)                            # hidden state

Key properties:
  - f_t decides what historical information to erase from cell memory.
  - i_t * g_t writes new information into the cell.
  - Additive cell update: c_t = f_t * c_{t-1} + i_t * g_t
    allows gradient to flow with minimal vanishing across long sequences.
  - Window mode (stateless): hidden states reset at each window start.

Architecture:
  (batch, W=100, 34) -> LSTM(128, 2 layers, dropout=0.2)
  -> last h_T -> Linear(128->64) -> ReLU -> Dropout(0.2) -> Linear(64->3)

Loss: MSE over 3 targets.
Gap-aware: windows crossing dt > 0.1s gaps excluded.
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
    ALL_FEATURE_COLS, TARGET_COLS, SAVED_DIR, RESULTS_DIR
)

MODEL_NAME = "LSTM"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class DriftLSTM(nn.Module):
    """
    Stacked LSTM for multi-output regression on IMU sequences.
    Input:  (batch, window, n_features)
    Output: (batch, 3)
    """
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2,
                 lstm_dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        # x: (B, W, F)
        out, _ = self.lstm(x)   # out: (B, W, H)
        last = out[:, -1, :]    # last hidden state (B, H)
        return self.head(last)  # (B, 3)


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
    print(f"  MODEL 5: LSTM   [device: {DEVICE}]")
    print("=" * 60)

    cfg = load_config("lstm")
    hp  = cfg["hyperparameters"]
    W   = hp["window_size"]
    S   = hp["stride"]

    # ── Load & split ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    # ── Build windows ─────────────────────────────────────────────────────────
    print(f"\n[2/5] Building windows (W={W}, stride={S})...")
    X_tr, y_tr, _ = build_windows(train_df, ALL_FEATURE_COLS, window_size=W, stride=S)
    X_v,  y_v,  _ = build_windows(val_df,   ALL_FEATURE_COLS, window_size=W, stride=S)
    X_te, y_te, _ = build_windows(test_df,  ALL_FEATURE_COLS, window_size=W, stride=5)
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
    print(f"\n[3/5] Building LSTM (n_features={n_features}, hidden={hp['hidden_size']}, "
          f"layers={hp['num_layers']})...")
    model = DriftLSTM(
        n_features=n_features,
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        lstm_dropout=0.2,
    ).to(DEVICE)
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
            torch.save(model.state_dict(), os.path.join(SAVED_DIR, "lstm_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= hp["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t_start
    print(f"  Total training time: {train_time:.1f}s")
    pd.DataFrame(history).to_csv(
        os.path.join(RESULTS_DIR, "lstm_training_history.csv"), index=False)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(SAVED_DIR, "lstm_best.pt"),
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
    metrics["hidden_size"]      = hp["hidden_size"]
    metrics["num_layers"]       = hp["num_layers"]
    metrics["best_val_loss"]    = round(float(best_val_loss), 8)
    metrics["epochs_trained"]   = len(history)
    metrics["train_time_s"]     = round(train_time, 1)
    metrics["infer_ms_per_1k"]  = round(infer_ms, 3)
    metrics["n_features"]       = n_features
    metrics["total_params"]     = total_params
    print_metrics(metrics, MODEL_NAME)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(SAVED_DIR, "lstm_model.pt"))
    with open(os.path.join(SAVED_DIR, "lstm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved → saved_models/lstm_model.pt")
    save_metrics(metrics, MODEL_NAME)

    print(f"\n✓ LSTM complete.\n")
    return y_pred, y_te


if __name__ == "__main__":
    main()
