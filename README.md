# VIO Drift Reduction — Multi-Model Benchmark

This repository implements, trains, and evaluates 5 machine learning and deep learning models to predict velocity drift residuals ($\Delta v_N, \Delta v_E, \Delta v_D$) on unmanned aerial vehicle (UAV) flight logs under GPS-denied or degraded conditions using IMU-derived features.

---

## 1. Models Implemented & Compared

| Model | Type | Architecture / Config | Parameters | Latency (1k samples) |
|---|---|---|---|---|
| **Ridge Regression** | Regularized Linear | Multi-output Ridge ($\alpha=100.0$) | 105 | 0.056 ms |
| **Random Forest** | Tree Ensemble | 300 trees, `max_depth=20`, `max_features='sqrt'` | ~256 MB pkl | 18.64 ms |
| **MLP** | Feedforward Neural Net | 34 → 256 → BN → ReLU → Drop(0.3) → 128 → BN → ReLU → Drop(0.2) → 64 → ReLU → 3 | 51,075 | 1.62 ms |
| **1D CNN** | Temporal ConvNet | Conv1D(64, k=5) → Conv1D(128, k=3) → Conv1D(64, k=3) → GAP → FC(64) → FC(3) | 61,315 | 831.1 ms |
| **LSTM** | Recurrent Neural Net | 2-layer Stacked LSTM(128), Drop(0.2) → FC(64) → FC(3) | 224,515 | 796.5 ms |

---

## 2. Benchmark Results on Unseen Test Flight (`log_54`)

Models were strictly trained on `log_51` + `log_53`, validated on `log_52` (early stopping / tuning), and tested on `log_54` (8,306 samples, 57.6 s):

| Model | MAE Mean (m/s) | RMSE Mean (m/s) | $R^2$ Mean | RMSE $\Delta v_N$ | RMSE $\Delta v_E$ | RMSE $\Delta v_D$ |
|---|---|---|---|---|---|---|
| **1D CNN** | **0.1038** | **0.1460** | **-0.0329** | 0.1052 | **0.0996** | **0.2332** |
| **Random Forest** | 0.1100 | 0.1536 | -0.0570 | **0.0992** | 0.1033 | 0.2582 |
| **Ridge Regression** | 0.1185 | 0.1616 | -0.1621 | 0.1022 | 0.1085 | 0.2742 |
| **LSTM** | 0.1263 | 0.1652 | -0.3500 | 0.1135 | 0.1026 | 0.2795 |
| **MLP** | 0.1284 | 0.1738 | -0.3204 | 0.1090 | 0.1110 | 0.3015 |

---

## 3. Directory Layout

```text
├── configs/                  # Saved hyperparameters and architecture configurations
│   ├── ridge_config.json
│   ├── rf_config.json
│   ├── mlp_config.json
│   ├── cnn_config.json
│   └── lstm_config.json
├── models/                   # Training and evaluation pipelines
│   ├── 00_data_preparation.py
│   ├── 01_ridge_regression.py
│   ├── 02_random_forest.py
│   ├── 03_mlp.py
│   ├── 04_cnn1d.py
│   ├── 05_lstm.py
│   ├── 06_comparison.py
│   └── utils.py
├── saved_models/             # Checkpoints, pickled models, and fitted scalers
├── results/                  # Evaluation outputs
│   ├── metrics_table.csv
│   ├── model_comparison_report.md
│   └── plots/                # Evaluation visualizations
```

---

## 4. How to Reproduce

```bash
# 1. Verify data preparation and leakage safeguards
python3 models/00_data_preparation.py

# 2. Train baseline models
python3 models/01_ridge_regression.py
python3 models/02_random_forest.py

# 3. Train neural architectures
python3 models/03_mlp.py
python3 models/04_cnn1d.py
python3 models/05_lstm.py

# 4. Run cross-model benchmark and generate comparison plots & report
python3 models/06_comparison.py
```