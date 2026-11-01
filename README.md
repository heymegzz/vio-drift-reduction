# ML-Assisted IMU Drift Reduction for GNSS-Denied UAV Navigation

## Project Background

Unmanned aerial vehicles (UAVs) rely on inertial measurement units (IMUs) for
attitude and velocity estimation. In GNSS-denied environments—indoors, urban
canyons, or under deliberate jamming—the inertial navigation solution drifts
over time due to sensor bias, noise, and integration error. This project
investigates whether machine-learning models can predict the velocity residual
between a PX4 estimator output and GPS reference measurements, with the
long-term goal of applying learned corrections during GNSS outages.

## Current Stage: Phase 1 — Machine Learning

This repository contains:
- **Exploratory Data Analysis (EDA)** — dataset understanding, quality
  assessment, target verification, and visualization.
- **Feature Engineering** — leakage-safe IMU feature extraction with temporal
  features (rolling stats, differences, lags, jerk, cross-axis interactions).
- **Data Splitting** — log-based train/val/test split preventing temporal
  leakage.
- **Ridge Regression** — regularized linear baseline model with leave-one-log-out
  cross-validation for hyperparameter tuning.
- **Literature Review** — survey of 12+ relevant works on IMU drift reduction,
  ML for inertial navigation, and sensor fusion.

### Models Implemented

| Model | Type | α / Config | Test RMSE (mean) | Test R² (mean) |
|---|---|---|---|---|
| **Ridge Regression** | Regularized Linear | α=1000.0 | 0.1734 m/s | -0.3726 |

*Additional models (Random Forest, Gradient Boosting) to be added by collaborators.*

## Dataset

| Item     | Value                                        |
|----------|----------------------------------------------|
| File     | `imu_drift_feature_dataset.csv` (repo root)  |
| Rows     | 38,967                                       |
| Columns  | 31 (raw) → 148 (after feature engineering)   |
| Features | 142 input features (IMU-only, leakage-safe)  |
| Size     | ~12 MB                                       |
| Logs     | 4 recording segments                         |

### Data Split (Log-Based)

| Split | Logs | Rows | Ratio |
|-------|------|------|-------|
| Train | log_51, log_53 | 22,004 | 56.5% |
| Val   | log_52 | 8,657 | 22.2% |
| Test  | log_54 | 8,306 | 21.3% |

## Repository Structure

```
README.md
requirements.txt
.gitignore
imu_drift_feature_dataset.csv
feature_engineering.py            # Feature engineering pipeline
models/
  utils.py                        # Shared utilities (metrics, splitting, I/O)
  00_data_preparation.py          # Data loading, feature eng, splitting verification
  01_ridge_regression.py          # Ridge Regression training & evaluation
configs/
  ridge_config.json               # Saved hyperparameters
saved_models/
  ridge_model.pkl                 # Trained Ridge model
  ridge_scaler.pkl                # Fitted StandardScaler
results/
  ridge_metrics.csv               # Test set metrics
  ridge_coefficients.csv          # Feature coefficient analysis
  split_info.csv                  # Train/val/test split details
  plots/
    ridge_alpha_search.png        # Hyperparameter tuning plot
    ridge_scatter_test.png        # Predicted vs true (test)
    ridge_scatter_val.png         # Predicted vs true (validation)
    ridge_residuals_test.png      # Residual distribution (test)
    ridge_coefficients.png        # Top feature coefficients
    ridge_timeseries_test.png     # Time-series comparison (test)
  eda/
    figures/                      # EDA plots
    tables/                       # EDA summary tables
notebooks/
  01_dataset_eda.py               # Jupytext percent-format source
  01_dataset_eda.ipynb            # Jupyter notebook (generated from .py)
docs/
  data_dictionary.md              # Column-level documentation
  target_definition.md            # Target variable definition & caveats
  eda_report.md                   # Full EDA findings report
  literature_review.md            # Literature review (12 papers)
```

## Reproduction

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the EDA notebook (optional — results already saved)
jupyter notebook notebooks/01_dataset_eda.ipynb

# 4. Run data preparation (feature engineering + split verification)
python3 models/00_data_preparation.py

# 5. Train and evaluate Ridge Regression
python3 models/01_ridge_regression.py
```

## Feature Engineering

The pipeline (`feature_engineering.py`) transforms 8 raw IMU channels into
142 input features:

| Feature Group | Count | Description |
|---|---|---|
| Raw IMU | 8 | accel x/y/z, gyro x/y/z, magnitudes |
| Rolling mean/std | 48 | Windows of 10, 50, 100 samples |
| Rolling min/max/range | 54 | Windows of 10, 50, 100 samples |
| First differences | 8 | Rate of change (≈ jerk for accel) |
| Lag features | 18 | 1, 5, 10 step lags |
| Jerk | 3 | Second-order accel differences |
| Cross-axis | 3 | XY angles, accel/gyro magnitude ratio |

All features are computed **per-log** to prevent cross-log contamination.
GPS-derived columns are excluded to maintain GNSS-denied validity.

## Documentation

- [EDA Report](docs/eda_report.md) — detailed findings from exploratory analysis
- [Data Dictionary](docs/data_dictionary.md) — column-level documentation
- [Target Definition](docs/target_definition.md) — what the targets mean
- [Literature Review](docs/literature_review.md) — survey of 12 relevant works

## License

This project is part of an academic coursework submission.
