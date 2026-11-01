# ML-Assisted IMU Drift Reduction for GNSS-Denied UAV Navigation

## Project Background

Unmanned aerial vehicles (UAVs) rely on inertial measurement units (IMUs) for attitude and velocity estimation. In GNSS-denied environments—indoors, urban canyons, or under deliberate jamming—the inertial navigation solution drifts over time due to sensor bias, noise, and integration error. This project investigates whether machine-learning models can predict the velocity residual between a PX4 estimator output and GPS reference measurements ($\Delta v_N, \Delta v_E, \Delta v_D$), with the long-term goal of applying learned corrections during GNSS outages.

---

## Phase 1: Machine Learning Benchmark

In Phase 1, we implement, benchmark, and evaluate **3 classical machine learning models** trained exclusively on leakage-safe IMU features:
1. **Ridge Regression** (Multi-output $L_2$ regularized linear baseline)
2. **Random Forest Regressor** (De-correlated bagged tree ensemble)
3. **Histogram Gradient Boosting** (Stage-wise additive tree booster)

### Benchmark Summary on Unseen Test Flight (`log_54`)

All models were trained on `log_51` + `log_53` (22,004 rows), validated on `log_52` (8,657 rows), and evaluated on the held-out `log_54` flight (8,306 rows, 57.6 s):

| Model | Test RMSE (m/s) | Test MAE (m/s) | Test $R^2$ | RMSE $\Delta v_N$ | RMSE $\Delta v_E$ | RMSE $\Delta v_D$ | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **0.1603** | **0.1155** | **-0.1319** | **0.1003** | **0.1017** | **0.2789** | 1.25 ms |
| **Gradient Boosting** | 0.1682 | 0.1241 | -0.2541 | 0.1063 | 0.1081 | 0.2901 | 0.42 ms |
| **Ridge Regression** | 0.1734 | 0.1315 | -0.3726 | 0.1155 | 0.1161 | 0.2885 | **0.05 ms** |

> **Key Observation:** Tree ensembles significantly improve upon the linear Ridge baseline, demonstrating positive $R^2$ on horizontal axes ($\Delta v_N, \Delta v_E$) and confirming non-linear kinematic couplings.

---

## Dataset

| Property | Value |
|---|---|
| File | `imu_drift_feature_dataset.csv` |
| Total Rows | 38,967 |
| Native Columns | 31 (1 string, 30 numeric) |
| Engineered Features | **142 input features** (IMU-only, leakage-safe) |
| Recording Logs | 4 flights (`log_51` to `log_54`) |
| Cadence | ~200 Hz (~5 ms intervals) |

### Split Strategy (Autocorrelation & Leakage Protection)

Because velocity residuals exhibit strong temporal autocorrelation over multiple seconds, random row-wise splitting causes severe data leakage. We use strict whole-flight log partitioning:

- **Train:** `log_51` + `log_53` (22,004 rows, 56.5%)
- **Validation:** `log_52` (8,657 rows, 22.2%)
- **Test:** `log_54` (8,306 rows, 21.3%)

---

## Feature Engineering Pipeline

The pipeline ([`feature_engineering.py`](feature_engineering.py)) transforms 8 raw IMU channels into 142 features computed strictly **per flight log**:

- **Rolling Statistics:** Windows of 10, 50, and 100 samples (~50ms, 250ms, 500ms) for moving mean, standard deviation, and dynamic range.
- **Temporal Derivatives:** First-order differences (rate of change) and second-order acceleration jerk ($d^2a/dt^2$).
- **Lag States:** Prior observations at $t-1$, $t-5$, $t-10$.
- **Kinematic Cross-Axis Interactions:** Planar tilt angles ($\text{atan2}$) and specific force to angular rate ratios.
- **Leakage Quarantine:** All GPS coordinates, GPS velocities, satellite counts, quality metrics (EPH/EPV), and estimator flags are permanently excluded.

---

## Repository Structure

```
├── configs/                          # Hyperparameters & training configurations
│   ├── ridge_config.json
│   ├── rf_config.json
│   └── gbm_config.json
├── docs/                             # Academic & technical documentation
│   ├── data_dictionary.md            # Column-level telemetry schema
│   ├── target_definition.md          # Velocity residual derivations & caveats
│   ├── eda_report.md                 # Full exploratory data analysis report
│   ├── literature_review.md          # Survey of 12 academic papers
│   ├── theoretical_rigor.md          # Mathematical formulas & model justifications
│   ├── project_presentation.md       # Slide deck notes & viva voce Q&A
│   └── project_report_ieee.tex       # Complete IEEE format conference paper
├── models/                           # Pipelines & model implementations
│   ├── utils.py                      # Shared evaluation metrics & dataset loaders
│   ├── 00_data_preparation.py        # Data loading, feature eng & split auditing
│   ├── 01_ridge_regression.py        # Multi-output Ridge regression baseline
│   ├── 02_random_forest.py           # Multi-output Random Forest ensemble
│   ├── 03_gradient_boosting.py       # Multi-output HistGradientBoosting regressor
│   └── 04_comparison.py              # Cross-model evaluation & benchmark plots
├── results/                          # Benchmark tables & figures
│   ├── metrics_table.csv             # Combined model benchmark metrics
│   ├── model_comparison_report.md    # Analysis of results
│   ├── rf_feature_importances.csv    # Gini importance scores
│   ├── split_info.csv                # Dataset partition statistics
│   ├── eda/                          # Exploratory analysis figures & tables
│   └── plots/                        # Model evaluation & comparison plots
│       ├── comparison_rmse_bar.png
│       ├── comparison_mae_bar.png
│       ├── comparison_timeseries_all_models.png
│       ├── rf_feature_importance.png
│       └── ...
├── saved_models/                     # Pickled model checkpoints & fitted scalers
├── feature_engineering.py            # End-to-end feature engineering pipeline
├── requirements.txt                  # Python package dependencies
└── README.md
```

---

## How to Reproduce

```bash
# 1. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run data preparation and verify split integrity
python3 models/00_data_preparation.py

# 3. Train models
python3 models/01_ridge_regression.py
python3 models/02_random_forest.py
python3 models/03_gradient_boosting.py

# 4. Generate comparison benchmark tables and plots
python3 models/04_comparison.py
```

---

## Authors & Contributions

- **Pranjal Shukla:** Data pipeline architecture, exploratory data analysis, leakage quarantine, feature engineering, and Ridge regression baseline.
- **Shreya Narayani:** Random Forest ensemble, Gradient Boosting implementation, model comparison benchmark, theoretical justifications, and project presentation/report.
