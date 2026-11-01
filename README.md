# ML-Assisted IMU Drift Reduction for GNSS-Denied UAV Navigation

## Project Background

Unmanned aerial vehicles (UAVs) rely on inertial measurement units (IMUs) for attitude and velocity estimation. In GNSS-denied environments—indoors, urban canyons, or under deliberate jamming—the inertial navigation solution drifts over time due to sensor bias, noise, and integration error. This project investigates whether machine-learning models can predict the velocity residual between a PX4 estimator output and GPS reference measurements (Δv_N, Δv_E, Δv_D), with the long-term goal of applying learned corrections during GNSS outages.

> **Important caveat:** The PX4 velocity estimates were recorded from the PX4 EKF2 estimator, which likely used GPS aiding during the flight. Removing GPS *input features* from the ML model does **not** remove GPS dependence from the recorded PX4 estimator output. This experiment should be described as *PX4–GPS velocity residual prediction using IMU-derived features*, not as validated GNSS-denied or IMU-only drift reduction. See [`docs/target_definition.md`](docs/target_definition.md) for a full discussion.

---

## Phase 1: Machine Learning Benchmark

In Phase 1, we implement, benchmark, and evaluate **3 classical ML models** and **2 baselines** trained exclusively on leakage-safe IMU features:

**Baselines:**
- **Zero Correction** — predicted residual = 0 (leave PX4 velocity uncorrected)
- **Mean Predictor** — predicted residual = training-set target mean

**ML Models:**
1. **Ridge Regression** (Multi-output L₂ regularized linear baseline)
2. **Random Forest Regressor** (Bagged tree ensemble)
3. **Histogram Gradient Boosting** (Stage-wise additive tree booster)

### Benchmark Summary on Test Flight (`log_54`)

All models were trained on `log_51` + `log_53` (22,004 rows), validated on `log_52` (8,657 rows), and evaluated on the held-out `log_54` flight (8,306 rows):

| Model | RMSE (m/s) | MAE (m/s) | R² | Bias (m/s) |
|---|---|---|---|---|
| **Zero Correction** | **0.1472** | **0.1065** | **-0.013** | 0.017 |
| Random Forest | 0.1626 | 0.1189 | -0.171 | 0.058 |
| Gradient Boosting | 0.1684 | 0.1252 | -0.255 | 0.064 |
| Mean Predictor | 0.1699 | 0.1268 | -0.239 | 0.069 |
| Ridge Regression | 0.1746 | 0.1319 | -0.387 | 0.070 |

> **Key finding:** No ML model achieves lower RMSE than zero correction on this test flight. Applying any of the trained corrections **worsens** the overall velocity estimate compared to leaving PX4 output uncorrected. All R² values are negative, meaning every model performs worse than predicting the target mean.

> This is a common outcome when residuals are small, noisy, and weakly correlated with the available IMU features. The result is reported honestly as a negative finding and serves as a reproducible benchmark for Phase 2.

---

## Dataset

| Property | Value |
|---|---|
| File | `data/imu_drift_feature_dataset.csv` |
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

> **Note:** `log_54` has already been inspected during development and is no longer a pristine unseen test. A new independent recording is recommended for stronger confirmation.

---

## Feature Engineering Pipeline

The pipeline ([`src/feature_engineering.py`](src/feature_engineering.py)) transforms 8 raw IMU channels into 142 features computed strictly **per continuous segment**:

- **Gap Detection:** Timestamps are in microseconds. Intervals > 50 ms (~10× the median 5 ms cadence) are treated as temporal discontinuities. Each log is split into continuous segments; all temporal features are reset at segment boundaries.
- **Rolling Statistics:** Sample-count windows of 10, 50, and 100 (~50 ms, 250 ms, 500 ms at 200 Hz) for moving mean, standard deviation, and dynamic range.
- **Temporal Derivatives:** First-order differences (rate of change).
- **Jerk:** da/dt computed using actual elapsed time between consecutive samples (m/s³), not second-order differencing.
- **Lag States:** Prior observations at t−1, t−5, t−10 samples. Startup rows are forward-filled (causal — no future leakage).
- **Kinematic Cross-Axis Interactions:** Planar tilt angles (atan2) and specific force to angular rate ratios.
- **Leakage Quarantine:** All GPS coordinates, GPS velocities, satellite counts, quality metrics (EPH/EPV), and estimator flags are permanently excluded.

---

## Repository Structure

```
├── data/                               # Dataset
│   └── imu_drift_feature_dataset.csv   # Raw telemetry (38,967 × 31)
├── src/                                # Shared library code
│   ├── __init__.py
│   ├── feature_engineering.py          # End-to-end feature engineering pipeline
│   └── utils.py                        # Evaluation metrics, data loading & splitting
├── models/                             # Training & evaluation scripts
│   ├── 00_data_preparation.py          # Data loading, feature eng & split auditing
│   ├── 01_ridge_regression.py          # Multi-output Ridge regression baseline
│   ├── 02_random_forest.py             # Multi-output Random Forest ensemble
│   ├── 03_gradient_boosting.py         # Multi-output HistGradientBoosting regressor
│   └── 04_comparison.py               # Cross-model benchmark with baselines
├── notebooks/                          # Interactive analysis notebooks
│   ├── 01_dataset_eda.py               # EDA (Jupytext percent-format source)
│   ├── 01_dataset_eda.ipynb            # EDA (Jupyter notebook)
│   ├── 02_model_benchmark.py           # Model benchmark (Jupytext source)
│   └── 02_model_benchmark.ipynb        # Model benchmark (Jupyter notebook)
├── configs/                            # Hyperparameters & training configurations
│   ├── ridge_config.json
│   ├── rf_config.json
│   └── gbm_config.json
├── docs/                               # Academic & technical documentation
│   ├── data_dictionary.md              # Column-level telemetry schema
│   ├── target_definition.md            # Velocity residual derivations & caveats
│   ├── eda_report.md                   # Full exploratory data analysis report
│   ├── literature_review.md            # Survey of academic papers
│   ├── theoretical_rigor.md            # Mathematical formulas & model justifications
│   ├── project_presentation.md         # Slide deck notes & viva voce Q&A
│   ├── project_report_ieee.tex         # IEEE format conference paper
│   └── references.bib                  # BibTeX bibliography
├── results/                            # All experimental outputs
│   ├── eda/                            # Exploratory analysis
│   │   ├── figures/                    # EDA diagnostic plots
│   │   └── tables/                     # EDA summary tables
│   ├── models/                         # Model evaluation metrics & reports
│   │   ├── metrics_table.csv           # Combined benchmark with baselines
│   │   ├── correction_evaluation.csv   # Velocity correction quality assessment
│   │   ├── model_comparison_report.md  # Analysis of results
│   │   └── ...                         # Per-model metrics
│   └── plots/                          # Model evaluation & comparison plots
│       ├── correction_velocity_comparison.png
│       ├── comparison_rmse_bar.png
│       └── ...                         # Per-model scatter, residual, timeseries
├── saved_models/                       # Pickled model checkpoints & fitted scalers
├── tests/                              # Unit test suite
│   ├── __init__.py
│   ├── test_feature_engineering.py     # Feature pipeline, gap handling & jerk tests
│   └── test_model_inference.py         # Model loading & prediction shape tests
├── .github/workflows/ci.yml           # GitHub Actions CI pipeline
├── Makefile                            # Build automation targets
├── requirements.txt                    # Python package dependencies
└── README.md
```

---

## How to Reproduce

```bash
# 1. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the EDA notebook (exploratory data analysis)
jupyter notebook notebooks/01_dataset_eda.ipynb
# Or run the Python source directly:
#   cd notebooks && python 01_dataset_eda.py

# 4. Run data preparation and verify split integrity
python3 models/00_data_preparation.py

# 5. Train models
python3 models/01_ridge_regression.py
python3 models/02_random_forest.py
python3 models/03_gradient_boosting.py

# 6. Generate comparison benchmark (includes baselines and correction evaluation)
python3 models/04_comparison.py
```

All saved figures, tables, and model metrics are checked into the `results/` directory.

For detailed EDA findings, see [`docs/eda_report.md`](docs/eda_report.md).
For column-level documentation, see [`docs/data_dictionary.md`](docs/data_dictionary.md).
For target variable definitions, see [`docs/target_definition.md`](docs/target_definition.md).
For the literature survey, see [`docs/literature_review.md`](docs/literature_review.md).

---

## Authors & Contributions

- **Meghna Nair:** Dataset understanding, quality assessment, target verification, exploratory data analysis (EDA), data dictionary, and EDA report documentation.
- **Pranjal Shukla:** Data pipeline architecture, leakage quarantine, feature engineering, and Ridge regression baseline.
- **Shreya Narayani:** Random Forest ensemble, Gradient Boosting implementation, model comparison benchmark, theoretical justifications, and project presentation/report.

---

## License

This project is part of an academic coursework submission.
