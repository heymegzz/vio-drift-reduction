# ML-Assisted IMU Drift Reduction for GNSS-Denied UAV Navigation

## Project Background

Unmanned aerial vehicles (UAVs) rely on inertial measurement units (IMUs) for
attitude and velocity estimation. In GNSS-denied environments—indoors, urban
canyons, or under deliberate jamming—the inertial navigation solution drifts
over time due to sensor bias, noise, and integration error. This project
investigates whether machine-learning models can predict the velocity residual
between a PX4 estimator output and GPS reference measurements, with the
long-term goal of applying learned corrections during GNSS outages.

## Current Stage: Exploratory Data Analysis (EDA)

This repository currently contains only the **dataset understanding, quality
assessment, target verification, and exploratory data analysis** work. The
objective is to thoroughly characterize the dataset before any modelling begins.

### What this stage covers

- Dataset provenance and integrity verification.
- Schema inspection and column documentation.
- Missing-value and duplicate auditing.
- Recording-level sampling analysis.
- IMU, GPS, and PX4 signal profiling.
- Target-variable definition and numerical verification.
- Distribution analysis and potential-outlier flagging.
- Temporal and cross-signal visualizations.
- Correlation structure and temporal-dependence analysis.
- Data dictionary and full EDA report.

### What this stage does NOT demonstrate

- IMU drift reduction or correction has not been achieved.
- No model has been trained, validated, or evaluated.
- No feature engineering, feature selection, or train/test splitting has been
  performed.
- No data cleaning (imputation, clipping, outlier removal) has been applied.
- The dataset has not been regenerated or augmented.

## Dataset

| Item     | Value                                        |
|----------|----------------------------------------------|
| File     | `imu_drift_feature_dataset.csv` (repo root)  |
| Rows     | 38,967                                       |
| Columns  | 31                                           |
| Size     | ~12 MB                                       |
| Logs     | 4 recording segments                         |

The dataset was extracted from PX4 `.ulg` flight-log files. The original
extraction script is not included in this repository; see
`docs/eda_report.md` for a discussion of missing provenance materials.

## Repository Structure

```
README.md
requirements.txt
.gitignore
imu_drift_feature_dataset.csv
notebooks/
  01_dataset_eda.ipynb
docs/
  data_dictionary.md
  target_definition.md
  eda_report.md
results/
  eda/
    figures/          # Saved plots
    tables/           # Exported summary tables
```

## Reproduction

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the EDA notebook
jupyter notebook notebooks/01_dataset_eda.ipynb
# Execute all cells in order from a clean kernel.
```

## License

This project is part of an academic coursework submission.