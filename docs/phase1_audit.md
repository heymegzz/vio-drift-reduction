# Phase 1 audit and Phase 2 integration

## Decision

Phase 1 does not need to be discarded or rewritten. Its complete offline ML
benchmark is a useful historical baseline. Phase 2 preserves its dataset,
feature code, fitted models, metrics, notebooks, report and presentation. The
scientific requirement is to compare learned corrections against leaving PX4
unchanged, rather than assuming a larger model will help.

## Repository findings

| Area | Existing implementation and status |
| --- | --- |
| Problem | Predict recorded PX4 velocity minus GPS velocity along North/East/Down. This is an instantaneous residual target, not integrated trajectory drift. |
| Dataset | `data/imu_drift_feature_dataset.csv`: 38,967 rows, 31 native columns, four recordings. Source ULogs/extraction and fusion configuration are not supplied. |
| Data preparation | `models/00_data_preparation.py`, `src/utils.py`: explicit log-based split; 22,004 training, 8,657 validation, 8,306 test rows. |
| Feature pipeline | `src/feature_engineering.py`: eight IMU inputs expand to 142 features using causal rolling statistics, differences, elapsed-time jerk, lags and interactions. History resets across gaps over 50 ms. The dataset has 44 continuous segments. |
| Leakage controls | GPS/estimator/reference columns and identifiers are excluded from predictors; scalers fit training only. This does not remove possible GPS aiding from the recorded PX4 target component. |
| Classical models | `models/01_ridge_regression.py`, `02_random_forest.py`, `03_gradient_boosting.py`: fitted multi-output Ridge, RF and wrapped HGB, with saved estimators/scalers and configurations. |
| Evaluation | `models/04_comparison.py`: zero and training-mean baselines, per-axis/macro metrics, correction comparison and plots. Saved historical predictions reproduce the original metric values under sklearn 1.9.0. |
| Notebooks | Paired Python/Jupyter EDA and benchmark notebooks exist under `notebooks/`; they support analysis rather than a separate DL training implementation. |
| Documentation | Data dictionary, EDA report, target definition, literature/theory, IEEE source/PDF and Phase 1 presentation exist. Report and PPT remain unchanged. |
| Tests/automation | Fourteen existing tests cover features, split separation, metric signs and saved-model inference. Makefile and CI already automate the ML workflow. |

## Findings requiring care

All three original regressors have higher historical-test mean-axis RMSE than
zero correction: Ridge 0.174600, RF 0.162646, HGB 0.168448, versus zero 0.147245
m/s. This is a completed negative experiment, not evidence of a failed codebase.

The original HGB constructor leaves internal early stopping at its default,
which can use a random internal validation subset on this dataset. Phase 2
retuning explicitly disables it and chooses iteration count on whole log 52.
The historical model remains an unchanged comparator.

The four logs have strong temporal dependence and come from a short recording
session. Their count, not the number of overlapping windows, limits evidence for
generalization. Log 54 was already inspected during Phase 1. It cannot be
relabelled as a pristine unseen test.

The Down residual mean changes sign between training and validation/test. This
is directly observable distribution shift. Missing attitude/heading and source
timing/fusion information prevent assigning a physical explanation or validating
a true GNSS outage. The repository contains no camera stream or visual frontend;
the completed benchmark does not implement visual-inertial odometry.

## Changes made for Phase 2

1. Retain the established splits, target sign, feature definitions and historical
   results. Reuse the feature pipeline as the bridge to the feature MLP.
2. Record a frozen experimental protocol and separate output directories.
3. Validate raw data and build lazy, masked causal windows with training-only
   normalization. Add causality, boundary and streaming parity checks.
4. Retune all three classical families separately, then compare feature/raw
   MLPs, causal TCN and GRU using bounded searches and five prescribed seeds.
5. Add gain selection including zero, common-history evaluation, conditional
   block intervals, input sensitivity, and end-to-end local CPU timing.
6. Pin sklearn to the historical artifact version and require pandas 2.2–2.x
   for the existing feature API. Keep PyTorch in a separate dependency file.
7. Keep the Phase 1 CI/test entry point usable without installing PyTorch; add
   a separate DL correctness workflow and full-suite target.

The trained result and limitations are in
[`REPORT.md`](../results/phase2/phase2_v2/REPORT.md). The next scientific step is
independent data with verified synchronization, coordinate conventions and
estimator aiding, followed by an independently scored outage experiment.
Repeatedly tuning on the existing test would not resolve those limitations.
