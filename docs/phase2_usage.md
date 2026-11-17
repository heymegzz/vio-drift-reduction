# Phase 2 user and research guide

## Start here

Open `results/phase2/phase2_v2/REPORT.html` for the standalone visual report.
`REPORT.md` is the version suitable for GitHub. The original Phase 1 report and
PPT are unchanged. This phase investigates recorded PX4–GPS residuals; no
onboard system or independently validated GNSS outage is claimed.

## Reproduce

Use Python 3.12 in a virtual environment. Install
`requirements-phase2-lock.txt` for the recorded environment, or
`requirements-dl.txt` for the supported direct dependencies. The reference run
uses scikit-learn 1.9.0, matching the original saved estimator metadata.

To reproduce evaluation from the included checkpoints, run from the repository
root in the recorded environment:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m src.dl.experiment --stage evaluate
python -m src.dl.explain
python -m src.dl.report
```

The default configuration is `configs/dl/benchmark.json`. Training resumes
identical completed trials, but refuses a changed dataset, training source or
environment. Once a run has test results, it cannot be retuned in place. To
retrain from scratch, copy the configuration to `configs/dl/reproduction.json`
and change its `name` to `phase2_reproduction`, then run:

```bash
python -m src.dl.experiment --config configs/dl/reproduction.json --stage all
python -m src.dl.explain --run phase2_reproduction
python -m src.dl.report --run phase2_reproduction
```

Do not select changes by their historical test performance.
`--smoke` uses a separate run name, one candidate per family, one seed and two
epochs; it is a functional check, not a scientific result.

## Artifact map

| File | Meaning |
| --- | --- |
| `saved_models/phase2/phase2_v2/decision.json` | Full configuration choices, five replicas and gain policy frozen using validation |
| `identity.json` in that directory | Training source fingerprint, dataset checksum, configuration and versions |
| `search/*/*/training.json` | Every neural trial's settings, training curve, chosen epoch and validation metrics |
| `classical/*/*/training.json` | Every classical trial's configuration, time and validation metrics |
| `results/phase2/phase2_v2/summary.csv` | Readable main comparison |
| `test_metrics.csv` | Every raw, gated, ensemble and individual-seed metric |
| `test_predictions.npz` + `test_rows.csv` | Aligned predictions and immutable sample identifiers |
| `validation_metrics.csv` | Evidence used for selection |
| `common_history_metrics.csv` | Same full-history rows for all models at 100/200/400 samples |
| `conditional_block_intervals.csv` | Recording-conditional paired block interval sensitivity |
| `selected_correction.csv` | Selected policy, original PX4, GPS reference and corrected velocities |
| `input_sensitivity.csv` | Validation-only training-mean input replacement diagnostic |
| `streaming_timing.csv` | Local CPU preprocessing plus inference for one neural checkpoint |

Unselected classical checkpoint binaries and smoke runs are excluded from Git
to avoid repository bloat. All trial metadata and selected classical models are
retained. Neural checkpoints are compact and retained. Paths in decision files
are repository-relative and portable. Load only trusted model checkpoints.

`verification.json` records the completed local checks and `tests.log` contains
the 27-test result. `SHA256SUMS.txt` records the delivered code, data and artifact
hashes. Verify it from the repository root with
`shasum -a 256 -c results/phase2/phase2_v2/SHA256SUMS.txt` on macOS, or
`sha256sum -c results/phase2/phase2_v2/SHA256SUMS.txt` on Linux. Re-evaluation
updates timing/manifest files; check the delivered hashes before regenerating.

## Inference

`src.dl.inference.ResidualPredictor` accepts integer microsecond timestamps and
a dictionary containing the six named IMU axes and two magnitudes. It rejects
nonfinite readings and non-increasing timestamps, automatically resets across
gaps, and exposes `reset()` for a new recording. Its output is a predicted
N/E/D residual in m/s; subtraction, not addition, produces the offline corrected
velocity. It receives no GPS or true target history.

The class exposes a single experimental neural checkpoint. It does not apply
the validation-selected policy or five-model ensemble automatically. Consult
the decision manifest first: a zero gain means leave PX4 unchanged. This is
research software, not a flight-controller integration.

## Interpretation and viva questions

**Why train DL when ML did not win?** To test whether learned short-time sensor
patterns add information beyond fixed rolling statistics. Failure remains an
informative outcome, provided the implementation has positive-control tests.

**Why not random train/test splitting?** Adjacent rows repeat reference values
and overlap in temporal context. Recording separation avoids putting adjacent
examples into opposite partitions, although one session still limits diversity.

**Why can a lower error than RF still be a negative result?** Correction must
improve over leaving the estimator unchanged. A smaller harmful correction can
beat RF while still worsening the original velocity estimate.

**Why not subtract each test log's mean?** That uses unavailable test labels.
The observed vertical shift is a diagnosis, not permission to calibrate on truth.

**Does the model learn physical IMU bias?** No. Its label combines estimator and
reference disagreement; it is not a calibrated sensor-bias measurement.

**Why are uncertainty intervals limited?** Seed repetitions measure optimization
variation; block resampling measures within-recording variability. Neither adds
independent flights or establishes population generalization.

**What would strengthen the project most?** Independent sessions, source timing,
verified aiding configuration and frame information, followed by a genuine
outage experiment with an independent reference.
