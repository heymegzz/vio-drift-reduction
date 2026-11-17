# Phase 2 experimental protocol

This protocol is fixed before Phase 2 model fitting. Phase 1 commit
`98901938c46d107f09e973c09d673b613414b7bf` and its saved results are historical
artifacts. Its report and presentation must remain unchanged.

## Question and scope

Can causal IMU history improve prediction of recorded PX4–GPS velocity
residuals over zero correction and classical regressors? This is an offline
residual benchmark, not validated GNSS-denied navigation. GPS fusion state,
source alignment and recording independence remain unverified. No trajectory
accuracy or onboard readiness claim follows from this experiment.

## Fixed data contract

- Train: logs 51 and 53. Validation: log 52. Historical evaluation: log 54.
- The evaluation recording was inspected in Phase 1; it is not pristine.
- Targets remain PX4 velocity minus GPS velocity in N/E/D order, in m/s.
- Inputs: six IMU axes and two magnitudes; optional actual dt. Absolute time,
  log/segment identifiers, GPS, PX4 velocity and past true targets are forbidden.
- Split before windowing. Reset at gaps strictly greater than 50,000 us.
- Reject duplicate/non-increasing timestamps, unexpected logs and nonfinite data.
- Normalize using unique training rows only. Never center using holdout labels.
- Windows end at the prediction timestamp. Startup uses normalized zero padding
  and an explicit validity channel. GRU receives only valid observations.
- Primary evaluation includes every row. Full-history subsets use a common mask
  for all models. No interpolation across missing intervals.

## Selection

Primary metric: arithmetic mean of N/E/D RMSE in m/s. Also report each axis,
MAE, R² and prediction bias (prediction minus target). Use the same primary
metric to select all classical and neural configurations.

Zero correction and training-mean predictions are mandatory baselines. Preserve
the three Phase 1 predictions as historical comparators. Retuned Ridge, RF and
HGB are separate models. HGB disables random internal early stopping; maximum
iterations are explicit validation-selected hyperparameters.

Neural candidates: MLP on 142 features, current-row MLP on eight inputs, causal
TCN, and GRU. Use bounded configuration lists, modest capacity and AdamW.
Early stopping uses a fixed, uniformly spaced validation endpoint subset for
efficiency; final configuration selection uses all validation rows. Training
endpoint subsampling is explicit. It does not create independent examples.
All architecture/configuration decisions precede test evaluation.

The output head starts at zero. Model checkpoints are selected from trained
epochs (epoch >= 1), keeping failed learned corrections visible in the results.
Zero correction competes separately in the final validation decision. A setup
smoke run revealed sklearn 1.9.0 in the historical checkpoints; the full run
uses that matching version, and environment versions are part of run identity.

Five predetermined seeds (11, 23, 37, 51, 71) quantify optimization variation for
selected neural configurations. Average predictions across those seeds for the
declared ensemble, while also reporting every seed. Never select a test-best seed.

Optional global gains are selected from {0, .25, .5, .75, 1} using validation
only, equally for ML and DL. Report ungated and gain-adjusted values. A selected
zero gain means no demonstrated benefit from that correction. Freeze a decision
manifest before scoring test targets in the evaluation stage.

## Evidence and stopping

An improvement over RF alone is insufficient: useful correction must beat zero.
Report all results, including failures and unfavorable axes. Paired contiguous
block bootstrap intervals are conditional on the observed recording; they are
not independent-flight confidence intervals. Report multiple block lengths.
Do not change the protocol in response to historical test outcomes.

Use CPU float32 with recorded thread count and deterministic settings. Save
checksums, software versions, seeds, trial histories, validation decisions,
aligned predictions, checkpoints and complete-run timings. Measure streaming
preprocessing plus inference separately from batch throughput.

Additional independent sessions with verified frames, source timestamps and
outage/fusion configuration are required for confirmatory navigation claims.

## Contribution record

Commits record actual work dates and the configured repository author. Three
balanced review areas are data/baselines, neural models/training, and evaluation/
documentation. These are review assignments, not claims of completed human work.
