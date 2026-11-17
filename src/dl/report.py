"""Generate explanatory scientific figures and a self-contained HTML report."""
import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dl.data import ROOT
from src.utils import TARGET_COLUMNS

LABELS = {"zero": "Zero correction", "train_mean": "Training mean", "ridge": "Retuned Ridge",
          "rf": "Retuned RF", "hgb": "Retuned HGB", "mlp": "MLP (142 features)",
          "raw_mlp": "MLP (8 inputs)", "tcn": "Causal TCN", "gru": "GRU",
          "historical_ridge": "Phase 1 Ridge", "historical_rf": "Phase 1 RF",
          "historical_gbm": "Phase 1 HGB", "validation_selected": "Validation-selected policy"}
MAIN = ["zero", "train_mean", "ridge", "rf", "hgb", "mlp", "raw_mlp", "tcn", "gru"]
COLORS = {"zero": "#153f4f", "train_mean": "#93a2ab", "ridge": "#9c755f", "rf": "#568356",
          "hgb": "#94a85c", "mlp": "#d1883f", "raw_mlp": "#be9b65", "tcn": "#317da6", "gru": "#815a9e"}


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def generate_report(results, decision):
    figures = results / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    test = pd.read_csv(results / "test_metrics.csv").set_index("model")
    val = pd.read_csv(results / "validation_metrics.csv").set_index("model")
    trials = pd.read_csv(results / "neural_trials.csv")
    rows = pd.read_csv(results / "test_rows.csv")
    predictions = np.load(results / "test_predictions.npz")
    means = pd.read_csv(ROOT / "results/eda/tables/per_log_means.csv")
    figure_names = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figure_names.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (name, frame) in zip(axes, [("Validation: log 52", val), ("Historical test: log 54", test)]):
        ordered = frame.loc[MAIN].sort_values("rmse_mean", ascending=False)
        ax.barh([LABELS[n] for n in ordered.index], ordered.rmse_mean,
                color=[COLORS[n] for n in ordered.index])
        ax.axvline(frame.loc["zero", "rmse_mean"], color="#153f4f", linestyle="--", linewidth=1)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_xlabel("Mean of axis RMSE (m/s); lower is better")
        ax.grid(axis="x", alpha=.15)
    save(fig, "benchmark.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([n[:6] for n in means.log_id], means.target_delta_v_d_m_s,
                color=["#317da6", "#d1883f", "#317da6", "#815a9e"])
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_title("Vertical target mean changes across recordings", loc="left")
    axes[0].set_ylabel("Mean Down residual (m/s)")
    compare = test.loc[["zero", "rf", "mlp", "tcn", "gru"]]
    mse = compare.rmse_D ** 2
    bias2 = compare.bias_D ** 2
    axes[1].bar([LABELS[n] for n in compare.index], bias2, label="Squared prediction bias", color="#d1883f")
    axes[1].bar([LABELS[n] for n in compare.index], np.maximum(mse - bias2, 0), bottom=bias2,
                label="Centered error variance", color="#317da6")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].set_title("Down error decomposition on historical test", loc="left")
    axes[1].set_ylabel("MSE ((m/s)²)")
    axes[1].legend(frameon=False)
    save(fig, "vertical_shift.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, axis in zip(axes, "NED"):
        names = ["zero", "rf", "mlp", "tcn", "gru"]
        ax.bar([LABELS[n] for n in names], test.loc[names, f"rmse_{axis}"], color=[COLORS[n] for n in names])
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_title(f"{axis} axis", loc="left", fontweight="bold")
        ax.set_ylabel("RMSE (m/s)")
    save(fig, "axis_errors.png")

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for i, ax in enumerate(axes):
        for _, group in rows.groupby("segment_id"):
            indices = group.index.to_numpy()[::5]
            ax.plot(rows.loc[indices, "time_s"], rows.loc[indices, TARGET_COLUMNS[i]], color="#153f4f", lw=1.2,
                    label="Recorded residual" if group.index[0] == 0 else None)
            for name in ["tcn", "gru"]:
                ax.plot(rows.loc[indices, "time_s"], predictions[name][indices, i], color=COLORS[name],
                        lw=.8, alpha=.8, label=LABELS[name] if group.index[0] == 0 else None)
        ax.axhline(0, color="#777777", ls="--", lw=.7)
        ax.set_ylabel(f"{'NED'[i]} residual (m/s)")
    axes[0].legend(ncol=3, frameon=False)
    axes[0].set_title("Complete historical test recording; lines stop at missing intervals", loc="left")
    axes[-1].set_xlabel("Elapsed recording time (s)")
    save(fig, "test_timeseries.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (family, record) in zip(axes.ravel(), decision["neural"].items()):
        for replica in record["replicas"]:
            history_path = (ROOT / replica["path"]).parent / "training.json"
            history = pd.DataFrame(json.loads(history_path.read_text())["history"])
            ax.plot(history.epoch, history.val_rmse, alpha=.75, label=f"seed {replica['seed']}")
        ax.axhline(val.loc["zero", "rmse_mean"], color="#153f4f", ls="--", label="Zero correction")
        ax.set_title(LABELS[family], loc="left", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation mean-axis RMSE")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    save(fig, "learning_curves.png")

    if (results / "input_sensitivity.csv").exists():
        sensitivity = pd.read_csv(results / "input_sensitivity.csv")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels = [f"{LABELS[r.model]} / {r.group}" for r in sensitivity.itertuples()]
        ax.barh(labels, sensitivity.delta_rmse, color=["#d1883f" if d < 0 else "#317da6" for d in sensitivity.delta_rmse])
        ax.axvline(0, color="black", lw=.8)
        ax.set_xlabel("Validation RMSE change after training-mean replacement (m/s)")
        ax.set_title("Input sensitivity: negative means replacement helps", loc="left")
        save(fig, "input_sensitivity.png")

    summary = test.loc[MAIN + ["validation_selected"],
                       ["rmse_mean", "mae_mean", "r2_mean", "rmse_N", "rmse_E", "rmse_D", "improvement_pct_vs_zero"]].reset_index()
    summary.model = summary.model.map(LABELS)
    neural_best = test.loc[["mlp", "raw_mlp", "tcn", "gru"], "rmse_mean"].idxmin()
    # Descriptive ranking only; the deployable policy was frozen on validation.
    selected = decision["selected_model"]
    difference = float(test.loc["validation_selected", "improvement_pct_vs_zero"])
    outcome = ("Validation selected zero correction." if selected == "zero" or decision["selected_gain"] == 0 else
               f"Validation selected {LABELS.get(selected, selected)} with gain {decision['selected_gain']}.")
    baseline_trials = sum(len(v["trials"]) for v in decision["classical"].values())
    fitted_neural = len(trials) + sum(len(v["replicas"]) - 1 for v in decision["neural"].values())
    text_blocks = [
        ("Finding", f"{outcome} Its historical-test RMSE is {test.loc['validation_selected', 'rmse_mean']:.6f} m/s "
         f"({difference:+.2f}% improvement versus zero correction). The lowest descriptive DL test RMSE belongs to "
         f"{LABELS[neural_best]} ({test.loc[neural_best, 'rmse_mean']:.6f} m/s); this ranking was not used to select the policy."),
        ("What was tested", f"{baseline_trials} classical configurations and {len(trials)} neural configurations were evaluated "
         f"on validation log 52. Selected neural configurations were repeated across {len(decision['identity']['config']['seeds'])} "
         f"fixed seeds ({fitted_neural} neural fits in total). Train logs are 51 and 53. Test log 54 has 8,306 rows and was already "
         "inspected during Phase 1. Neural architecture rows use the mean prediction of all prescribed seeds; individual seed scores remain in the CSV."),
        ("What the prediction means", "The target is recorded PX4 velocity minus recorded GPS velocity. A positive predicted residual "
         "is subtracted from PX4. The corrected error is exactly target minus prediction, so correction RMSE and residual RMSE are "
         "the same calculation on identical rows. No trajectory, filter replay or genuine GNSS outage is evaluated."),
        ("Why learning is difficult", "The Down target mean is positive in both training recordings and negative in validation and test. "
         "This directly establishes distribution shift, but not its physical cause. Missing attitude/heading, source synchronization and "
         "estimator state limit what body-frame IMU readings can tell us about NED residuals. Label autocorrelation does not establish "
         "predictability from IMU history. Model outputs can reproduce training offsets instead of correcting new recordings."),
        ("Validation and leakage safeguards", "Scaling is fit on unique training rows. Input windows contain only present/past IMU samples "
         "within one continuous segment; gaps above 50 ms reset history. Startup uses explicit masks. GPS, PX4 velocity, absolute time and "
         "true residual history are excluded from predictors. HGB random internal validation is disabled; all model selection uses whole log 52. "
         "A decision manifest is written before historical-test scoring."),
        ("Fair comparison and remaining limits", "The feature MLP and retuned classical methods share 142 engineered features. The raw MLP "
         "controls for instantaneous nonlinear modeling. Temporal models test 1, 100, 200 and 400 sample histories and selected dt/Huber variants. "
         "The raw-MLP control has two candidates; main model families have eight. Neural training uses stride-5 endpoints, classical training "
         "uses all rows, and early stopping monitors stride-5 validation endpoints before full-validation selection. Longer neural contexts "
         "exceed the historical feature horizon; those differences cannot be attributed purely to architecture. Five-seed ensembles also "
         "have a greater inference cost than one model. Equal trial counts do not imply equal computational budgets."),
        ("How to read uncertainty", "Per-seed scores describe optimizer variability. Paired block intervals resample contiguous blocks within "
         "segments, using 100/200/400-row block lengths. They are conditional sensitivity intervals for this recording, not population-level "
         "confidence intervals across independent flights. Four logs from one session cannot support a strong deployment claim."),
        ("How to read explanations", "Input sensitivity replaces a sensor group by its training mean in normalized coordinates for the first "
         "fixed seed, on validation only. A negative delta means the replacement improved validation error. This is a perturbation diagnostic, "
         "not causal importance; replacing channels creates unusual inputs and may break relationships between axes and magnitudes. Error "
         "decomposition separates squared bias from centered variance without attributing either to downwash or sensor physics."),
        ("What is complete", "The project includes trained checkpoints, scalers and schemas, bounded searches, replicated neural experiments, "
         "aligned prediction exports, causal/streaming correctness tests, conditional uncertainty analysis and local CPU timing. Original Phase 1 "
         "reports, presentation, models and metrics are preserved. Current-date commits describe actual changes under the configured Git author."),
        ("Next scientific milestone", "Acquire independent recording sessions and recover original ULogs, extraction scripts, source measurement "
         "timestamps, coordinate conventions and fusion configuration. Evaluate a verified GNSS outage while retaining independent reference "
         "measurements for scoring. Until then, this is a complete offline residual benchmark, not a deployed navigation correction system.")
    ]
    md = ["# Phase 2: causal IMU residual learning", "", "## Results", markdown_table(summary), ""]
    for title, body in text_blocks:
        md.extend([f"## {title}", body, ""])
    md.extend(["## Figures", *[f"![{name.removesuffix('.png').replace('_', ' ')}](figures/{name})" for name in figure_names],
               "", "## Reproduce", "From the repository root, install requirements-phase2-lock.txt, then run:",
               "```bash", "python -m src.dl.experiment --stage train", "python -m src.dl.experiment --stage evaluate",
               "python -m src.dl.explain", "python -m src.dl.report", "```",
               "Existing evaluated runs cannot be retuned in place. Use a new config name for a different experiment.",
               "", "## Research context", "- Bai et al., temporal convolutional networks: https://arxiv.org/abs/1803.01271",
               "- Cioffi et al., drone inertial odometry (uses thrust unavailable here): https://arxiv.org/abs/2210.15287"])
    (results / "REPORT.md").write_text("\n\n".join(md) + "\n")
    sections = "".join(f"<section><h2>{html.escape(title)}</h2><p>{html.escape(body)}</p></section>" for title, body in text_blocks)
    images = "".join(f'<figure><img alt="{html.escape(name)}" src="data:image/png;base64,{base64.b64encode((figures / name).read_bytes()).decode()}"></figure>' for name in figure_names)
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 2 · IMU residual benchmark</title><style>
body{{margin:0;background:#f5f3ed;color:#183340;font:17px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:48px 28px}}
header{{border-top:6px solid #317da6;padding:28px 0}}.eyebrow{{font-size:13px;letter-spacing:.13em;text-transform:uppercase;color:#65747c}}
h1{{font-size:44px;line-height:1.15;max-width:850px;margin:18px 0}}h2{{font-size:25px;line-height:1.25;margin-top:32px}}p{{max-width:950px}}
.lead{{font-size:22px;max-width:920px}}section{{border-top:1px solid #ccd3d3;margin-top:24px;padding-top:6px}}
table{{border-collapse:collapse;font-size:13px;width:100%;background:white}}th,td{{padding:11px 9px;text-align:right;border-bottom:1px solid #dce1e1}}th:first-child,td:first-child{{text-align:left}}th{{background:#153f4f;color:white}}
.table{{overflow-x:auto}}figure{{margin:30px 0;background:white;padding:12px}}img{{width:100%;height:auto}}footer{{font-size:13px;color:#65747c;margin-top:40px}}
@media(max-width:650px){{h1{{font-size:32px}}main{{padding:24px 16px}}body{{font-size:16px}}}}@media print{{body{{background:white}}figure,section{{break-inside:avoid}}}}
</style><main><header><div class="eyebrow">Phase 2 / reproducible research</div><h1>Can causal IMU history improve velocity residual correction?</h1>
<p class="lead">{html.escape(text_blocks[0][1])}</p><p>Recorded PX4–GPS disagreement · 4 recording groups · 3 residual axes</p></header>
<div class="table">{summary.to_html(index=False, float_format=lambda x: f'{x:.5f}', border=0)}</div>{sections}{images}
<footer>Source: repository data and frozen Phase 2 artifacts. Figures show observed experiments; no simulated performance is presented as real.</footer></main></html>'''
    (results / "REPORT.html").write_text(content)
    summary.to_csv(results / "summary.csv", index=False)
    return results / "REPORT.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="phase2_v2")
    args = parser.parse_args()
    results = ROOT / "results/phase2" / args.run
    decision = json.loads((ROOT / "saved_models/phase2" / args.run / "decision.json").read_text())
    print(generate_report(results, decision))


if __name__ == "__main__":
    main()
