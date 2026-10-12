"""The nine essential figures (roadmap Section 15), each built from a
stored CSV/parquet result file, with the plotted data saved beside the
figure (Section 14: "Every figure should be generated from stored result
files, and the plotted data should be stored beside the figure.").
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def _save(fig, out_dir, name, data_df=None):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close(fig)
    if data_df is not None:
        data_df.to_csv(os.path.join(out_dir, f"{name}_data.csv"), index=False)


def fig_architecture(out_dir):
    """Schematic block diagram (Section 15: "Architecture")."""
    fig, ax = plt.subplots(figsize=(9, 4))
    boxes = [
        ("Direct sensors\n(RPM,MAP,TPS,IAT,ECT)", 0.02, 0.55),
        ("Speed-density\nbaseline", 0.22, 0.75),
        ("Residual AI\n(16-16-8 MLP)", 0.22, 0.35),
        ("Bounded scale s\n(calibration)", 0.44, 0.55),
        ("Conditioned lambda /\nfuel feedback", 0.44, 0.10),
        ("Scalar Fisher\ninfo gate", 0.64, 0.55),
        ("Causal accept /\nrollback", 0.80, 0.55),
        ("Legacy voltage /\nfrequency output", 0.80, 0.90),
    ]
    for text, x, y in boxes:
        ax.add_patch(plt.Rectangle((x, y - 0.08), 0.16, 0.16, fill=False, edgecolor="black"))
        ax.text(x + 0.08, y, text, ha="center", va="center", fontsize=7.5)
    arrows = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (3, 5), (5, 6), (6, 3), (1, 7)]
    for a, b in arrows:
        xa, ya = boxes[a][1] + 0.16, boxes[a][2]
        xb, yb = boxes[b][1], boxes[b][2]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="->", lw=1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("System architecture (Section 15)")
    _save(fig, out_dir, "fig01_architecture")


def fig_data_workflow(out_dir):
    fig, ax = plt.subplots(figsize=(8, 3))
    stages = ["Hidden plant\n(virtual_engine.py)", "Sensor corruption\n(sensor_models.py)",
              "Run-level split\n(prepare_splits.py)", "Evaluation path\n(evaluate.py)"]
    for i, s in enumerate(stages):
        ax.add_patch(plt.Rectangle((i * 0.26, 0.3), 0.2, 0.4, fill=False))
        ax.text(i * 0.26 + 0.1, 0.5, s, ha="center", va="center", fontsize=8)
        if i > 0:
            ax.annotate("", xy=(i * 0.26, 0.5), xytext=((i - 1) * 0.26 + 0.2, 0.5),
                        arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Virtual data workflow (Section 15)")
    _save(fig, out_dir, "fig02_data_workflow")


def fig_global_model_comparison(results_dir, out_dir):
    path = os.path.join(results_dir, "test_metrics_all_models.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    sub = df[(df.metric == "nrmse") & (df.scenario == "nominal_test")].copy()
    sub["family"] = sub["model"].str.replace(r"_seed\d+", "", regex=True)
    agg = sub.groupby("family").agg(mean=("mean", "mean"), lo=("ci_lo", "mean"), hi=("ci_hi", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(agg["family"], agg["mean"], yerr=[agg["mean"] - agg["lo"], agg["hi"] - agg["mean"]], capsize=4)
    ax.set_ylabel("NRMSE (nominal_test)")
    ax.set_title("Global model comparison (E1)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, out_dir, "fig03_global_model_comparison", agg)


def fig_boundary_corner(results_dir, out_dir):
    path = os.path.join(results_dir, "boundary_corner_metrics.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    sub = df[df.model.str.startswith("residual_ai") | (df.model == "physics_only")]
    agg = sub.groupby(["scenario", "model"])["nrmse"].mean().reset_index()
    piv = agg.pivot(index="scenario", columns="model", values="nrmse")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("NRMSE")
    ax.set_title("Boundary and corner robustness (E2)")
    _save(fig, out_dir, "fig04_boundary_corner", agg)


def fig_transient_response(results_dir, out_dir):
    path = os.path.join(results_dir, "test_predictions.parquet")
    if not os.path.exists(path):
        return
    df = pd.read_parquet(path)
    sub = df[df.scenario == "transient_unseen"]
    if len(sub) == 0:
        return
    run_id = sub.run_id.iloc[0]
    sub = sub[sub.run_id == run_id]
    fig, ax = plt.subplots(figsize=(8, 4))
    ref = sub[sub.model == sub.model.iloc[0]][["t_s", "ref_g_s"]].drop_duplicates().sort_values("t_s")
    ax.plot(ref.t_s, ref.ref_g_s, label="hidden reference", color="black", lw=1.5)
    for model_name in [m for m in sub.model.unique() if "seed1" in m or m == "physics_only"]:
        s = sub[sub.model == model_name].sort_values("t_s")
        ax.plot(s.t_s, s.pred_g_s, label=model_name, alpha=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("airflow (g/s)")
    ax.set_title(f"Transient response (E3) -- {run_id}")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "fig05_transient_response", sub)


def fig_calibration_ablation(results_dir, out_dir):
    events_path = os.path.join(results_dir, "calibration_events.csv")
    drift_path = os.path.join(results_dir, "drift_recovery_metrics.csv")
    if not (os.path.exists(events_path) and os.path.exists(drift_path)):
        return
    events = pd.read_csv(events_path)
    drift = pd.read_csv(drift_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    holdout_agg = drift.groupby("mode")["holdout_nrmse"].mean().sort_index()
    axes[0, 0].bar(holdout_agg.index, holdout_agg.values)
    axes[0, 0].set_title("Holdout NRMSE by mode")

    false_rate = events.groupby("mode")["false_update"].mean() if "false_update" in events else pd.Series(dtype=float)
    axes[0, 1].bar(false_rate.index, false_rate.values)
    axes[0, 1].set_title("False-update rate by mode")

    if "scalar_fisher_score" in events.columns:
        c3 = events[(events["mode"] == "C3") & events["scalar_fisher_score"].notna()]
        axes[1, 0].scatter(range(len(c3)), c3["scalar_fisher_score"], s=10)
        if len(c3):
            axes[1, 0].axhline(c3["scalar_fisher_threshold"].iloc[0], color="red", ls="--", label="I_s,min")
        axes[1, 0].set_title("C3 scalar Fisher score vs threshold")
        axes[1, 0].legend(fontsize=7)

    rollback_counts = drift.groupby("mode")["n_rollback_events"].sum()
    axes[1, 1].bar(rollback_counts.index, rollback_counts.values)
    axes[1, 1].set_title("Rollback events by mode")

    fig.suptitle("Calibration ablation C0-C5 (E5)")
    _save(fig, out_dir, "fig06_calibration_ablation", drift)


def fig_data_efficiency(results_dir, out_dir):
    path = os.path.join(results_dir, "calibration_data_efficiency.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["max_accepted_valid_samples"], df["median_holdout_nrmse"], marker="o")
    ax.fill_between(df["max_accepted_valid_samples"], df["ci_lo"], df["ci_hi"], alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("cumulative Stage-A-valid samples available")
    ax.set_ylabel("median holdout NRMSE")
    ax.set_title("Informative-data efficiency (E6)")
    _save(fig, out_dir, "fig07_data_efficiency", df)


def fig_mismatch_crossover(results_dir, out_dir):
    path = os.path.join(results_dir, "mismatch_crossover.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    sub = df[df.model.str.startswith("residual_ai") | (df.model == "physics_only") | df.model.str.startswith("pure_ai")]
    agg = sub.groupby(["scenario", sub.model.str.replace(r"_seed\d+", "", regex=True)])["nrmse_mean"].mean()
    agg = agg.reset_index().rename(columns={"model": "family"})
    piv = agg.pivot(index="scenario", columns="family", values="nrmse_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot(marker="o", ax=ax)
    ax.set_ylabel("NRMSE")
    ax.set_title("Plant mismatch crossover (E4)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, out_dir, "fig08_mismatch_crossover", agg)


def fig_legacy_output(results_dir, out_dir, profile_cfg_path="configs/output_profiles.yaml"):
    import yaml
    from legacy_output import LegacyOutputProfile

    with open(profile_cfg_path) as f:
        out_cfg = yaml.safe_load(f)
    profile = LegacyOutputProfile(out_cfg["voltage_profile"])
    airflow = np.linspace(0, 220, 300)
    voltage = profile.transfer(airflow)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(airflow, voltage, marker=".", ms=3)
    ax.set_xlabel("airflow (g/s)")
    ax.set_ylabel("legacy voltage output (V)")
    ax.set_title("Legacy output profile (software emulation only)")
    _save(fig, out_dir, "fig09_legacy_output", pd.DataFrame({"airflow_g_s": airflow, "voltage_v": voltage}))


def main():
    parser = argparse.ArgumentParser(description="Generate the nine essential publication figures.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fig_architecture(args.out_dir)
    fig_data_workflow(args.out_dir)
    fig_global_model_comparison(args.results_dir, args.out_dir)
    fig_boundary_corner(args.results_dir, args.out_dir)
    fig_transient_response(args.results_dir, args.out_dir)
    fig_calibration_ablation(args.results_dir, args.out_dir)
    fig_data_efficiency(args.results_dir, args.out_dir)
    fig_mismatch_crossover(args.results_dir, args.out_dir)
    fig_legacy_output(args.results_dir, args.out_dir)

    print(f"Figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
