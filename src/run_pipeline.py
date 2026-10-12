"""End-to-end pipeline orchestrator (roadmap Section 7, 14).

Runs generate -> split -> train (Pure-AI + residual-AI, 5 seeds) ->
calibrate (C0-C5) -> evaluate (E1-E7) -> statistics -> figures, and writes
``environment.json`` (Debian release, kernel, Python version, CPU,
package lock, source revision, seeds, exact commands -- Section 7).

``--mode quick`` shrinks dataset sizes (configs/data_generation.yaml
quick_divisor) and training epochs for a fast smoke run; ``--mode full``
uses every value in configs/*.yaml exactly as frozen. Neither mode edits
a frozen scientific parameter -- quick mode only reduces *how much* data
is generated and *how long* training runs, never thresholds, bounds, or
the primary endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from generate_dataset import generate_all
from physics_model import load_engine_config
import prepare_splits as prepare_splits_mod
import train_pure_ai
import train_residual_ai
from online_calibration import run_all_modes
import evaluate as evaluate_mod
import statistical_analysis
import make_publication_figures


def _cmd_output(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unavailable"


def write_environment_json(out_path: str, extra: dict = None):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or _cmd_output(["uname", "-p"]),
        "os_release": _cmd_output(["cat", "/etc/os-release"]),
        "kernel": _cmd_output(["uname", "-r"]),
        "cpu_info": _cmd_output(["nproc"]),
        "source_revision": _cmd_output(["git", "rev-parse", "HEAD"]),
        "pip_freeze": _cmd_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload


def run(mode: str, refreeze: bool = False):
    quick = mode == "quick"

    with open("configs/data_generation.yaml") as f:
        data_gen_cfg = yaml.safe_load(f)
    with open("configs/sensors.yaml") as f:
        sensors_cfg = yaml.safe_load(f)
    with open("configs/model_pure_ai.yaml") as f:
        pure_cfg = yaml.safe_load(f)
    with open("configs/model_residual_ai.yaml") as f:
        residual_cfg = yaml.safe_load(f)
    with open("configs/calibration.yaml") as f:
        calib_cfg = yaml.safe_load(f)
    with open("configs/experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    engine_cfg = load_engine_config("configs/engine_m30b35.yaml")

    if quick:
        for arch_cfg in (pure_cfg["architecture"], residual_cfg["architecture"]):
            arch_cfg["max_iter"] = min(arch_cfg["max_iter"], 40)
            arch_cfg["n_iter_no_change"] = min(arch_cfg["n_iter_no_change"], 8)

    results_path = "results/test_metrics_all_models.csv"
    if os.path.exists(results_path) and not refreeze and mode == "full":
        print(
            "results/test_metrics_all_models.csv already exists. Per the "
            "Section-13 acceptance gate (no post-hoc threshold/result "
            "changes), re-running the full-mode final-test stage requires "
            "--allow-refreeze. Delete results/ manually or pass "
            "--allow-refreeze to proceed deliberately."
        )
        return

    print(f"== [1/7] generate_dataset (mode={mode}) ==")
    generate_all(data_gen_cfg, engine_cfg, sensors_cfg, out_dir="data/raw", quick=quick)

    print("== [2/7] prepare_splits ==")
    import pandas as pd
    raw_manifest = pd.read_csv("data/raw/raw_manifest.csv")
    manifest = prepare_splits_mod.build_splits(raw_manifest)
    os.makedirs("data/splits", exist_ok=True)
    manifest.to_csv("data/splits/splits.csv", index=False)
    prepare_splits_mod.write_manifest(manifest, data_gen_cfg, "results/dataset_manifest.json")

    print("== [3/7] train_pure_ai + train_residual_ai ==")
    features = pure_cfg["model"]["features"]
    train_pure_ai.assert_no_leakage(features)
    train_residual_ai.assert_no_leakage(residual_cfg["model"]["features"])

    os.makedirs("models", exist_ok=True)
    train_df = train_pure_ai.load_split("data/splits/splits.csv", pure_cfg["training"]["train_split_roles"])
    val_df = train_pure_ai.load_split("data/splits/splits.csv", pure_cfg["training"]["val_split_roles"])
    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df[pure_cfg["model"]["target"]].to_numpy(dtype=float)
    X_val = val_df[features].to_numpy(dtype=float)
    y_val = val_df[pure_cfg["model"]["target"]].to_numpy(dtype=float)

    import joblib
    import sklearn
    for seed in pure_cfg["training"]["seeds"]:
        model, scaler, val_mae, n_epochs = train_pure_ai.train_one_seed(
            X_train, y_train, X_val, y_val, pure_cfg["architecture"], seed
        )
        joblib.dump({"model": model, "scaler": scaler, "features": features,
                     "target": pure_cfg["model"]["target"], "seed": seed,
                     "val_mae_g_s": val_mae, "n_epochs": n_epochs,
                     "n_parameters": train_pure_ai.count_params(model),
                     "scikit_learn_version": sklearn.__version__},
                    f"models/pure_ai_seed{seed}.joblib")
        print(f"  pure_ai seed={seed} val_MAE={val_mae:.3f} epochs={n_epochs}")

    from physics_model import PhysicsModel
    physics = PhysicsModel.from_config(engine_cfg)
    for df in (train_df, val_df):
        df["_physics_anchor"] = physics.airflow_g_s(df["rpm"].to_numpy(), df["map_kpa"].to_numpy(), df["iat_c"].to_numpy())
    clip_lo, clip_hi = residual_cfg["training"]["residual_target_clip_g_s"]
    import numpy as np
    res_train = np.clip(train_df[residual_cfg["model"]["target"]].to_numpy() - train_df["_physics_anchor"].to_numpy(), clip_lo, clip_hi)
    res_val = np.clip(val_df[residual_cfg["model"]["target"]].to_numpy() - val_df["_physics_anchor"].to_numpy(), clip_lo, clip_hi)
    residual_features = residual_cfg["model"]["features"]
    X_res_train = train_df[residual_features].to_numpy(dtype=float)
    X_res_val = val_df[residual_features].to_numpy(dtype=float)
    for seed in residual_cfg["training"]["seeds"]:
        model, scaler, val_mae, n_epochs = train_residual_ai.train_one_seed(
            X_res_train, res_train, X_res_val, res_val, residual_cfg["architecture"], seed
        )
        joblib.dump({"model": model, "scaler": scaler, "features": residual_features,
                     "target": residual_cfg["model"]["target"], "correction_form": residual_cfg["model"]["correction_form"],
                     "engine_config_path": "configs/engine_m30b35.yaml", "seed": seed,
                     "val_residual_mae_g_s": val_mae, "n_epochs": n_epochs,
                     "n_parameters": train_pure_ai.count_params(model),
                     "scikit_learn_version": sklearn.__version__},
                    f"models/residual_ai_seed{seed}.joblib")
        print(f"  residual_ai seed={seed} val_residual_MAE={val_mae:.3f} epochs={n_epochs}")

    os.makedirs("results", exist_ok=True)

    print("== [4/7] evaluate E1-E4 ==")
    pure_models = evaluate_mod.load_pure_ai_models("models", exp_cfg["seeds"])
    residual_models = evaluate_mod.load_residual_ai_models("models", exp_cfg["seeds"])
    value_range = exp_cfg["primary_endpoint"]["normalization_airflow_range_g_s"]
    evaluate_mod.evaluate_e1_e4(
        "data/splits/splits.csv", engine_cfg, pure_models, residual_models,
        "results", value_range,
    )

    print("== [5/7] calibrate C0-C5 + evaluate E5-E7 ==")
    evaluate_mod.evaluate_e5_e7(calib_cfg, exp_cfg, "data/splits/splits.csv", "models",
                                 "configs/engine_m30b35.yaml", "results")

    print("== [6/7] statistical_analysis ==")
    stats_cfg = exp_cfg["statistics"]
    comparisons = [tuple(c) for c in stats_cfg["comparisons"]]
    model_comparisons = [c for c in comparisons if c[0] in ("residual_ai", "pure_ai", "physics_only")]
    calib_comparisons = [c for c in comparisons if c[0].startswith("C")]
    rows = []
    per_run_path = "results/per_run_model_metrics.csv"
    if os.path.exists(per_run_path):
        rows.extend(statistical_analysis.build_model_comparisons(pd.read_csv(per_run_path), model_comparisons))
    drift_path = "results/drift_recovery_metrics.csv"
    if os.path.exists(drift_path):
        rows.extend(statistical_analysis.build_calibration_comparisons(pd.read_csv(drift_path), calib_comparisons))
    result_df = pd.DataFrame(rows)
    if len(result_df):
        result_df["p_value_holm_adjusted"] = statistical_analysis.holm_correction(
            result_df["p_value"].fillna(1.0).to_numpy()
        )
    result_df.to_csv("results/statistical_tests.csv", index=False)

    print("== [7/7] figures ==")
    make_publication_figures.fig_architecture("results/figures")
    make_publication_figures.fig_data_workflow("results/figures")
    make_publication_figures.fig_global_model_comparison("results", "results/figures")
    make_publication_figures.fig_boundary_corner("results", "results/figures")
    make_publication_figures.fig_transient_response("results", "results/figures")
    make_publication_figures.fig_calibration_ablation("results", "results/figures")
    make_publication_figures.fig_data_efficiency("results", "results/figures")
    make_publication_figures.fig_mismatch_crossover("results", "results/figures")
    make_publication_figures.fig_legacy_output("results", "results/figures")

    write_environment_json("results/environment.json", extra={"mode": mode, "seeds": exp_cfg["seeds"]})
    print("Pipeline complete. See results/ for all required software outputs.")


def main():
    parser = argparse.ArgumentParser(description="Run the full M30 virtual-MAF pipeline.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--stage", choices=["environment"], default=None,
                         help="Run only a single named stage (currently: environment).")
    parser.add_argument("--allow-refreeze", action="store_true",
                         help="Explicitly allow re-running the full-mode final-test stage over "
                              "existing results/ (see Section 13 acceptance gate).")
    args = parser.parse_args()

    if args.stage == "environment":
        write_environment_json("results/environment.json")
        return

    run(args.mode, refreeze=args.allow_refreeze)


if __name__ == "__main__":
    main()
