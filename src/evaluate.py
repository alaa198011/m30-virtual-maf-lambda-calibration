"""Evaluation for model accuracy, calibration, data efficiency, and output shaping."""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml

from legacy_output import LegacyOutputProfile, evaluate_profile, host_timing_inference, model_resource_footprint
from online_calibration import align_phi_to_pseudo_reference, run_all_modes
from physics_model import PhysicsModel, load_engine_config
from train_residual_ai import predict_residual_ai
from uncertainty import bootstrap_ci

TRANSIENT_STATES = ["idle", "tip_in", "acceleration", "cruise", "high_load", "tip_out", "deceleration"]


def core_metrics(pred: np.ndarray, ref: np.ndarray, value_range: float) -> dict:
    pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    finite = np.isfinite(pred) & np.isfinite(ref)
    if not finite.any():
        return {k: float("nan") for k in
                ("mae_g_s", "rmse_g_s", "nrmse", "signed_bias_g_s", "p95_abs_error_g_s")}
    err = pred[finite] - ref[finite]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "mae_g_s": float(np.mean(np.abs(err))),
        "rmse_g_s": rmse,
        "nrmse": rmse / max(float(value_range), 1e-9),
        "signed_bias_g_s": float(np.mean(err)),
        "p95_abs_error_g_s": float(np.percentile(np.abs(err), 95)),
    }


def load_pure_ai_models(models_dir, seeds):
    return {s: joblib.load(os.path.join(models_dir, f"pure_ai_seed{s}.joblib")) for s in seeds}


def load_residual_ai_models(models_dir, seeds):
    return {s: joblib.load(os.path.join(models_dir, f"residual_ai_seed{s}.joblib")) for s in seeds}


def predict_pure_ai(artifact, df):
    X = df[artifact["features"]].to_numpy(dtype=float)
    return artifact["model"].predict(artifact["scaler"].transform(X))


def _lag_seconds(pred, ref, dt_s=0.02, max_lag_s=1.0):
    pred, ref = np.asarray(pred), np.asarray(ref)
    if len(pred) < 4 or np.std(pred) < 1e-12 or np.std(ref) < 1e-12:
        return 0.0
    p, r = pred - np.mean(pred), ref - np.mean(ref)
    max_lag = min(int(round(max_lag_s / dt_s)), len(pred) // 2)
    lags = range(-max_lag, max_lag + 1)
    scores = []
    for lag in lags:
        if lag < 0:
            scores.append(np.dot(p[-lag:], r[:lag]))
        elif lag > 0:
            scores.append(np.dot(p[:-lag], r[lag:]))
        else:
            scores.append(np.dot(p, r))
    return float(list(lags)[int(np.argmax(scores))] * dt_s)


def _rise_time(values, dt_s=0.02):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.ptp(values) < 1e-9:
        return 0.0
    lo, hi = np.min(values) + 0.1 * np.ptp(values), np.min(values) + 0.9 * np.ptp(values)
    i10 = np.where(values >= lo)[0]
    i90 = np.where(values >= hi)[0]
    if not len(i10) or not len(i90) or i90[0] < i10[0]:
        return float("nan")
    return float((i90[0] - i10[0]) * dt_s)


def evaluate_e1_e4(splits_csv, engine_cfg, pure_models, residual_models,
                    results_dir, value_range):
    splits = pd.read_csv(splits_csv)
    per_run_rows, pred_rows, transient_rows = [], [], []
    test_scenarios = splits.loc[splits.split_role == "test", "scenario"].unique().tolist()

    for _, run in splits[splits.split_role == "test"].iterrows():
        df = pd.read_parquet(run["path"])
        ref = df["maf_reference_g_s"].to_numpy(dtype=float)
        model_preds = {"physics_only": df["maf_physics_g_s"].to_numpy(dtype=float)}
        for seed, art in pure_models.items():
            model_preds[f"pure_ai_seed{seed}"] = predict_pure_ai(art, df)
        for seed, art in residual_models.items():
            model_preds[f"residual_ai_seed{seed}"], _ = predict_residual_ai(art, df, engine_cfg)

        for name, pred in model_preds.items():
            row = core_metrics(pred, ref, value_range)
            row.update({"run_id": run.run_id, "scenario": run.scenario,
                        "kind": run.kind, "mismatch": run.mismatch, "model": name})
            per_run_rows.append(row)
            step = max(1, len(df) // 50)
            for idx in range(0, len(df), step):
                pred_rows.append({"run_id": run.run_id, "scenario": run.scenario,
                                  "model": name, "t_s": float(df.timestamp_s.iloc[idx]),
                                  "pred_g_s": float(pred[idx]), "ref_g_s": float(ref[idx])})

        if run.scenario == "transient_unseen":
            for state in TRANSIENT_STATES:
                mask = (df.state == state).to_numpy()
                if not mask.any():
                    continue
                for name, pred in model_preds.items():
                    e = pred[mask] - ref[mask]
                    transient_rows.append({
                        "run_id": run.run_id, "state": state, "model": name,
                        "integrated_abs_error_g": float(np.sum(np.abs(e)) * 0.02),
                        "mae_g_s": float(np.mean(np.abs(e))),
                        "peak_error_g_s": float(np.max(np.abs(e))),
                        "lag_s": _lag_seconds(pred[mask], ref[mask]),
                        "rise_time_error_s": _rise_time(pred[mask]) - _rise_time(ref[mask]),
                        "n_samples": int(mask.sum()),
                    })

    per_run = pd.DataFrame(per_run_rows)
    per_run.to_csv(os.path.join(results_dir, "per_run_model_metrics.csv"), index=False)
    agg = []
    for (name, scenario), g in per_run.groupby(["model", "scenario"]):
        for metric in ["mae_g_s", "nrmse", "signed_bias_g_s", "p95_abs_error_g_s"]:
            mean, lo, hi = bootstrap_ci(g[metric].to_numpy(), n_boot=2000)
            agg.append({"model": name, "scenario": scenario, "metric": metric,
                        "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_runs": len(g)})
    pd.DataFrame(agg).to_csv(os.path.join(results_dir, "test_metrics_all_models.csv"), index=False)
    pd.DataFrame(pred_rows).to_parquet(os.path.join(results_dir, "test_predictions.parquet"), index=False)
    pd.DataFrame(transient_rows).to_csv(os.path.join(results_dir, "transient_state_metrics.csv"), index=False)

    mismatch = [s for s in test_scenarios if s.startswith("mismatch_") or s == "nominal_test"]
    cross = []
    for (name, scenario), g in per_run[per_run.scenario.isin(mismatch)].groupby(["model", "scenario"]):
        cross.append({"model": name, "scenario": scenario,
                      "nrmse_mean": float(g.nrmse.mean()),
                      "mae_mean_g_s": float(g.mae_g_s.mean()), "n_runs": len(g)})
    pd.DataFrame(cross).to_csv(os.path.join(results_dir, "mismatch_crossover.csv"), index=False)
    per_run[per_run.scenario.isin(["boundary", "corner"])].to_csv(
        os.path.join(results_dir, "boundary_corner_metrics.csv"), index=False
    )
    return per_run


def _safe_event_counts(events, run_id, mode, seed):
    if len(events) == 0:
        return pd.DataFrame()
    return events[(events.run_id == run_id) & (events["mode"] == mode) &
                  (events.model_seed == seed)]


def _pseudo_r2(df):
    mask = df["calibration_allowed"].to_numpy(dtype=bool)
    if mask.sum() < 2:
        return float("nan")
    hidden_aligned = align_phi_to_pseudo_reference(
        df, df["maf_reference_g_s"].to_numpy(dtype=float)
    )
    y = hidden_aligned[mask]
    p = df.loc[mask, "pseudo_reference_g_s"].to_numpy()
    denom = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / denom) if denom > 1e-12 else float("nan")


def scale_at_valid_sample_budget(payload, df, budget):
    """Return the committed scale after at most ``budget`` valid learning samples."""
    if budget < 0:
        raise ValueError("sample budget must be non-negative")
    active = np.where(df.state.to_numpy() == "adaptation")[0]
    eligible = active[payload["valid_count_by_index"][active] <= budget]
    return float(payload["committed_s"][eligible[-1]]) if len(eligible) else 1.0


def build_acceptance_report(exp_cfg, holdout_df, events_df, out_path):
    primary = exp_cfg["primary_endpoint"]
    final = holdout_df[holdout_df.run_id.str.startswith("calibration_shift_test")]
    family = final.groupby(["run_id", "mode"]).holdout_nrmse.mean().unstack()
    c3_mean = float(family["C3"].mean()) if "C3" in family else float("nan")
    c0_mean = float(family["C0"].mean()) if "C0" in family else float("nan")
    improvement = (c0_mean - c3_mean) / max(c0_mean, 1e-12)
    bounds_ok = bool((holdout_df.bound_violations == 0).all())

    def false_rate(mode):
        g = events_df[(events_df["mode"] == mode) & (events_df.decision == "accept")]
        return float(g.false_update.mean()) if len(g) else 0.0

    c3_false, c1_false = false_rate("C3"), false_rate("C1")
    checks = {
        "absolute_target": bool(c3_mean <= primary["absolute_target"]),
        "relative_improvement_over_C0": bool(improvement >= primary["min_improvement_margin_over_C0"]),
        "zero_bound_violations": bool(bounds_ok),
        "lower_false_update_rate_than_C1": bool(c3_false < c1_false),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed": {"C3_mean_holdout_nrmse": c3_mean, "C0_mean_holdout_nrmse": c0_mean,
                     "relative_improvement_over_C0": improvement,
                     "C3_false_update_rate": c3_false, "C1_false_update_rate": c1_false},
        "thresholds": primary,
        "note": "Quick-mode reports are diagnostic; publication claims require full mode.",
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


def evaluate_e5_e7(calib_cfg, exp_cfg, splits_csv, models_dir,
                    engine_cfg_path, results_dir):
    value_range = exp_cfg["primary_endpoint"]["normalization_airflow_range_g_s"]
    events, fisher, payloads, trajectory = run_all_modes(
        splits_csv, models_dir, calib_cfg, engine_cfg_path,
        scenarios=("calibration_shift_threshold", "calibration_shift_test"),
        model_seeds=tuple(exp_cfg["seeds"]),
    )
    events.to_csv(os.path.join(results_dir, "calibration_events.csv"), index=False)
    fisher.to_csv(os.path.join(results_dir, "scalar_fisher_trace.csv"), index=False)
    trajectory.to_csv(os.path.join(results_dir, "parameter_trajectory.csv"), index=False)

    holdout_rows = []
    for (run_id, mode, seed), payload in payloads.items():
        df, phi, final_s = payload["df"], payload["phi_all"], payload["final_s"]
        holdout = (df.state == "holdout").to_numpy()
        if not holdout.any():
            continue
        m = core_metrics(final_s * phi[holdout], df.maf_reference_g_s.to_numpy()[holdout], value_range)
        ev = _safe_event_counts(events, run_id, mode, seed)
        accepted = ev[ev.decision == "accept"] if len(ev) else ev
        adaptation_start = float(df.loc[df.state == "adaptation", "timestamp_s"].min())
        post_drift = accepted[accepted.t_acceptance_end >= adaptation_start] if len(accepted) else accepted
        adaptation_time = (float(post_drift.t_acceptance_end.iloc[0]) - adaptation_start
                           if len(post_drift) else float("nan"))
        accepted_samples = int((accepted.get("n_valid_proposal", pd.Series(dtype=float)).fillna(0) +
                                accepted.get("n_valid_acceptance", pd.Series(dtype=float)).fillna(0)).sum())
        candidates = ev[ev.decision.isin(["accept", "reject", "no_proposal"])] if len(ev) else ev
        s_trace = payload["committed_s"]
        lo, hi = calib_cfg["parameterization"]["bounds"]
        holdout_rows.append({
            "run_id": run_id, "mode": mode, "model_seed": seed, "final_s": final_s,
            "holdout_nrmse": m["nrmse"], "holdout_mae_g_s": m["mae_g_s"],
            "n_accepted_updates": int((ev.decision == "accept").sum()) if len(ev) else 0,
            "n_rejected_updates": int((ev.decision == "reject").sum()) if len(ev) else 0,
            "n_rollback_events": int(ev.rollback.fillna(False).sum()) if len(ev) and "rollback" in ev else 0,
            "n_false_updates": int(ev.false_update.fillna(False).sum()) if len(ev) and "false_update" in ev else 0,
            "false_update_rate": float(accepted.false_update.mean()) if len(accepted) else 0.0,
            "missed_update_rate": float((candidates.decision == "no_proposal").mean()) if len(candidates) else 0.0,
            "accepted_calibration_samples": accepted_samples,
            "adaptation_time_s": adaptation_time,
            "pseudo_reference_quality_r2": _pseudo_r2(df),
            "bound_violations": int(np.sum((s_trace < lo) | (s_trace > hi))),
        })
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(os.path.join(results_dir, "drift_recovery_metrics.csv"), index=False)

    grid = exp_cfg["experiments"]["E6_informative_data_efficiency"]["accepted_sample_grid"]
    efficiency = []
    for budget in grid:
        errors = []
        for (run_id, mode, seed), payload in payloads.items():
            if mode != "C3":
                continue
            df = payload["df"]
            s_eff = scale_at_valid_sample_budget(payload, df, budget)
            holdout = (df.state == "holdout").to_numpy()
            errors.append(core_metrics(s_eff * payload["phi_all"][holdout],
                                       df.maf_reference_g_s.to_numpy()[holdout], value_range)["nrmse"])
        mean, lo, hi = bootstrap_ci(errors, n_boot=2000) if errors else (float("nan"),) * 3
        efficiency.append({"max_accepted_valid_samples": budget,
                           "median_holdout_nrmse": float(np.median(errors)) if errors else float("nan"),
                           "mean_holdout_nrmse": mean, "ci_lo": lo, "ci_hi": hi,
                           "n_run_seed_pairs": len(errors)})
    pd.DataFrame(efficiency).to_csv(os.path.join(results_dir, "calibration_data_efficiency.csv"), index=False)

    comp = holdout_df[holdout_df["mode"].isin(["C3", "C5"])].pivot_table(
        index=["run_id", "model_seed"], columns="mode", values=["holdout_nrmse", "final_s"]
    )
    comp.columns = [f"{mode}_{metric}" for metric, mode in comp.columns]
    comp.reset_index().to_csv(os.path.join(results_dir, "classical_comparator_metrics.csv"), index=False)

    gate_rows = []
    if len(fisher):
        for (run_id, mode, seed), g in fisher.groupby(["run_id", "mode", "model_seed"]):
            full = g[g.n_in_window == calib_cfg["stage_b_information"]["valid_samples_per_window"]]
            gate_rows.append({"run_id": run_id, "mode": mode, "model_seed": seed,
                              "n_valid_samples_per_window": calib_cfg["stage_b_information"]["valid_samples_per_window"],
                              "pseudo_reference_variance_value": calib_cfg["stage_b_information"]["pseudo_reference_variance_value"],
                              "scalar_fisher_threshold": calib_cfg["stage_b_information"]["scalar_fisher_threshold"],
                              "mean_scalar_fisher_score": float(full.scalar_fisher_score.mean()) if len(full) else float("nan"),
                              "gate_open_fraction": float(full.gate_open.mean()) if len(full) else 0.0,
                              "n_full_information_windows": int(len(full))})
    pd.DataFrame(gate_rows).to_csv(os.path.join(results_dir, "information_gate_report.csv"), index=False)

    with open("configs/output_profiles.yaml") as f:
        out_cfg = yaml.safe_load(f)
    profile = LegacyOutputProfile(out_cfg["voltage_profile"])
    art = joblib.load(os.path.join(models_dir, "residual_ai_seed1.joblib"))
    sample_payload = next(iter(payloads.values())) if payloads else None
    legacy_rows = []
    if sample_payload:
        df = sample_payload["df"]
        estimated, _ = predict_residual_ai(art, df, load_engine_config(engine_cfg_path))
        prof = evaluate_profile(profile, estimated, dt_s=float(np.median(np.diff(df.timestamp_s))))
        X = df[art["features"]].to_numpy(dtype=float)
        legacy_rows.append({**prof, **model_resource_footprint(art), **host_timing_inference(art, X)})
    pd.DataFrame(legacy_rows).to_csv(os.path.join(results_dir, "legacy_output_metrics.csv"), index=False)

    build_acceptance_report(exp_cfg, holdout_df, events,
                            os.path.join(results_dir, "acceptance_report.json"))
    return events, fisher, holdout_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--calibration-config", default="configs/calibration.yaml")
    parser.add_argument("--engine-config", default="configs/engine_m30b35.yaml")
    parser.add_argument("--splits-csv", default="data/splits/splits.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    with open(args.config) as f:
        exp_cfg = yaml.safe_load(f)
    with open(args.calibration_config) as f:
        calib_cfg = yaml.safe_load(f)
    engine_cfg = load_engine_config(args.engine_config)
    os.makedirs(args.results_dir, exist_ok=True)
    value_range = exp_cfg["primary_endpoint"]["normalization_airflow_range_g_s"]
    seeds = exp_cfg["seeds"]
    evaluate_e1_e4(args.splits_csv, engine_cfg, load_pure_ai_models(args.models_dir, seeds),
                   load_residual_ai_models(args.models_dir, seeds), args.results_dir, value_range)
    evaluate_e5_e7(calib_cfg, exp_cfg, args.splits_csv, args.models_dir,
                   args.engine_config, args.results_dir)


if __name__ == "__main__":
    main()
