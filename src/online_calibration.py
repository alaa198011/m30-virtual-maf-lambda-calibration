"""Causal, delay-aligned online calibration policies C0--C5."""
from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import yaml

from calibration_gate import bounded_proposal, causal_acceptance, is_false_accepted_update, stage_a_validity
from classical_rls import OneStateRLS
from fuel_lambda_path import condition_series_like_pseudo_reference
from physics_model import load_engine_config
from scalar_fisher_gate import ScalarFisherGate
from train_residual_ai import predict_residual_ai


def _least_squares_scale(phi, y):
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(phi) & np.isfinite(y)
    phi, y = phi[finite], y[finite]
    denom = float(np.sum(phi ** 2))
    if denom < 1e-9:
        return None
    return float(np.sum(phi * y) / denom)


def _rmse(pred, ref):
    pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    finite = np.isfinite(pred) & np.isfinite(ref)
    if not finite.any():
        return np.nan
    return float(np.sqrt(np.mean((pred[finite] - ref[finite]) ** 2)))


def align_phi_to_pseudo_reference(df: pd.DataFrame, phi_all: np.ndarray) -> np.ndarray:
    """Apply the recorded pseudo-reference delay/filter contract to ``phi``."""
    if "pseudo_fixed_delay_samples" not in df or "pseudo_filter_alpha" not in df:
        raise KeyError("dataset lacks pseudo-reference alignment metadata")
    delay_steps = int(df["pseudo_fixed_delay_samples"].iloc[0])
    alpha = float(df["pseudo_filter_alpha"].iloc[0])
    return condition_series_like_pseudo_reference(phi_all, delay_steps, alpha)


class WindowBuf:
    def __init__(self):
        self.phi, self.y, self.hidden, self.t = [], [], [], []

    def add(self, phi_k, y_k, hidden_k, t_k):
        self.phi.append(float(phi_k))
        self.y.append(float(y_k))
        self.hidden.append(float(hidden_k))
        self.t.append(float(t_k))

    def span_s(self):
        return 0.0 if len(self.t) < 2 else self.t[-1] - self.t[0]

    def clear(self):
        self.phi, self.y, self.hidden, self.t = [], [], [], []

    def n(self):
        return len(self.t)


def _event_base(run_id, mode_name, model_seed, window_id, proposal_buf,
                fisher_score, fisher_threshold, candidate_s):
    return {
        "run_id": run_id, "mode": mode_name, "model_seed": model_seed,
        "window_id": window_id,
        "t_proposal_start": proposal_buf.t[0],
        "t_proposal_end": proposal_buf.t[-1],
        "n_valid_proposal": proposal_buf.n(),
        "scalar_fisher_score": fisher_score,
        "scalar_fisher_threshold": fisher_threshold,
        "candidate_s": candidate_s,
    }


def run_calibration_on_run(df: pd.DataFrame, phi_all: np.ndarray, mode_name: str,
                            mode_cfg: dict, calib_cfg: dict, dt_s: float,
                            run_id: str, model_seed: int = 1) -> dict:
    """Run one calibration policy with disjoint proposal/acceptance/holdout data."""
    n = len(df)
    param_cfg = calib_cfg["parameterization"]
    stage_a_cfg = calib_cfg["stage_a_validity"]
    stage_b_cfg = calib_cfg["stage_b_information"]
    stage_d_cfg = calib_cfg["stage_d_acceptance"]
    windows_cfg = calib_cfg["windows"]
    false_cfg = calib_cfg["false_update_labelling"]
    initial_s = float(param_cfg["initial_value"])

    t = df["timestamp_s"].to_numpy(dtype=float)
    ect = df["ect_c"].to_numpy(dtype=float)
    cal_allowed = df["calibration_allowed"].to_numpy(dtype=bool)
    tps = df["tps_pct"].to_numpy(dtype=float)
    pseudo_ref = df["pseudo_reference_g_s"].to_numpy(dtype=float)
    hidden_ref = df["maf_reference_g_s"].to_numpy(dtype=float)
    state = df["state"].to_numpy(dtype=object)
    phi_aligned = align_phi_to_pseudo_reference(df, phi_all)
    hidden_aligned = align_phi_to_pseudo_reference(df, hidden_ref)
    tps_rate = np.concatenate([[0.0], np.diff(tps) / max(dt_s, 1e-9)])

    committed_s = np.full(n, initial_s, dtype=float)
    valid_count_by_index = np.zeros(n, dtype=int)
    # The pre-drift segment establishes the frozen baseline.  Online learning
    # begins only after the declared mismatch is introduced; this prevents a
    # nominal-condition update from consuming the limited learning drive.
    active_mask = state == "adaptation"
    active_idx = np.where(active_mask)[0]
    holdout_idx = np.where(state == "holdout")[0]
    events, fisher_trace, trajectory = [], [], []

    def validity(i):
        row = {"calibration_allowed": bool(cal_allowed[i]), "ect_c": float(ect[i]),
               "tps_rate_pct_per_s": float(tps_rate[i])}
        return stage_a_validity(row, stage_a_cfg)

    if mode_name == "C0" or len(active_idx) == 0:
        if len(holdout_idx):
            committed_s[holdout_idx[0]:] = initial_s
        return {"committed_s": committed_s, "events": events,
                "fisher_trace": fisher_trace, "final_s": initial_s,
                "phi_aligned": phi_aligned,
                "valid_count_by_index": valid_count_by_index,
                "parameter_trajectory": trajectory}

    use_stage_a = mode_cfg.get("stage_a", False)
    use_stage_b = mode_cfg.get("stage_b", False)
    use_stage_d = mode_cfg.get("stage_d", False)
    always_accept = mode_cfg.get("always_accept", False)
    uses_oracle = mode_cfg.get("uses_hidden_oracle_feedback", False)
    is_rls = mode_cfg.get("estimator") == "classical_rls"
    last_known_good_s = initial_s
    last_commit_t = float(t[active_idx[0]])
    cumulative_valid = 0

    if is_rls:
        rls = OneStateRLS(
            initial_value=initial_s, bounds=tuple(param_cfg["bounds"]),
            per_update_limit=param_cfg["per_update_limit"],
            rate_limit_per_s=param_cfg["rate_limit_per_s"],
            forgetting_factor=mode_cfg.get("forgetting_factor", 0.98),
        )
        for i in active_idx:
            valid, _ = validity(i)
            if valid:
                cumulative_valid += 1
                rls.s = rls.update(phi_aligned[i], pseudo_ref[i], dt_s)
            committed_s[i] = rls.s
            valid_count_by_index[i] = cumulative_valid
            trajectory.append({"run_id": run_id, "mode": mode_name,
                               "model_seed": model_seed, "t_s": t[i],
                               "committed_s": rls.s,
                               "cumulative_valid_samples": cumulative_valid})
        last_known_good_s = rls.s
        if len(holdout_idx):
            committed_s[holdout_idx[0]:] = last_known_good_s
            valid_count_by_index[holdout_idx[0]:] = cumulative_valid
        return {"committed_s": committed_s, "events": events,
                "fisher_trace": fisher_trace, "final_s": last_known_good_s,
                "phi_aligned": phi_aligned,
                "valid_count_by_index": valid_count_by_index,
                "parameter_trajectory": trajectory}

    fisher_gate = ScalarFisherGate(
        n_valid=stage_b_cfg["valid_samples_per_window"],
        sigma2=stage_b_cfg["pseudo_reference_variance_value"],
        threshold=stage_b_cfg["scalar_fisher_threshold"],
    )
    proposal_buf, acceptance_buf = WindowBuf(), WindowBuf()
    phase, candidate_s, pending = "proposal", None, None
    window_id = 0
    acceptance_not_before = -np.inf

    for i in active_idx:
        stage_a_ok, stage_a_reason = validity(i)
        valid = stage_a_ok if use_stage_a else True
        if stage_a_ok:
            cumulative_valid += 1
        valid_count_by_index[i] = cumulative_valid
        committed_s[i] = last_known_good_s

        phi_k, y_k, hid_k = phi_aligned[i], pseudo_ref[i], hidden_aligned[i]
        if phase == "proposal":
            if valid:
                proposal_buf.add(phi_k, y_k, hid_k, t[i])
                if use_stage_b:
                    fisher_gate.push_valid_sample(phi_k)
                    gate_open, score, n_win = fisher_gate.evaluate()
                    fisher_trace.append({
                        "run_id": run_id, "mode": mode_name, "model_seed": model_seed,
                        "window_id": window_id + 1, "t_s": float(t[i]),
                        "scalar_fisher_score": score,
                        "scalar_fisher_threshold": fisher_gate.threshold,
                        "gate_open": gate_open, "n_in_window": n_win,
                    })
            elif use_stage_b and stage_b_cfg.get("reset_on_invalid_sample", True):
                fisher_gate.reset()

            if proposal_buf.span_s() >= windows_cfg["proposal_window_s"] and proposal_buf.n() >= 2:
                window_id += 1
                target = proposal_buf.hidden if uses_oracle else proposal_buf.y
                s_hat = _least_squares_scale(proposal_buf.phi, target)
                gate_open, score = True, None
                if use_stage_b:
                    gate_open, score, _ = fisher_gate.evaluate()
                threshold = fisher_gate.threshold if use_stage_b else None
                base = _event_base(run_id, mode_name, model_seed, window_id,
                                   proposal_buf, score, threshold, s_hat)
                if s_hat is None or not gate_open:
                    base.update({"decision": "no_proposal",
                                 "rejection_reason": "fisher_gate_closed" if not gate_open else "insufficient_data",
                                 "committed_s_after": last_known_good_s,
                                 "rollback": False, "false_update": False,
                                 "cumulative_valid_samples": cumulative_valid})
                    events.append(base)
                    proposal_buf.clear()
                    fisher_gate.reset()
                    continue

                elapsed = max(float(t[i]) - last_commit_t, dt_s)
                proposed, changed, c_reason = bounded_proposal(
                    last_known_good_s, s_hat, param_cfg, elapsed
                )
                if not changed:
                    base.update({"decision": "no_proposal", "rejection_reason": "deadband",
                                 "committed_s_after": last_known_good_s,
                                 "rollback": False, "false_update": False,
                                 "cumulative_valid_samples": cumulative_valid})
                    events.append(base)
                    proposal_buf.clear()
                    fisher_gate.reset()
                    continue

                proposal_target = np.asarray(target, dtype=float)
                proposal_phi = np.asarray(proposal_buf.phi, dtype=float)
                proposal_loss = float(np.mean((proposed * proposal_phi - proposal_target) ** 2))
                base.update({"candidate_s": proposed,
                             "unbounded_candidate_s": s_hat,
                             "proposal_stage_c_reason": c_reason,
                             "proposal_loss": proposal_loss})
                candidate_s, pending = proposed, base
                acceptance_not_before = float(t[i]) + float(windows_cfg.get("min_gap_proposal_to_acceptance_s", 0.0))
                proposal_buf.clear()
                acceptance_buf.clear()
                fisher_gate.reset()
                phase = "acceptance"

        else:
            if t[i] <= acceptance_not_before:
                continue
            if valid:
                acceptance_buf.add(phi_k, y_k, hid_k, t[i])
            if acceptance_buf.span_s() >= windows_cfg["acceptance_window_s"] and acceptance_buf.n() >= 2:
                phi_a = np.asarray(acceptance_buf.phi)
                y_a = np.asarray(acceptance_buf.y)
                hid_a = np.asarray(acceptance_buf.hidden)
                target_a = hid_a if uses_oracle else y_a
                candidate_loss = float(np.mean((candidate_s * phi_a - target_a) ** 2))
                reference_loss = float(np.mean((last_known_good_s * phi_a - target_a) ** 2))
                if always_accept:
                    accept, improvement = True, np.nan
                elif use_stage_d:
                    accept, improvement = causal_acceptance(candidate_loss, reference_loss, stage_d_cfg)
                else:
                    accept, improvement = True, np.nan

                hidden_before = _rmse(last_known_good_s * phi_a, hid_a)
                hidden_after = _rmse(candidate_s * phi_a, hid_a)
                false_update = is_false_accepted_update(
                    accept, hidden_before, hidden_after,
                    false_cfg["hidden_airflow_error_tolerance_g_s"],
                )
                if accept:
                    last_known_good_s = candidate_s
                    last_commit_t = float(t[i])
                    rollback = False
                    committed_s[i] = last_known_good_s
                else:
                    rollback = bool(stage_d_cfg.get("rollback_on_reject", True))

                pending.update({
                    "t_acceptance_start": acceptance_buf.t[0],
                    "t_acceptance_end": acceptance_buf.t[-1],
                    "n_valid_acceptance": acceptance_buf.n(),
                    "acceptance_window_loss": candidate_loss,
                    "reference_loss": reference_loss,
                    "relative_improvement": improvement,
                    "decision": "accept" if accept else "reject",
                    "rejection_reason": "" if accept else "insufficient_improvement",
                    "committed_s_after": last_known_good_s,
                    "rollback": rollback, "false_update": bool(false_update),
                    "hidden_error_before_g_s": hidden_before,
                    "hidden_error_after_g_s": hidden_after,
                    "cumulative_valid_samples": cumulative_valid,
                })
                events.append(pending)
                acceptance_buf.clear()
                phase, candidate_s, pending = "proposal", None, None

        trajectory.append({"run_id": run_id, "mode": mode_name,
                           "model_seed": model_seed, "t_s": t[i],
                           "committed_s": last_known_good_s,
                           "cumulative_valid_samples": cumulative_valid})

    if pending is not None:
        pending.update({"decision": "incomplete", "rejection_reason": "acceptance_window_incomplete",
                        "committed_s_after": last_known_good_s, "rollback": True,
                        "false_update": False, "cumulative_valid_samples": cumulative_valid})
        events.append(pending)
    if len(holdout_idx):
        committed_s[holdout_idx[0]:] = last_known_good_s
        valid_count_by_index[holdout_idx[0]:] = cumulative_valid

    return {"committed_s": committed_s, "events": events,
            "fisher_trace": fisher_trace, "final_s": last_known_good_s,
            "phi_aligned": phi_aligned,
            "valid_count_by_index": valid_count_by_index,
            "parameter_trajectory": trajectory}


def run_all_modes(splits_csv: str, models_dir: str, calib_cfg: dict,
                  engine_cfg_path: str,
                  scenarios=("calibration_shift_threshold", "calibration_shift_test"),
                  model_seeds=(1,)):
    splits = pd.read_csv(splits_csv)
    runs = splits[splits.scenario.isin(scenarios)]
    engine_cfg = load_engine_config(engine_cfg_path)
    all_events, all_fisher, all_trajectory = [], [], []
    payloads = {}

    for model_seed in model_seeds:
        artifact = joblib.load(os.path.join(models_dir, f"residual_ai_seed{model_seed}.joblib"))
        for _, run_row in runs.iterrows():
            df = pd.read_parquet(run_row["path"])
            phi_all, _ = predict_residual_ai(artifact, df, engine_cfg)
            dt_s = float(np.median(np.diff(df["timestamp_s"]))) if len(df) > 1 else 0.02
            for mode_name, mode_cfg in calib_cfg["modes"].items():
                result = run_calibration_on_run(
                    df, phi_all, mode_name, mode_cfg, calib_cfg, dt_s,
                    run_row["run_id"], model_seed=model_seed,
                )
                all_events.extend(result["events"])
                all_fisher.extend(result["fisher_trace"])
                all_trajectory.extend(result["parameter_trajectory"])
                payloads[(run_row["run_id"], mode_name, model_seed)] = {
                    **result, "df": df, "phi_all": phi_all,
                }
    return pd.DataFrame(all_events), pd.DataFrame(all_fisher), payloads, pd.DataFrame(all_trajectory)


def main():
    parser = argparse.ArgumentParser(description="Run C0-C5 calibration-policy ablation.")
    parser.add_argument("--config", default="configs/calibration.yaml")
    parser.add_argument("--engine-config", default="configs/engine_m30b35.yaml")
    parser.add_argument("--splits-csv", default="data/splits/splits.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    with open(args.config) as f:
        calib_cfg = yaml.safe_load(f)
    events, fisher, _, trajectory = run_all_modes(
        args.splits_csv, args.models_dir, calib_cfg, args.engine_config
    )
    os.makedirs(args.results_dir, exist_ok=True)
    events.to_csv(os.path.join(args.results_dir, "calibration_events.csv"), index=False)
    fisher.to_csv(os.path.join(args.results_dir, "scalar_fisher_trace.csv"), index=False)
    trajectory.to_csv(os.path.join(args.results_dir, "parameter_trajectory.csv"), index=False)


if __name__ == "__main__":
    main()
