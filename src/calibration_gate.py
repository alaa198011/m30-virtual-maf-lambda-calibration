"""Four-stage calibration gate (roadmap Section 9): pure, testable stage
functions. ``online_calibration.py`` drives these per-window/per-sample for
each mode C0-C5.

  Stage A - operational validity
  Stage B - scalar Fisher-information gate (scalar_fisher_gate.py)
  Stage C - bounded proposal (absolute / per-update / rate limits, deadband)
  Stage D - causal acceptance (strictly later window, predeclared loss,
            rollback to last-known-good on rejection)
"""
from __future__ import annotations

import numpy as np


def stage_a_validity(row: dict, cfg: dict) -> tuple[bool, str]:
    """row must provide: calibration_allowed (bool, already encodes lambda
    validity / fuel-cut / enrichment / cold-start / saturation / misfire /
    dropout / fault-code blocking per generate_dataset.py), ect_c, and
    optionally tps_rate_pct_per_s for the severe-transient cutoff."""
    reasons = []
    if not row.get("calibration_allowed", False):
        reasons.append("data_invalid_or_blocked")
    if row.get("ect_c", 999) < cfg["min_ect_c"]:
        reasons.append("cold")
    tps_rate = row.get("tps_rate_pct_per_s")
    if tps_rate is not None and abs(tps_rate) > cfg["max_abs_tps_rate_pct_per_s"]:
        reasons.append("severe_transient")
    valid = len(reasons) == 0
    return valid, ("ok" if valid else ",".join(reasons))


def bounded_proposal(current_committed_s: float, candidate_s: float, cfg_param: dict,
                      dt_since_last_commit_s: float) -> tuple[float, bool, str]:
    """Stage C: deadband -> per-update limit -> absolute bounds -> rate limit.

    Returns (proposed_s, changed: bool, reason).
    """
    delta = candidate_s - current_committed_s
    if abs(delta) < cfg_param["deadband"]:
        return current_committed_s, False, "deadband"

    delta = float(np.clip(delta, -cfg_param["per_update_limit"], cfg_param["per_update_limit"]))
    proposed = current_committed_s + delta

    lo, hi = cfg_param["bounds"]
    bound_violation = not (lo <= proposed <= hi)
    proposed = float(np.clip(proposed, lo, hi))

    max_rate_delta = cfg_param["rate_limit_per_s"] * max(dt_since_last_commit_s, 1e-6)
    proposed = float(np.clip(proposed, current_committed_s - max_rate_delta,
                              current_committed_s + max_rate_delta))

    if abs(proposed - current_committed_s) < 1e-12:
        return current_committed_s, False, "rate_limited_to_zero"

    reason = "bound_clamped" if bound_violation else "ok"
    return proposed, True, reason


def causal_acceptance(candidate_loss: float, reference_loss: float, cfg_stage_d: dict) -> tuple[bool, float]:
    """Stage D: candidate must beat the last-known-good reference loss on a
    strictly later acceptance window by >= min_relative_improvement.

    Returns (accept: bool, relative_improvement: float).
    """
    denom = max(reference_loss, 1e-9)
    improvement = (reference_loss - candidate_loss) / denom
    accept = improvement >= cfg_stage_d["min_relative_improvement"]
    return accept, improvement


def is_false_accepted_update(accepted: bool, hidden_error_before: float, hidden_error_after: float,
                              tolerance_g_s: float) -> bool:
    """Offline-only labelling (Section 10): an update is 'false' when it
    passes acceptance but INCREASES hidden clean-airflow error on the same
    acceptance window beyond a predeclared tolerance. Hidden airflow is
    NEVER available to the online proposal/calibration/acceptance logic --
    this function is called only by evaluate.py / online_calibration.py's
    offline logging path, after the online decision has already been made.
    """
    if not accepted:
        return False
    return (hidden_error_after - hidden_error_before) > tolerance_g_s
