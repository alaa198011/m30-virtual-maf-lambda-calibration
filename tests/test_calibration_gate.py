"""Stage A / C / D pure-function tests (roadmap Section 9)."""
import yaml

from calibration_gate import bounded_proposal, causal_acceptance, is_false_accepted_update, stage_a_validity


def _param_cfg():
    with open("configs/calibration.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["parameterization"], cfg["stage_a_validity"], cfg["stage_d_acceptance"]


def test_stage_a_blocks_on_invalid_data():
    _param, stage_a_cfg, _d = _param_cfg()
    row = {"calibration_allowed": False, "ect_c": 90.0}
    valid, reason = stage_a_validity(row, stage_a_cfg)
    assert not valid
    assert "data_invalid_or_blocked" in reason


def test_stage_a_blocks_on_cold_engine():
    _param, stage_a_cfg, _d = _param_cfg()
    row = {"calibration_allowed": True, "ect_c": 20.0}
    valid, reason = stage_a_validity(row, stage_a_cfg)
    assert not valid
    assert "cold" in reason


def test_stage_a_blocks_on_severe_transient():
    _param, stage_a_cfg, _d = _param_cfg()
    row = {"calibration_allowed": True, "ect_c": 90.0, "tps_rate_pct_per_s": 1000.0}
    valid, reason = stage_a_validity(row, stage_a_cfg)
    assert not valid
    assert "severe_transient" in reason


def test_stage_a_passes_clean_row():
    _param, stage_a_cfg, _d = _param_cfg()
    row = {"calibration_allowed": True, "ect_c": 90.0, "tps_rate_pct_per_s": 5.0}
    valid, reason = stage_a_validity(row, stage_a_cfg)
    assert valid
    assert reason == "ok"


def test_bounded_proposal_respects_absolute_bounds():
    param_cfg, _a, _d = _param_cfg()
    # candidate wildly outside bounds must be clamped, not just limited by rate/step
    proposed, changed, reason = bounded_proposal(
        current_committed_s=1.0, candidate_s=5.0, cfg_param=param_cfg, dt_since_last_commit_s=8.0
    )
    lo, hi = param_cfg["bounds"]
    assert lo <= proposed <= hi
    assert changed

    proposed_lo, _c, _r = bounded_proposal(
        current_committed_s=1.0, candidate_s=-5.0, cfg_param=param_cfg, dt_since_last_commit_s=8.0
    )
    assert proposed_lo >= lo


def test_bounded_proposal_respects_per_update_limit():
    param_cfg, _a, _d = _param_cfg()
    proposed, changed, _reason = bounded_proposal(
        current_committed_s=1.0, candidate_s=1.0 + 10 * param_cfg["per_update_limit"],
        cfg_param=param_cfg, dt_since_last_commit_s=100.0,  # generous rate budget
    )
    assert changed
    assert abs(proposed - 1.0) <= param_cfg["per_update_limit"] + 1e-9


def test_bounded_proposal_respects_rate_limit():
    param_cfg, _a, _d = _param_cfg()
    # Even a tiny candidate delta should be capped further by a very short dt.
    proposed, changed, _reason = bounded_proposal(
        current_committed_s=1.0, candidate_s=1.0 + param_cfg["per_update_limit"],
        cfg_param=param_cfg, dt_since_last_commit_s=0.01,
    )
    max_allowed = param_cfg["rate_limit_per_s"] * 0.01
    assert abs(proposed - 1.0) <= max_allowed + 1e-9


def test_bounded_proposal_deadband_suppresses_tiny_changes():
    param_cfg, _a, _d = _param_cfg()
    proposed, changed, reason = bounded_proposal(
        current_committed_s=1.0, candidate_s=1.0 + param_cfg["deadband"] / 10.0,
        cfg_param=param_cfg, dt_since_last_commit_s=8.0,
    )
    assert not changed
    assert reason == "deadband"
    assert proposed == 1.0


def test_causal_acceptance_requires_minimum_improvement():
    _param, _a, stage_d_cfg = _param_cfg()
    # Small improvement below the frozen minimum must be rejected.
    accept, improvement = causal_acceptance(candidate_loss=9.9, reference_loss=10.0, cfg_stage_d=stage_d_cfg)
    assert not accept
    # Large improvement must be accepted.
    accept2, improvement2 = causal_acceptance(candidate_loss=5.0, reference_loss=10.0, cfg_stage_d=stage_d_cfg)
    assert accept2
    assert improvement2 > improvement


def test_false_update_labelling_only_applies_to_accepted_updates():
    assert not is_false_accepted_update(accepted=False, hidden_error_before=1.0, hidden_error_after=10.0, tolerance_g_s=1.5)
    assert is_false_accepted_update(accepted=True, hidden_error_before=1.0, hidden_error_after=10.0, tolerance_g_s=1.5)
    assert not is_false_accepted_update(accepted=True, hidden_error_before=1.0, hidden_error_after=1.2, tolerance_g_s=1.5)
