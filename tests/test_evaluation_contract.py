import json

import numpy as np
import pandas as pd

from evaluate import build_acceptance_report, core_metrics, scale_at_valid_sample_budget


def test_nrmse_uses_declared_range():
    metrics = core_metrics(np.array([2.0, 4.0]), np.array([0.0, 0.0]), value_range=10.0)
    assert np.isclose(metrics["nrmse"], np.sqrt(10.0) / 10.0)


def test_data_efficiency_budget_counts_valid_samples_not_updates():
    df = pd.DataFrame({"state": ["pre_drift", "adaptation", "adaptation", "adaptation"]})
    payload = {"valid_count_by_index": np.array([0, 1, 2, 3]),
               "committed_s": np.array([1.0, 1.0, 0.98, 0.96])}
    assert scale_at_valid_sample_budget(payload, df, 1) == 1.0
    assert scale_at_valid_sample_budget(payload, df, 2) == 0.98
    assert scale_at_valid_sample_budget(payload, df, 3) == 0.96


def test_acceptance_report_evaluates_every_primary_gate(tmp_path):
    exp = {"primary_endpoint": {
        "absolute_target": 0.12,
        "min_improvement_margin_over_C0": 0.10,
        "require_zero_bound_violations": True,
        "require_lower_false_update_rate_than": "C1",
    }}
    holdout = pd.DataFrame([
        {"run_id": "calibration_shift_test__run000", "mode": "C0", "holdout_nrmse": 0.10, "bound_violations": 0},
        {"run_id": "calibration_shift_test__run000", "mode": "C3", "holdout_nrmse": 0.08, "bound_violations": 0},
    ])
    events = pd.DataFrame([
        {"mode": "C1", "decision": "accept", "false_update": True},
        {"mode": "C3", "decision": "accept", "false_update": False},
    ])
    out = tmp_path / "acceptance.json"
    report = build_acceptance_report(exp, holdout, events, out)
    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert json.loads(out.read_text())["status"] == "PASS"
