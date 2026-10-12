from copy import deepcopy

import numpy as np
import pandas as pd
import yaml

from online_calibration import run_calibration_on_run


def _base_frame(n=80, dt=0.1):
    return pd.DataFrame({
        "timestamp_s": np.arange(n) * dt,
        "ect_c": np.full(n, 90.0),
        "calibration_allowed": np.ones(n, dtype=bool),
        "tps_pct": np.full(n, 30.0),
        "pseudo_reference_g_s": np.full(n, 8.0),
        "maf_reference_g_s": np.full(n, 11.0),
        "state": np.array(["adaptation"] * n, dtype=object),
        "pseudo_fixed_delay_samples": np.zeros(n, dtype=int),
        "pseudo_filter_alpha": np.ones(n),
    })


def _cfg():
    with open("configs/calibration.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = deepcopy(cfg)
    cfg["windows"]["proposal_window_s"] = 1.0
    cfg["windows"]["acceptance_window_s"] = 1.0
    cfg["parameterization"]["per_update_limit"] = 0.5
    cfg["parameterization"]["rate_limit_per_s"] = 1.0
    cfg["parameterization"]["bounds"] = [0.5, 1.5]
    return cfg


def test_c4_oracle_proposes_from_hidden_airflow_not_pseudo_reference():
    df = _base_frame()
    phi = np.full(len(df), 10.0)
    cfg = _cfg()
    result = run_calibration_on_run(
        df, phi, "C4", cfg["modes"]["C4"], cfg, 0.1, "oracle-test"
    )
    completed = [e for e in result["events"] if e["decision"] in ("accept", "reject")]
    assert completed
    assert abs(completed[0]["unbounded_candidate_s"] - 1.1) < 1e-9
    assert completed[0]["decision"] == "accept"


def test_proposal_and_acceptance_windows_do_not_overlap():
    df = _base_frame()
    cfg = _cfg()
    result = run_calibration_on_run(
        df, np.full(len(df), 10.0), "C3", cfg["modes"]["C3"],
        cfg, 0.1, "causal-test"
    )
    for event in result["events"]:
        if event["decision"] in ("accept", "reject"):
            assert event["t_acceptance_start"] > event["t_proposal_end"]


def test_c3_fisher_score_comes_from_current_proposal_window():
    df = _base_frame()
    cfg = _cfg()
    cfg["stage_b_information"]["valid_samples_per_window"] = 3
    cfg["stage_b_information"]["scalar_fisher_threshold"] = 1.0
    result = run_calibration_on_run(
        df, np.full(len(df), 10.0), "C3", cfg["modes"]["C3"],
        cfg, 0.1, "fisher-test"
    )
    proposals = [e for e in result["events"] if e.get("scalar_fisher_score") is not None]
    assert proposals
    expected = 3 * 10.0 ** 2 / cfg["stage_b_information"]["pseudo_reference_variance_value"]
    assert abs(proposals[0]["scalar_fisher_score"] - expected) < 1e-9
