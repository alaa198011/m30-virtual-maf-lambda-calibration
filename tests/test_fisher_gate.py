"""Scalar Fisher-information gate behavior (roadmap Section 9, Stage B):
the frozen threshold should separate a genuinely low-information window
(sustained idle-like small base-airflow estimates) from a genuinely
informative one (normal driving), using the exact frozen N/sigma^2/threshold
declared in configs/calibration.yaml.
"""
import yaml

from scalar_fisher_gate import ScalarFisherGate, scalar_fisher_information


def _load_stage_b_cfg():
    with open("configs/calibration.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["stage_b_information"]


def test_scalar_fisher_information_formula():
    # I_s = sum(phi_k^2) / sigma^2_pseudo
    phi = [10.0, 10.0, 10.0]
    sigma2 = 4.0
    assert scalar_fisher_information(phi, sigma2) == 300.0 / 4.0


def test_gate_closed_during_sustained_idle_like_window():
    cfg = _load_stage_b_cfg()
    gate = ScalarFisherGate(
        n_valid=cfg["valid_samples_per_window"],
        sigma2=cfg["pseudo_reference_variance_value"],
        threshold=cfg["scalar_fisher_threshold"],
    )
    idle_phi = 3.0  # g/s -- representative low-load base-airflow estimate
    for _ in range(cfg["valid_samples_per_window"]):
        gate.push_valid_sample(idle_phi)
    gate_open, i_s, n_in_window = gate.evaluate()
    assert n_in_window == cfg["valid_samples_per_window"]
    assert not gate_open, f"idle-level window unexpectedly opened the gate (I_s={i_s})"


def test_gate_open_during_normal_driving_window():
    cfg = _load_stage_b_cfg()
    gate = ScalarFisherGate(
        n_valid=cfg["valid_samples_per_window"],
        sigma2=cfg["pseudo_reference_variance_value"],
        threshold=cfg["scalar_fisher_threshold"],
    )
    driving_phi = 40.0  # g/s -- representative cruise/acceleration base estimate
    for _ in range(cfg["valid_samples_per_window"]):
        gate.push_valid_sample(driving_phi)
    gate_open, i_s, _n = gate.evaluate()
    assert gate_open, f"normal-driving window unexpectedly stayed closed (I_s={i_s})"


def test_gate_stays_closed_until_window_is_full():
    cfg = _load_stage_b_cfg()
    gate = ScalarFisherGate(
        n_valid=cfg["valid_samples_per_window"],
        sigma2=cfg["pseudo_reference_variance_value"],
        threshold=cfg["scalar_fisher_threshold"],
    )
    for _ in range(cfg["valid_samples_per_window"] - 1):
        gate.push_valid_sample(1000.0)  # huge phi, but window not yet full
        gate_open, _i_s, _n = gate.evaluate()
        assert not gate_open
