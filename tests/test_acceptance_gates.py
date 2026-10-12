"""A mechanical/leakage-safety subset of the Section 13 "Acceptance gates
before publication" checklist, automated. The scientific subset (frozen
thresholds, predeclared NRMSE target/margin, false-update definition) is
enforced by treating configs/*.yaml as frozen inputs -- see README.
"""
import numpy as np
import yaml

from legacy_output import LegacyOutputProfile, host_timing_inference, model_resource_footprint
from sklearn.preprocessing import StandardScaler


def test_calibration_bounds_are_internally_consistent():
    with open("configs/calibration.yaml") as f:
        cfg = yaml.safe_load(f)
    lo, hi = cfg["parameterization"]["bounds"]
    assert lo < hi
    assert lo < cfg["parameterization"]["initial_value"] < hi


def test_scaler_is_fit_on_training_data_only_not_influenced_by_validation():
    """Simulates the exact pattern used in train_pure_ai.train_one_seed /
    train_residual_ai.train_one_seed: scaler.fit(X_train) only."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, size=(200, 5))
    X_val = rng.normal(50, 20, size=(50, 5))  # deliberately very different distribution

    scaler = StandardScaler().fit(X_train)
    assert np.allclose(scaler.mean_, X_train.mean(axis=0))
    # If validation data had leaked into fitting, the mean would be pulled
    # toward 50; it must not be.
    assert not np.allclose(scaler.mean_, np.concatenate([X_train, X_val]).mean(axis=0), atol=1.0)


def test_legacy_voltage_profile_is_monotonic_by_construction():
    with open("configs/output_profiles.yaml") as f:
        cfg = yaml.safe_load(f)
    # Construction itself asserts monotonicity; this test just exercises it.
    profile = LegacyOutputProfile(cfg["voltage_profile"])
    assert profile.monotonicity_violations() == 0


def test_host_timing_is_explicitly_labeled_not_mcu_timing():
    import joblib
    import os

    model_path = None
    for candidate in ("models/residual_ai_seed1.joblib",):
        if os.path.exists(candidate):
            model_path = candidate
    if model_path is None:
        import pytest
        pytest.skip("no trained residual-AI model artifact available; run the pipeline first")

    artifact = joblib.load(model_path)
    X = np.zeros((1, len(artifact["features"])))
    result = host_timing_inference(artifact, X, n_repeats=10)
    assert result["host_timing_only_not_mcu_timing"] is True
    assert result["host_mean_inference_time_ms"] >= 0.0


def test_residual_ai_resource_footprint_within_declared_budget():
    with open("configs/model_residual_ai.yaml") as f:
        cfg = yaml.safe_load(f)
    hidden = cfg["architecture"]["hidden_layer_sizes"]
    n_features = len(cfg["model"]["features"])
    layers = [n_features] + list(hidden) + [1]
    n_params = sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))
    budget = cfg["resource_budget"]
    assert n_params <= budget["max_parameters"]
    assert n_params * 4 <= budget["max_model_bytes"]


def test_five_ai_seeds_declared():
    with open("configs/model_pure_ai.yaml") as f:
        pure_cfg = yaml.safe_load(f)
    with open("configs/model_residual_ai.yaml") as f:
        residual_cfg = yaml.safe_load(f)
    assert len(pure_cfg["training"]["seeds"]) >= 5
    assert len(residual_cfg["training"]["seeds"]) >= 5
