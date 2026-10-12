"""Section 7 / Section 13 acceptance gate: "The pipeline must fail if a
hidden reference feature is passed to the deployed estimator." / "hidden
reference features are absent from deployed estimator inputs."
"""
import yaml
import pytest

from generate_dataset import FORBIDDEN_DEPLOYED_FEATURES
from train_pure_ai import assert_no_leakage as pure_assert
from train_residual_ai import assert_no_leakage as residual_assert


def test_pure_ai_feature_list_has_no_leakage():
    with open("configs/model_pure_ai.yaml") as f:
        cfg = yaml.safe_load(f)
    pure_assert(cfg["model"]["features"])  # must not raise


def test_residual_ai_feature_list_has_no_leakage():
    with open("configs/model_residual_ai.yaml") as f:
        cfg = yaml.safe_load(f)
    residual_assert(cfg["model"]["features"])  # must not raise


def test_forbidden_features_include_hidden_reference_columns():
    assert "maf_reference_g_s" in FORBIDDEN_DEPLOYED_FEATURES
    assert "maf_physics_g_s" in FORBIDDEN_DEPLOYED_FEATURES
    assert "lambda_measured" in FORBIDDEN_DEPLOYED_FEATURES


@pytest.mark.parametrize("leaked_feature", ["maf_reference_g_s", "lambda_measured", "injector_pw_ms"])
def test_leakage_guard_actually_fails_closed(leaked_feature):
    with pytest.raises(AssertionError):
        pure_assert(["rpm", "map_kpa", leaked_feature])


def test_deployed_feature_list_is_exactly_the_declared_i_o_contract():
    """Section 3: the frozen deployed estimator uses only RPM, MAP, TPS,
    IAT, ECT -- no OBD-II or CAN diagnostic PID, no hidden reference."""
    with open("configs/model_pure_ai.yaml") as f:
        pure_cfg = yaml.safe_load(f)
    with open("configs/model_residual_ai.yaml") as f:
        residual_cfg = yaml.safe_load(f)
    expected = {"rpm", "map_kpa", "tps_pct", "iat_c", "ect_c"}
    assert set(pure_cfg["model"]["features"]) == expected
    assert set(residual_cfg["model"]["features"]) == expected
