"""Unit-correctness tests (roadmap Section 5): Pa vs kPa, Celsius vs
kelvin, RPM vs rev/s, kg/s vs g/s, and the factor of 120 for a four-stroke
intake-cycle relationship.
"""
import numpy as np
import pytest

from physics_model import (
    PhysicsModel,
    celsius_to_kelvin,
    kelvin_to_celsius,
    kg_s_to_g_s,
    kpa_to_pa,
    load_engine_config,
    pa_to_kpa,
    rev_per_s_to_rpm,
    rpm_to_rev_per_s,
    g_s_to_kg_s,
    worked_example,
)


def test_kpa_pa_roundtrip():
    assert pa_to_kpa(kpa_to_pa(101.325)) == pytest.approx(101.325)
    assert kpa_to_pa(1.0) == pytest.approx(1000.0)


def test_celsius_kelvin_roundtrip():
    assert celsius_to_kelvin(0.0) == pytest.approx(273.15)
    assert kelvin_to_celsius(celsius_to_kelvin(25.0)) == pytest.approx(25.0)


def test_rpm_rev_per_s_roundtrip():
    assert rpm_to_rev_per_s(6000) == pytest.approx(100.0)
    assert rev_per_s_to_rpm(rpm_to_rev_per_s(3500)) == pytest.approx(3500.0)


def test_kg_s_g_s_roundtrip():
    assert kg_s_to_g_s(0.05) == pytest.approx(50.0)
    assert g_s_to_kg_s(kg_s_to_g_s(0.123)) == pytest.approx(0.123)


def test_factor_of_120_four_stroke():
    """Directly verifies the speed-density constant against the manual
    ideal-gas + event-rate derivation documented in physics_model.py."""
    cfg = load_engine_config()
    model = PhysicsModel.from_config(cfg)

    rpm, map_kpa, iat_c = 3000.0, 80.0, 20.0
    eta_v = model.eta_v(rpm, map_kpa)

    p_pa = kpa_to_pa(map_kpa)
    t_k = celsius_to_kelvin(iat_c)
    rho = p_pa / (cfg["constants"]["R_air_J_per_kgK"] * t_k)  # kg/m^3
    events_per_s = rpm / 120.0  # one full induction of V_d per 2 crank rev
    expected_kg_s = rho * cfg["engine"]["displacement_m3"] * eta_v * events_per_s
    expected_g_s = kg_s_to_g_s(expected_kg_s)

    got = model.airflow_g_s(rpm, map_kpa, iat_c)
    assert got == pytest.approx(float(expected_g_s), rel=1e-6)


def test_worked_example_is_physically_plausible():
    example = worked_example()
    # A 3.43 L six-cylinder engine at 3500 RPM / 90 kPa should draw a
    # physically plausible airflow, well within the bounded fallback range.
    assert 20.0 < example["airflow_g_s"] < 200.0
    assert 0.0 < example["eta_v"] <= 1.0


def test_bounded_fallback_clamps_extremes():
    cfg = load_engine_config()
    model = PhysicsModel.from_config(cfg)
    huge = model.airflow_g_s(rpm=20000.0, map_kpa=500.0, iat_c=-50.0)
    tiny = model.airflow_g_s(rpm=1.0, map_kpa=0.001, iat_c=200.0)
    assert huge <= cfg["bounded_fallback"]["max_airflow_g_s"]
    assert tiny >= cfg["bounded_fallback"]["min_airflow_g_s"]
