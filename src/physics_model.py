"""Speed-density physics baseline (roadmap Section 5).

Implements the estimator's *assumed* volumetric-efficiency surface and the
four-stroke speed-density relationship

    mdot_air = (P_MAP * V_d * N * eta_v) / (120 * R_air * T_IAT)

with explicit, unit-tested conversions (Pa vs kPa, Celsius vs kelvin, RPM
vs rev/s, kg/s vs g/s, and the factor of 120 for a four-stroke intake-cycle
relationship — one intake event per cylinder every 2 crank revolutions).

This module implements ONLY the estimator's baseline. The hidden plant
(structurally different VE surface, dynamics, and mismatch) lives in
``virtual_engine.py`` and must never import from here in a way that could
leak plant internals into the deployed estimator.
"""
from __future__ import annotations

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Unit conversion helpers (kept as standalone functions so tests can target
# each conversion independently -- roadmap Section 5 "unit tests" list).
# ---------------------------------------------------------------------------


def kpa_to_pa(value_kpa):
    """kPa -> Pa."""
    return np.asarray(value_kpa, dtype=float) * 1000.0


def pa_to_kpa(value_pa):
    """Pa -> kPa."""
    return np.asarray(value_pa, dtype=float) / 1000.0


def celsius_to_kelvin(value_c):
    """Celsius -> kelvin."""
    return np.asarray(value_c, dtype=float) + 273.15


def kelvin_to_celsius(value_k):
    """Kelvin -> Celsius."""
    return np.asarray(value_k, dtype=float) - 273.15


def rpm_to_rev_per_s(rpm):
    """RPM -> revolutions per second."""
    return np.asarray(rpm, dtype=float) / 60.0


def rev_per_s_to_rpm(rev_per_s):
    """Revolutions per second -> RPM."""
    return np.asarray(rev_per_s, dtype=float) * 60.0


def kg_s_to_g_s(value_kg_s):
    """kg/s -> g/s."""
    return np.asarray(value_kg_s, dtype=float) * 1000.0


def g_s_to_kg_s(value_g_s):
    """g/s -> kg/s."""
    return np.asarray(value_g_s, dtype=float) / 1000.0


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_engine_config(path="configs/engine_m30b35.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class VETable:
    """Bilinear-interpolated volumetric-efficiency lookup, eta_v(rpm, map_kpa)."""

    def __init__(self, rpm_breakpoints, map_breakpoints_kpa, table):
        self.rpm_bp = np.asarray(rpm_breakpoints, dtype=float)
        self.map_bp = np.asarray(map_breakpoints_kpa, dtype=float)
        self.table = np.asarray(table, dtype=float)
        assert self.table.shape == (len(self.rpm_bp), len(self.map_bp)), (
            "VE table shape must match (len(rpm_breakpoints), len(map_breakpoints_kpa))"
        )

    @classmethod
    def from_config(cls, cfg):
        ve = cfg["ve_baseline"]
        return cls(ve["rpm_breakpoints"], ve["map_breakpoints_kpa"], ve["table"])

    def __call__(self, rpm, map_kpa):
        rpm = np.atleast_1d(np.asarray(rpm, dtype=float))
        map_kpa = np.atleast_1d(np.asarray(map_kpa, dtype=float))
        rpm_c = np.clip(rpm, self.rpm_bp[0], self.rpm_bp[-1])
        map_c = np.clip(map_kpa, self.map_bp[0], self.map_bp[-1])

        i = np.clip(np.searchsorted(self.rpm_bp, rpm_c) - 1, 0, len(self.rpm_bp) - 2)
        j = np.clip(np.searchsorted(self.map_bp, map_c) - 1, 0, len(self.map_bp) - 2)

        r0, r1 = self.rpm_bp[i], self.rpm_bp[i + 1]
        m0, m1 = self.map_bp[j], self.map_bp[j + 1]
        tr = np.where(r1 > r0, (rpm_c - r0) / (r1 - r0), 0.0)
        tm = np.where(m1 > m0, (map_c - m0) / (m1 - m0), 0.0)

        v00 = self.table[i, j]
        v01 = self.table[i, j + 1]
        v10 = self.table[i + 1, j]
        v11 = self.table[i + 1, j + 1]

        v0 = v00 * (1 - tm) + v01 * tm
        v1 = v10 * (1 - tm) + v11 * tm
        result = v0 * (1 - tr) + v1 * tr
        return result if result.size > 1 else float(result[0])


class PhysicsModel:
    """Speed-density baseline estimator used by the residual-AI model and C0-C5.

    mdot_air_g_s = (P_MAP_Pa * V_d_m3 * (RPM/2/60) * eta_v) / R_air * (1/T_IAT_K) * 1000

    Derivation note (factor of 120): for a 4-stroke engine there is one
    intake event per cylinder every 2 crank revolutions, i.e. the *engine*
    completes one full induction of V_d every 2 revolutions. Combining the
    ideal-gas mass in the swept volume (m = P*V/(R*T)) with an event rate of
    RPM/2 events-per-minute converted to per-second (i.e. RPM/120 with RPM in
    rev/min) reproduces the classical speed-density constant "120":

        mdot [kg/s] = P[Pa] * V_d[m^3] * eta_v * RPM[rev/min] / (120 * R_air * T[K])
    """

    def __init__(self, ve_table: VETable, displacement_m3: float, R_air: float,
                 bounded_fallback=(0.5, 260.0)):
        self.ve_table = ve_table
        self.V_d = float(displacement_m3)
        self.R_air = float(R_air)
        self.min_g_s, self.max_g_s = bounded_fallback

    @classmethod
    def from_config(cls, cfg):
        ve_table = VETable.from_config(cfg)
        return cls(
            ve_table=ve_table,
            displacement_m3=cfg["engine"]["displacement_m3"],
            R_air=cfg["constants"]["R_air_J_per_kgK"],
            bounded_fallback=(
                cfg["bounded_fallback"]["min_airflow_g_s"],
                cfg["bounded_fallback"]["max_airflow_g_s"],
            ),
        )

    def eta_v(self, rpm, map_kpa):
        return self.ve_table(rpm, map_kpa)

    def airflow_g_s(self, rpm, map_kpa, iat_c):
        """Speed-density mass airflow estimate in g/s (bounded)."""
        rpm = np.asarray(rpm, dtype=float)
        map_kpa = np.asarray(map_kpa, dtype=float)
        iat_c = np.asarray(iat_c, dtype=float)

        p_pa = kpa_to_pa(map_kpa)
        t_k = celsius_to_kelvin(iat_c)
        eta_v = np.asarray(self.eta_v(rpm, map_kpa), dtype=float)

        mdot_kg_s = (p_pa * self.V_d * rpm * eta_v) / (120.0 * self.R_air * t_k)
        mdot_g_s = kg_s_to_g_s(mdot_kg_s)
        return np.clip(mdot_g_s, self.min_g_s, self.max_g_s)


def worked_example():
    """Section 5: one worked numerical example with declared M30B35 values.

    Declared: displacement 3.430 L, MAP 90 kPa, IAT 25 C, RPM 3500, eta_v
    read from the baseline table (bilinear-interpolated).
    """
    cfg = load_engine_config()
    model = PhysicsModel.from_config(cfg)
    rpm, map_kpa, iat_c = 3500.0, 90.0, 25.0
    eta_v = model.eta_v(rpm, map_kpa)
    airflow = model.airflow_g_s(rpm, map_kpa, iat_c)
    return {
        "rpm": rpm,
        "map_kpa": map_kpa,
        "iat_c": iat_c,
        "eta_v": float(eta_v),
        "airflow_g_s": float(airflow),
    }


if __name__ == "__main__":
    example = worked_example()
    print("Worked speed-density example (Section 5):")
    for k, v in example.items():
        print(f"  {k}: {v}")
