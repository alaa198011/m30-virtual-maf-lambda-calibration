"""Independent hidden virtual plant (roadmap Section 6).

Structurally different from ``physics_model.py``: a parametric VE surface
with its own shape/peak location, manifold filling-and-emptying dynamics,
throttle actuator lag, pressure pulsation, intake-air/coolant thermal
dynamics, injector saturation/wall-film behavior, and a driver/load
protocol. This plant is the sole source of "hidden reference airflow" and
must never be imported by the deployed estimator's inference path.

Declared mismatch families (Section 6 "Mismatch design"):
  - VE-scale mismatch (primary family)
  - VE-shape / peak-location mismatch (primary family, folded into ve_scale
    tests via ``ve_shape_shift``)
  - MAP gain/offset mismatch (secondary, deferred)
  - manifold time-constant mismatch (secondary, deferred)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physics_model import celsius_to_kelvin, kpa_to_pa, kg_s_to_g_s


@dataclass
class MismatchConfig:
    name: str = "nominal"
    ve_scale: float = 1.0
    ve_shape_shift: float = 0.0
    map_gain: float = 1.0
    map_offset_kpa: float = 0.0
    manifold_tau_scale: float = 1.0
    primary: bool = True


@dataclass
class DriverCommand:
    """One instant of driver/load protocol commands."""
    tps_command_pct: float
    load_torque_proxy: float = 0.0  # unused placeholder for future BSFC work


class HiddenVESurface:
    """Parametric hidden VE surface, structurally different from VETable.

    eta_v(rpm, map_kpa) = eta_max * f_rpm(rpm) * f_map(map_kpa)

    f_rpm is a shifted Gaussian bump (peak location = rpm_peak + shape_shift);
    f_map is a saturating power law in map/map_ref. This functional form is
    intentionally NOT a bilinear table, so it cannot trivially coincide with
    the estimator's VETable regardless of parameter choice.
    """

    def __init__(self, eta_max, rpm_peak, rpm_width, map_ref_kpa, map_exponent,
                 ve_scale=1.0, ve_shape_shift=0.0):
        self.eta_max = eta_max * ve_scale
        self.rpm_peak = rpm_peak + ve_shape_shift
        self.rpm_width = rpm_width
        self.map_ref_kpa = map_ref_kpa
        self.map_exponent = map_exponent

    def __call__(self, rpm, map_kpa):
        rpm = np.asarray(rpm, dtype=float)
        map_kpa = np.asarray(map_kpa, dtype=float)
        f_rpm = np.exp(-0.5 * ((rpm - self.rpm_peak) / self.rpm_width) ** 2)
        f_rpm = 0.55 + 0.45 * f_rpm  # keep a nonzero floor at idle/redline
        f_map = np.clip(map_kpa / self.map_ref_kpa, 0.05, 1.3) ** self.map_exponent
        eta = self.eta_max * f_rpm * f_map
        return np.clip(eta, 0.05, 0.98)


@dataclass
class PlantState:
    map_kpa: float
    tps_actual_pct: float
    iat_c: float
    ect_c: float
    wall_film_g: float = 0.0


class VirtualEnginePlant:
    """Independent hidden plant producing clean (pre-sensor-corruption) states."""

    def __init__(self, engine_cfg: dict, plant_cfg: dict, mismatch: MismatchConfig,
                 dt_s: float, rng: np.random.Generator):
        self.engine_cfg = engine_cfg
        self.plant_cfg = plant_cfg
        self.mismatch = mismatch
        self.dt = dt_s
        self.rng = rng

        ve_cfg = plant_cfg["ve_surface"]
        self.ve_surface = HiddenVESurface(
            eta_max=ve_cfg["eta_max"],
            rpm_peak=ve_cfg["rpm_peak"],
            rpm_width=ve_cfg["rpm_width"],
            map_ref_kpa=ve_cfg["map_ref_kpa"],
            map_exponent=ve_cfg["map_exponent"],
            ve_scale=mismatch.ve_scale,
            ve_shape_shift=mismatch.ve_shape_shift,
        )

        self.V_d = engine_cfg["engine"]["displacement_m3"]
        self.R_air = engine_cfg["constants"]["R_air_J_per_kgK"]
        self.n_cyl = engine_cfg["engine"]["cylinders"]

        man = plant_cfg["manifold"]
        self.manifold_volume_m3 = man["volume_m3"]
        self.manifold_tau_s = man["time_constant_s"] * mismatch.manifold_tau_scale
        self.pulsation_amp_kpa = man["pulsation_amplitude_kpa"]
        self.pulsation_freq_per_rpm = man["pulsation_freq_hz_per_rpm"]

        self.throttle_tau_s = plant_cfg["throttle"]["actuator_time_constant_s"]

        th = plant_cfg["thermal"]
        self.iat_tau_s = th["iat_time_constant_s"]
        self.ect_tau_s = th["ect_time_constant_s"]
        self.ambient_c = th["ambient_c"]
        self.ect_operating_c = th["ect_operating_c"]

        # Runs represent a declared bench/dyno test protocol that begins
        # after a minimal warm-up period (engine already above the Stage-A
        # validity ECT threshold at t=0). Dedicated cold-start segments in
        # the "boundary"/"corner" scenarios force ect_c down explicitly
        # (see generate_dataset.py) to still exercise cold-start gate
        # blocking without requiring every run to spend its first ~90 s cold.
        self.state = PlantState(
            map_kpa=self.ambient_c_to_idle_map(),
            tps_actual_pct=0.0,
            iat_c=self.ambient_c,
            ect_c=self.ect_operating_c - 5.0,
        )
        self._t = 0.0

    def ambient_c_to_idle_map(self):
        return 35.0

    def _throttle_plane_flow_proxy(self, tps_actual_pct, rpm):
        """A monotone proxy for air entering the throttle plane, used only to
        drive manifold filling dynamics -- not the reported cylinder charge
        airflow (which the manifold dynamics make distinct during transients,
        per Section 6 "cylinder-charge airflow distinct from throttle-plane
        airflow")."""
        atmo_kpa = 101.325 * self.mismatch.map_gain + self.mismatch.map_offset_kpa
        opening = np.clip(tps_actual_pct / 100.0, 0.0, 1.0) ** 0.7
        return atmo_kpa * (0.15 + 0.85 * opening)

    def step(self, tps_command_pct, rpm, load_proxy=0.0):
        """Advance the hidden plant by one dt and return clean states + hidden
        cylinder-charge airflow in g/s."""
        dt = self.dt
        st = self.state

        # Throttle actuator lag: actual TPS lags commanded TPS (first-order)
        st.tps_actual_pct += (tps_command_pct - st.tps_actual_pct) * (dt / self.throttle_tau_s)
        st.tps_actual_pct = float(np.clip(st.tps_actual_pct, 0.0, 100.0))

        # Manifold filling/emptying: MAP relaxes toward a throttle-plane
        # target with the plant's own time constant, plus pulsation ripple.
        map_target_kpa = self._throttle_plane_flow_proxy(st.tps_actual_pct, rpm)
        st.map_kpa += (map_target_kpa - st.map_kpa) * (dt / self.manifold_tau_s)
        pulsation = self.pulsation_amp_kpa * np.sin(
            2 * np.pi * self.pulsation_freq_per_rpm * rpm * self._t
        )
        map_kpa_observed_clean = float(np.clip(st.map_kpa + pulsation, 10.0, 110.0))

        # Thermal dynamics: IAT rises slightly with load, ECT rises to
        # operating temperature and holds.
        iat_target = self.ambient_c + 8.0 * np.clip(load_proxy, 0.0, 1.0)
        st.iat_c += (iat_target - st.iat_c) * (dt / self.iat_tau_s)
        st.ect_c += (self.ect_operating_c - st.ect_c) * (dt / self.ect_tau_s)

        # Hidden cylinder-charge airflow via the plant's OWN VE surface,
        # evaluated on the (unpulsed, dynamically-lagged) manifold pressure
        # so that during transients cylinder charge differs from the
        # instantaneous throttle-plane value.
        eta_v_hidden = self.ve_surface(rpm, st.map_kpa)
        p_pa = kpa_to_pa(st.map_kpa)
        t_k = celsius_to_kelvin(st.iat_c)
        mdot_kg_s = (p_pa * self.V_d * rpm * eta_v_hidden) / (120.0 * self.R_air * t_k)
        hidden_airflow_g_s = float(np.clip(kg_s_to_g_s(mdot_kg_s), 0.0, 400.0))

        self._t += dt

        return {
            "map_kpa": map_kpa_observed_clean,
            "tps_pct": st.tps_actual_pct,
            "iat_c": float(st.iat_c),
            "ect_c": float(st.ect_c),
            "maf_reference_g_s": hidden_airflow_g_s,
            "rpm": float(rpm),
        }
