"""Fuel delivery and delay-consistent lambda pseudo-reference models.

The hidden fuel-delivery plant and deployed calibration observer are separate.
The plant converts an ECU injector command into fuel reaching the cylinders.
The observer reconstructs fuel flow from the measured command using imperfect
nominal injector parameters and combines it with delayed wideband lambda.
Neither path receives hidden airflow as an input.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def injector_mass_flow_g_s(pw_ms, q_injector_g_s_at_ref_dp, ref_dp_kpa, dp_kpa,
                            dead_time_ms, n_cyl, rpm, saturation_pw_ms):
    """Estimate whole-engine fuel mass flow from a four-stroke injector command."""
    pw_ms = np.asarray(pw_ms, dtype=float)
    rpm = np.asarray(rpm, dtype=float)
    q_dp = q_injector_g_s_at_ref_dp * np.sqrt(np.clip(dp_kpa, 1.0, None) / ref_dp_kpa)
    effective_open_ms = np.clip(pw_ms - dead_time_ms, 0.0, saturation_pw_ms)
    mass_per_event_g = q_dp * effective_open_ms / 1000.0
    event_freq_hz = n_cyl * rpm / 120.0
    return np.clip(mass_per_event_g * event_freq_hz, 0.0, None)


def pulse_width_from_fuel_flow_ms(fuel_g_s, q_injector_g_s_at_ref_dp, ref_dp_kpa,
                                  dp_kpa, dead_time_ms, n_cyl, rpm,
                                  saturation_pw_ms):
    """Inverse nominal injector equation used by the simulated ECU fuel map."""
    q_dp = q_injector_g_s_at_ref_dp * np.sqrt(max(float(dp_kpa), 1.0) / ref_dp_kpa)
    event_freq_hz = max(n_cyl * float(rpm) / 120.0, 1e-9)
    effective_ms = float(fuel_g_s) / max(q_dp * event_freq_hz, 1e-12) * 1000.0
    return float(np.clip(effective_ms + dead_time_ms, 0.0,
                         saturation_pw_ms + dead_time_ms))


class DelayLine:
    """Exact integer-sample FIFO delay.

    ``delay_s == dt_s`` returns the preceding sample; zero delay returns the
    current sample. This definition avoids the former off-by-one behavior.
    """

    def __init__(self, delay_s: float, dt_s: float, initial_value: float = 0.0):
        if dt_s <= 0 or delay_s < 0:
            raise ValueError("dt_s must be positive and delay_s non-negative")
        self.delay_steps = int(round(delay_s / dt_s))
        self.buffer = deque([float(initial_value)] * self.delay_steps)

    def push_pop(self, value: float) -> float:
        value = float(value)
        if self.delay_steps == 0:
            return value
        self.buffer.append(value)
        return float(self.buffer.popleft())


class FirstOrderFuelFilm:
    """First-order approximation of retained/released wall-film fuel rate."""

    def __init__(self, fraction: float, tau_s: float, dt_s: float):
        if not (0.0 <= fraction < 1.0) or tau_s <= 0 or dt_s <= 0:
            raise ValueError("invalid fuel-film parameters")
        self.fraction = float(fraction)
        self.alpha = min(1.0, float(dt_s) / float(tau_s))
        self.retained_rate_g_s = 0.0

    def step(self, injected_g_s: float) -> float:
        film_input = max(float(injected_g_s), 0.0) * self.fraction
        self.retained_rate_g_s += (film_input - self.retained_rate_g_s) * self.alpha
        return max(float(injected_g_s) - film_input + self.retained_rate_g_s, 0.0)


class ActualFuelDeliveryPlant:
    """Hidden injector/fuel-film path used to calculate physical lambda."""

    def __init__(self, plant_cfg: dict, engine_cfg: dict, dt_s: float):
        inj = plant_cfg["injector"]
        self.q = inj["q_injector_g_s_at_ref_dp"] * inj.get("actual_flow_scale", 1.0)
        self.ref_dp = inj["ref_dp_kpa"]
        self.dead_time = inj["dead_time_ms"] + inj.get("actual_dead_time_offset_ms", 0.0)
        self.saturation = inj["saturation_pw_ms"]
        self.n_cyl = engine_cfg["engine"]["cylinders"]
        self.film = FirstOrderFuelFilm(
            inj["wall_film_fraction"], inj["wall_film_time_constant_s"], dt_s
        )

    def step(self, pw_ms: float, dp_kpa: float, rpm: float) -> float:
        injected = float(injector_mass_flow_g_s(
            pw_ms, self.q, self.ref_dp, dp_kpa, self.dead_time,
            self.n_cyl, rpm, self.saturation,
        ))
        return self.film.step(injected)


class FuelLambdaPath:
    """Deployed observer that creates the conditioned airflow pseudo-reference."""

    def __init__(self, plant_cfg: dict, engine_cfg: dict, dt_s: float):
        inj = plant_cfg["injector"]
        lam = plant_cfg["lambda_path"]
        observer = inj.get("observer", {})
        self.q_injector = inj["q_injector_g_s_at_ref_dp"] * observer.get("flow_scale", 1.0)
        self.ref_dp_kpa = inj["ref_dp_kpa"]
        self.dead_time_ms = inj["dead_time_ms"] + observer.get("dead_time_offset_ms", 0.0)
        self.saturation_pw_ms = inj["saturation_pw_ms"]
        self.n_cyl = engine_cfg["engine"]["cylinders"]
        self.stoich_afr = engine_cfg["constants"]["stoich_afr_gasoline"]
        self.film = FirstOrderFuelFilm(
            inj["wall_film_fraction"], inj["wall_film_time_constant_s"], dt_s
        )

        fixed_delay_s = lam["transport_delay_s"] + lam["sensor_delay_s"]
        self.delay = DelayLine(fixed_delay_s, dt_s, initial_value=0.0)
        tau = float(lam["controller_filter_s"])
        self.filter_alpha = 1.0 if tau <= 0 else dt_s / (tau + dt_s)
        self.filtered = 0.0
        self.samples_seen = 0
        settling_multiples = float(lam.get("filter_settling_time_constants", 3.0))
        self.warmup_samples = self.delay.delay_steps + int(np.ceil(settling_multiples * tau / dt_s))

    @property
    def fixed_delay_samples(self) -> int:
        return self.delay.delay_steps

    def step(self, pw_ms, dp_kpa, rpm, lambda_measured, lambda_valid,
             validity_flags: dict):
        fuel_injected = float(injector_mass_flow_g_s(
            pw_ms, self.q_injector, self.ref_dp_kpa, dp_kpa,
            self.dead_time_ms, self.n_cyl, rpm, self.saturation_pw_ms,
        ))
        fuel_delivered = self.film.step(fuel_injected)
        raw_pseudo = fuel_delivered * self.stoich_afr * float(lambda_measured)
        delayed_pseudo = self.delay.push_pop(raw_pseudo)
        self.filtered += (delayed_pseudo - self.filtered) * self.filter_alpha
        self.samples_seen += 1

        path_warm = self.samples_seen > self.warmup_samples
        blocking = any([
            validity_flags.get("fuel_cut", False),
            validity_flags.get("enrichment", False),
            validity_flags.get("cold_start", False),
            float(pw_ms) >= self.saturation_pw_ms,
            validity_flags.get("misfire", False),
            not lambda_valid,
            float(pw_ms) <= 0.0,
            validity_flags.get("dropout", False),
            validity_flags.get("sensor_fault", False),
            validity_flags.get("fault_code", 0) not in (0, None, False),
            not path_warm,
        ])
        return {
            "fuel_g_s": fuel_delivered,
            "raw_pseudo_reference_g_s": raw_pseudo,
            "pseudo_reference_g_s": float(self.filtered),
            "pseudo_reference_valid": bool(not blocking),
            "pseudo_path_warm": bool(path_warm),
            "pseudo_fixed_delay_samples": self.fixed_delay_samples,
            "pseudo_filter_alpha": float(self.filter_alpha),
        }


def condition_series_like_pseudo_reference(values, fixed_delay_samples: int,
                                           filter_alpha: float) -> np.ndarray:
    """Apply the observer's fixed delay and low-pass filter to a base estimate."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    delay = DelayLine(float(fixed_delay_samples), 1.0, initial_value=float(values[0]))
    filtered = float(values[0])
    out = np.empty_like(values)
    for i, value in enumerate(values):
        delayed = delay.push_pop(float(value))
        filtered += (delayed - filtered) * float(filter_alpha)
        out[i] = filtered
    return out
