"""Sensor corruption models (roadmap Section 6: "sensor quantization, noise,
bias, gain drift, lag, dropout, and stuck values").

Applied to the hidden plant's clean states to produce the estimator-visible
(and calibration-feedback-visible) signals. All signals modeled here are
direct simulated electrical interfaces (RPM, MAP, TPS, IAT, ECT, lambda,
battery voltage, injector pulse width) -- no OBD-II parameter or CAN
diagnostic PID is modeled anywhere in this repository (Section 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorState:
    """Per-signal running state needed for lag / drift / dropout / stuck."""
    filtered_value: float = None
    stuck_value: float = None
    is_stuck: bool = False
    elapsed_s: float = 0.0
    last_observed: float = None


class SensorChannel:
    """One corrupted sensor channel with first-order lag, noise, quantization,
    bias, linear gain drift, dropout, and stuck-value behavior."""

    def __init__(self, cfg: dict, dt_s: float, rng: np.random.Generator, name: str):
        self.cfg = cfg
        self.dt = dt_s
        self.rng = rng
        self.name = name
        self.lo, self.hi = cfg["range"]
        self.noise_std = cfg.get("noise_std", 0.0)
        self.q_step = cfg.get("quantization_step", 0.0)
        self.bias = cfg.get("bias", cfg.get("bias_kpa", 0.0))
        self.gain_drift_per_hour = cfg.get("gain_drift_per_hour", 0.0)
        self.lag_s = cfg.get("lag_s", 0.0)
        self.dropout_rate_per_s = cfg.get("dropout_rate_per_s", cfg.get("dropout_prob", 0.0))
        self.stuck_rate_per_s = cfg.get("stuck_rate_per_s", cfg.get("stuck_prob", 0.0))
        self.state = SensorState()
        self._elapsed_s = 0.0

    def _quantize(self, value):
        if self.q_step and self.q_step > 0:
            return np.round(value / self.q_step) * self.q_step
        return value

    def sample(self, clean_value: float):
        """Return ``(observed, dropout, stuck)``.

        A detected dropout holds the last valid electrical sample instead of
        returning a newly corrupted value.  The status bit remains explicit so
        Stage A can close the calibration gate.
        """
        self._elapsed_s += self.dt

        # Linear gain drift (multiplicative, grows with elapsed time)
        gain = 1.0 + self.gain_drift_per_hour * (self._elapsed_s / 3600.0)
        true_biased = clean_value * gain + self.bias

        # First-order lag (sensor/thermal dynamics)
        if self.state.filtered_value is None:
            self.state.filtered_value = true_biased
        if self.lag_s > 1e-9:
            alpha = self.dt / (self.lag_s + self.dt)
            self.state.filtered_value += (true_biased - self.state.filtered_value) * alpha
        else:
            self.state.filtered_value = true_biased

        # Additive Gaussian noise
        noisy = self.state.filtered_value + self.rng.normal(0.0, self.noise_std)

        # Stuck-value behavior (persists once triggered, for this run)
        stuck_p_step = 1.0 - np.exp(-self.stuck_rate_per_s * self.dt)
        if not self.state.is_stuck and self.rng.random() < stuck_p_step:
            self.state.is_stuck = True
            self.state.stuck_value = noisy
        if self.state.is_stuck:
            noisy = self.state.stuck_value

        observed = self._quantize(np.clip(noisy, self.lo, self.hi))

        dropout_p_step = 1.0 - np.exp(-self.dropout_rate_per_s * self.dt)
        dropout = self.rng.random() < dropout_p_step
        if dropout and self.state.last_observed is not None:
            observed = self.state.last_observed
        elif not dropout:
            self.state.last_observed = float(observed)
        return float(observed), bool(dropout), bool(self.state.is_stuck)


class SensorSuite:
    """Bundles all corrupted channels declared in configs/sensors.yaml."""

    def __init__(self, sensors_cfg: dict, dt_s: float, rng: np.random.Generator):
        self.channels = {
            name: SensorChannel(cfg, dt_s, rng, name)
            for name, cfg in sensors_cfg.items()
        }
        self.lambda_light_off_ect_proxy_c = sensors_cfg["lambda"].get(
            "light_off_ect_proxy_c", 60.0
        )

    def observe(self, clean: dict, ect_c: float, lambda_true_valid: bool):
        """clean: dict with keys rpm, map_kpa, tps_pct, iat_c, ect_c, lambda_true,
        battery_v_true, injector_pw_ms_true (as applicable).

        Returns a dict of observed/corrupted signals + flags.
        """
        out = {}
        rpm_obs, rpm_drop, rpm_stuck = self.channels["rpm"].sample(clean["rpm"])
        map_obs, map_drop, map_stuck = self.channels["map"].sample(clean["map_kpa"])
        tps_obs, tps_drop, tps_stuck = self.channels["tps"].sample(clean["tps_pct"])
        iat_obs, iat_drop, iat_stuck = self.channels["iat"].sample(clean["iat_c"])
        ect_obs, ect_drop, ect_stuck = self.channels["ect"].sample(clean["ect_c"])
        batt_obs, batt_drop, batt_stuck = self.channels["battery_v"].sample(
            clean.get("battery_v_true", self.channels["battery_v"].cfg.get("nominal_v", 13.8))
        )
        pw_obs, pw_drop, pw_stuck = self.channels["injector_pw"].sample(
            clean.get("injector_pw_ms_true", 0.0)
        )
        lam_obs, lam_drop, lam_stuck = self.channels["lambda"].sample(clean.get("lambda_true", 1.0))

        # Lambda sensor "valid" requires physical light-off AND upstream validity
        lambda_light_off = ect_c >= self.lambda_light_off_ect_proxy_c
        lambda_valid = bool(lambda_true_valid and lambda_light_off and not lam_drop and not lam_stuck)
        dropout_any = bool(
            rpm_drop or map_drop or tps_drop or iat_drop or ect_drop or batt_drop
            or pw_drop or lam_drop
        )
        stuck_any = bool(
            rpm_stuck or map_stuck or tps_stuck or iat_stuck or ect_stuck
            or batt_stuck or pw_stuck or lam_stuck
        )

        out.update(
            rpm=rpm_obs,
            map_kpa=map_obs,
            tps_pct=tps_obs,
            iat_c=iat_obs,
            ect_c=ect_obs,
            battery_v=batt_obs,
            injector_pw_ms=pw_obs,
            lambda_measured=lam_obs,
            lambda_valid=lambda_valid,
            dropout_any=dropout_any,
            stuck_any=stuck_any,
            sensor_fault_any=bool(dropout_any or stuck_any),
        )
        return out
