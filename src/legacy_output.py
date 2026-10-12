"""Legacy output emulation (roadmap Sections 3, 11 E7, 14, 15).

Software-only conversion of estimated airflow into a declared voltage or
frequency profile. This is NOT physical ECU compatibility -- it is a
mathematical transfer function plus slew-rate limiting and simulated ADC
quantization, evaluated on the development host. All timing figures this
module reports are explicitly HOST TIMING, never MCU timing (Article 2
reserves target-hardware measurement).
"""
from __future__ import annotations

import time

import numpy as np


class LegacyOutputProfile:
    def __init__(self, profile_cfg: dict):
        self.cfg = profile_cfg
        self.kind = profile_cfg["kind"]
        if self.kind == "voltage":
            self.x_bp = np.asarray(profile_cfg["airflow_breakpoints_g_s"], dtype=float)
            self.y_bp = np.asarray(profile_cfg["voltage_breakpoints_v"], dtype=float)
            self.y_min, self.y_max = profile_cfg["v_min"], profile_cfg["v_max"]
            self.slew_limit = profile_cfg["slew_rate_limit_v_per_s"]
            self.q_bits = profile_cfg.get("quantization_bits")
        elif self.kind == "frequency":
            self.f_min = profile_cfg["f_min_hz"]
            self.k = profile_cfg["k_hz_per_sqrt_gs"]
            self.f_max = profile_cfg["f_max_hz"]
            self.slew_limit = profile_cfg["slew_rate_limit_hz_per_s"]
        else:
            raise ValueError(f"unknown legacy output kind: {self.kind}")

        if self.kind == "voltage" and profile_cfg.get("monotonic_required", True):
            assert np.all(np.diff(self.y_bp) >= 0), "voltage transfer function must be monotonic"

    def transfer(self, airflow_g_s: np.ndarray) -> np.ndarray:
        airflow_g_s = np.asarray(airflow_g_s, dtype=float)
        if self.kind == "voltage":
            raw = np.interp(airflow_g_s, self.x_bp, self.y_bp)
            return np.clip(raw, self.y_min, self.y_max)
        else:
            raw = self.f_min + self.k * np.sqrt(np.clip(airflow_g_s, 0.0, None))
            return np.clip(raw, self.f_min, self.f_max)

    def inverse_transfer(self, output: np.ndarray) -> np.ndarray:
        output = np.asarray(output, dtype=float)
        if self.kind == "voltage":
            # x_bp assumed increasing in airflow, y_bp non-decreasing -> invert via interp on y
            return np.interp(output, self.y_bp, self.x_bp)
        else:
            f = np.clip(output, self.f_min, self.f_max)
            return ((f - self.f_min) / max(self.k, 1e-9)) ** 2

    def apply_slew_and_quantize(self, raw_series: np.ndarray, dt_s: float) -> np.ndarray:
        raw_series = np.asarray(raw_series, dtype=float)
        out = np.empty_like(raw_series)
        out[0] = raw_series[0]
        max_step = self.slew_limit * dt_s
        for k in range(1, len(raw_series)):
            delta = np.clip(raw_series[k] - out[k - 1], -max_step, max_step)
            out[k] = out[k - 1] + delta
        if self.kind == "voltage" and self.q_bits:
            levels = 2 ** self.q_bits
            step = (self.y_max - self.y_min) / levels
            out = self.y_min + np.round((out - self.y_min) / step) * step
            out = np.clip(out, self.y_min, self.y_max)
        return out

    def saturation_events(self, out_series: np.ndarray) -> int:
        if self.kind == "voltage":
            return int(np.sum((out_series <= self.y_min + 1e-9) | (out_series >= self.y_max - 1e-9)))
        return int(np.sum((out_series <= self.f_min + 1e-9) | (out_series >= self.f_max - 1e-9)))

    def monotonicity_violations(self) -> int:
        if self.kind != "voltage":
            return 0
        return int(np.sum(np.diff(self.y_bp) < 0))


def evaluate_profile(profile: LegacyOutputProfile, airflow_series: np.ndarray, dt_s: float) -> dict:
    raw = profile.transfer(airflow_series)
    shaped = profile.apply_slew_and_quantize(raw, dt_s)
    recovered_airflow = profile.inverse_transfer(shaped)
    tf_error = float(np.sqrt(np.mean((recovered_airflow - airflow_series) ** 2)))
    slew_rate_error = float(np.max(np.abs(np.diff(shaped)) / max(dt_s, 1e-9)) - profile.slew_limit)
    return {
        "transfer_function_rmse_g_s": tf_error,
        "monotonicity_violations": profile.monotonicity_violations(),
        "saturation_events": profile.saturation_events(shaped),
        "slew_rate_error": max(slew_rate_error, 0.0),
    }


def model_resource_footprint(artifact: dict) -> dict:
    """Parameter count / model bytes / MAC estimate for a trained MLP
    artifact (Section 11 E7, Section 9 resource budget)."""
    model = artifact["model"]
    n_params = int(sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_))
    scaler = artifact.get("scaler")
    scaler_params = 0
    if scaler is not None:
        for name in ("mean_", "scale_"):
            value = getattr(scaler, name, None)
            if value is not None:
                scaler_params += int(np.asarray(value).size)
    model_bytes = (n_params + scaler_params) * 4  # float32 deployable payload
    # One forward pass MAC estimate: sum over layers of (in_features * out_features)
    macs = int(sum(c.shape[0] * c.shape[1] for c in model.coefs_))
    return {"n_parameters": n_params, "scaler_parameters": scaler_params,
            "model_bytes": model_bytes, "mac_estimate": macs}


def host_timing_inference(artifact: dict, X: np.ndarray, n_repeats: int = 200) -> dict:
    """HOST timing only (Section 11 E7 / Section 14 host_timing_label). Never
    to be reported or interpreted as MCU/embedded timing."""
    model = artifact["model"]
    scaler = artifact["scaler"]
    Xs = scaler.transform(X[:1])
    # warm-up
    for _ in range(5):
        model.predict(Xs)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(Xs)
    elapsed = time.perf_counter() - t0
    return {
        "host_timing_only_not_mcu_timing": True,
        "host_mean_inference_time_ms": (elapsed / n_repeats) * 1000.0,
    }
