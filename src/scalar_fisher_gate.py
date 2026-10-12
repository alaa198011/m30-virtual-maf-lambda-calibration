"""Scalar Fisher-information gate (roadmap Section 9, Stage B).

Primary implementation contract (frozen, ``configs/calibration.yaml``):
information_metric = scalar_fisher, a fixed valid-sample count N per
window, a declared pseudo-reference error-variance source/value, and a
frozen threshold I_s,min. No alternative metric selector is permitted in
the primary experiment; sensitivity energy, accumulated regressor energy,
persistent-excitation status, and RPM-MAP coverage do not gate updates.

Model: the calibrated prediction is ``y_hat = s * phi``, where ``phi`` is
the frozen base-airflow estimate (residual-AI output prior to any online
scale correction) and ``s`` is the scalar being calibrated. For a linear-
Gaussian observation ``y = s*phi + noise``, ``noise ~ N(0, sigma^2_pseudo)``,
the Fisher information for ``s`` contributed by one sample is
``phi^2 / sigma^2_pseudo`` (since d/ds (s*phi) = phi), and the information
of a window of N samples is the sum over those samples:

    I_s(W) = sum_{k in W} phi_k^2 / sigma^2_pseudo

The gate opens only when I_s(W) >= I_s,min, evaluated over the most recent
N Stage-A-valid samples.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def scalar_fisher_information(phi_values, sigma2_pseudo: float) -> float:
    if sigma2_pseudo <= 0:
        raise ValueError("sigma2_pseudo must be positive")
    phi_values = np.asarray(phi_values, dtype=float)
    if not np.isfinite(phi_values).all():
        raise ValueError("phi_values must be finite")
    return float(np.sum(phi_values ** 2) / sigma2_pseudo)


class ScalarFisherGate:
    """Rolling window of the most recent N Stage-A-valid ``phi_k`` samples."""

    def __init__(self, n_valid: int, sigma2: float, threshold: float):
        self.n_valid = int(n_valid)
        self.sigma2 = float(sigma2)
        self.threshold = float(threshold)
        if self.n_valid <= 0 or self.sigma2 <= 0 or self.threshold < 0:
            raise ValueError("n_valid and sigma2 must be positive; threshold must be non-negative")
        self._buffer = deque(maxlen=self.n_valid)

    def push_valid_sample(self, phi_k: float):
        self._buffer.append(float(phi_k))

    def reset(self):
        self._buffer.clear()

    @property
    def window_full(self) -> bool:
        return len(self._buffer) == self.n_valid

    def evaluate(self):
        """Returns (gate_open: bool, I_s: float, n_in_window: int)."""
        if not self.window_full:
            return False, 0.0, len(self._buffer)
        i_s = scalar_fisher_information(list(self._buffer), self.sigma2)
        return (i_s >= self.threshold), i_s, len(self._buffer)
