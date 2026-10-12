"""Matched one-state RLS scale comparator (roadmap Section 9, C5).

Standard recursive least squares for a single scalar parameter ``s`` in the
model ``y_k = phi_k * s + noise_k``, with a declared forgetting factor.
Per Section 10, C5 receives the SAME conditioned airflow pseudo-reference,
valid operating regions, parameter bounds, initial value, calibration
trajectories, delays, and final-test trajectory as C3 -- it does NOT
receive the scalar Fisher-information gate or the causal accept/rollback
mechanism. It may use its own declared forgetting factor and numerical
update rule (reported alongside all RLS-specific settings).
"""
from __future__ import annotations

import numpy as np


class OneStateRLS:
    def __init__(self, initial_value: float, bounds, per_update_limit: float,
                 rate_limit_per_s: float, forgetting_factor: float, p0: float = 10.0):
        self.s = float(initial_value)
        self.lo, self.hi = bounds
        self.per_update_limit = float(per_update_limit)
        self.rate_limit_per_s = float(rate_limit_per_s)
        self.lam = float(forgetting_factor)
        self.P = float(p0)
        if not 0.0 < self.lam <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        if self.P <= 0 or self.lo >= self.hi:
            raise ValueError("p0 must be positive and bounds must be ordered")

    def update(self, phi_k: float, y_k: float, dt_s: float) -> float:
        """One matched, bounded/rate-limited RLS update. Returns the new s."""
        phi_k = float(phi_k)
        if not np.isfinite(phi_k) or not np.isfinite(y_k) or dt_s < 0:
            raise ValueError("RLS inputs must be finite and dt_s non-negative")
        # Standard scalar RLS with forgetting factor lambda:
        #   K = P*phi / (lambda + phi^2 * P)
        #   s = s + K * (y - phi*s)
        #   P = (P - K*phi*P) / lambda
        denom = self.lam + (phi_k ** 2) * self.P
        if abs(denom) < 1e-12:
            return self.s
        K = (self.P * phi_k) / denom
        innovation = y_k - phi_k * self.s
        proposed_delta = K * innovation

        # Same bounding/rate-limiting contract as the proposed method's
        # Stage C, for a fair, matched comparison (Section 10).
        delta = float(np.clip(proposed_delta, -self.per_update_limit, self.per_update_limit))
        max_step = self.rate_limit_per_s * dt_s
        delta = float(np.clip(delta, -max_step, max_step))

        new_s = float(np.clip(self.s + delta, self.lo, self.hi))
        self.s = new_s
        # When the state update is clipped, inflate covariance relative to the
        # unconstrained RLS update instead of becoming falsely overconfident.
        unconstrained_p = (self.P - K * phi_k * self.P) / self.lam
        clipped = not np.isclose(delta, proposed_delta)
        self.P = max(unconstrained_p * (1.05 if clipped else 1.0), 1e-12)
        return self.s
