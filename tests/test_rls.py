"""One-state RLS comparator (roadmap Section 9, C5): must converge toward
the true scale under simulated linear feedback, and must respect the same
bounds/rate limits as the proposed method (Section 10 "matched" contract).
"""
import numpy as np

from classical_rls import OneStateRLS


def test_rls_converges_to_true_scale():
    rng = np.random.default_rng(0)
    true_s = 0.88
    rls = OneStateRLS(initial_value=1.0, bounds=(0.5, 1.5), per_update_limit=0.05,
                       rate_limit_per_s=1.0, forgetting_factor=0.995)
    dt_s = 0.02
    for _ in range(4000):
        phi_k = rng.uniform(10.0, 60.0)
        y_k = true_s * phi_k + rng.normal(0, 0.5)
        rls.update(phi_k, y_k, dt_s)
    assert abs(rls.s - true_s) < 0.03


def test_rls_respects_bounds():
    rng = np.random.default_rng(1)
    rls = OneStateRLS(initial_value=1.0, bounds=(0.9, 1.1), per_update_limit=0.5,
                       rate_limit_per_s=100.0, forgetting_factor=0.9)
    for _ in range(500):
        phi_k = rng.uniform(10.0, 60.0)
        y_k = 3.0 * phi_k  # a wildly out-of-bounds "true" scale
        rls.update(phi_k, y_k, dt_s=0.02)
        assert 0.9 - 1e-9 <= rls.s <= 1.1 + 1e-9


def test_rls_respects_rate_limit_per_step():
    rls = OneStateRLS(initial_value=1.0, bounds=(0.0, 2.0), per_update_limit=1.0,
                       rate_limit_per_s=0.01, forgetting_factor=0.9)
    dt_s = 0.02
    max_step = 0.01 * dt_s
    prev = rls.s
    for _ in range(20):
        rls.update(phi_k=50.0, y_k=200.0, dt_s=dt_s)  # push hard toward a big innovation
        assert abs(rls.s - prev) <= max_step + 1e-9
        prev = rls.s
