"""Small statistics helpers shared by evaluate.py / statistical_analysis.py
(roadmap Section 12: "Report effect sizes and 95% confidence intervals").
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0,
                  statistic=np.mean):
    """Nonparametric bootstrap CI for a statistic over run-level values
    (never raw, correlated time samples -- Section 12)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(statistic(values))
        return v, v, v
    rng = np.random.default_rng(seed)
    point = float(statistic(values))
    boots = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[b] = statistic(sample)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def cohens_d_paired(diffs) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2 or np.std(diffs, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def summarize(values, n_boot: int = 2000, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=float)
    mean, lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed, statistic=np.mean)
    return {
        "n": int(np.sum(~np.isnan(values))),
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "median": float(np.nanmedian(values)) if len(values) else float("nan"),
        "std": float(np.nanstd(values)) if len(values) else float("nan"),
    }
