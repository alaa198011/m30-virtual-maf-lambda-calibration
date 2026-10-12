"""Statistical plan (roadmap Section 12).

Aggregates over independent runs and seeds (never over correlated time
samples as if i.i.d.), uses paired run-level differences, reports effect
sizes and 95% confidence intervals, uses a paired t-test only if its
assumptions are reasonable (Shapiro-Wilk normality of the paired
differences), otherwise a Wilcoxon signed-rank test, and corrects families
of repeated comparisons with Holm's method.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from uncertainty import bootstrap_ci, cohens_d_paired


def paired_test(x, y):
    """x, y: matched arrays (same run_id order). Returns a result dict."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 3:
        return {"n": n, "test": "insufficient_data", "statistic": np.nan, "p_value": np.nan,
                "mean_diff": float(np.mean(diff)) if n else np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "effect_size": np.nan}

    if np.std(diff) < 1e-12:
        shapiro_p = 1.0
    else:
        try:
            _stat, shapiro_p = stats.shapiro(diff)
        except Exception:
            shapiro_p = 0.0

    if shapiro_p > 0.05:
        test_name = "paired_t"
        statistic, p_value = stats.ttest_rel(x, y)
    else:
        test_name = "wilcoxon_signed_rank"
        try:
            statistic, p_value = stats.wilcoxon(x, y)
        except ValueError:
            statistic, p_value = np.nan, 1.0

    mean_diff, ci_lo, ci_hi = bootstrap_ci(diff, n_boot=5000)
    effect_size = cohens_d_paired(diff)

    return {"n": n, "test": test_name, "shapiro_p": shapiro_p, "statistic": float(statistic),
            "p_value": float(p_value), "mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "effect_size": effect_size}


def holm_correction(p_values):
    """Holm step-down correction. Returns adjusted p-values in original order."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def build_model_comparisons(per_run_metrics: pd.DataFrame, comparisons):
    rows = []
    # average across AI seeds per (run_id, model_family) first, per Section 12
    per_run_metrics = per_run_metrics.copy()
    per_run_metrics["model_family"] = per_run_metrics["model"].str.replace(r"_seed\d+", "", regex=True)
    family_means = (
        per_run_metrics.groupby(["run_id", "model_family"])["nrmse"].mean().reset_index()
    )
    wide = family_means.pivot(index="run_id", columns="model_family", values="nrmse")

    for a, b in comparisons:
        if a not in wide.columns or b not in wide.columns:
            continue
        merged = wide[[a, b]].dropna()
        if len(merged) < 2:
            continue
        result = paired_test(merged[a].to_numpy(), merged[b].to_numpy())
        result.update({"comparison": f"{a}_vs_{b}", "metric": "nrmse"})
        rows.append(result)
    return rows


def build_calibration_comparisons(drift_recovery: pd.DataFrame, comparisons):
    rows = []
    wide = drift_recovery.pivot_table(index="run_id", columns="mode", values="holdout_nrmse")
    for a, b in comparisons:
        if a not in wide.columns or b not in wide.columns:
            continue
        merged = wide[[a, b]].dropna()
        if len(merged) < 2:
            continue
        result = paired_test(merged[a].to_numpy(), merged[b].to_numpy())
        result.update({"comparison": f"{a}_vs_{b}", "metric": "holdout_nrmse"})
        rows.append(result)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run the predeclared statistical plan (Section 12).")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        exp_cfg = yaml.safe_load(f)
    stats_cfg = exp_cfg["statistics"]
    comparisons = [tuple(c) for c in stats_cfg["comparisons"]]

    model_comparisons = [c for c in comparisons if c[0] in ("residual_ai", "pure_ai", "physics_only")]
    calib_comparisons = [c for c in comparisons if c[0].startswith("C")]

    rows = []
    per_run_path = os.path.join(args.results_dir, "per_run_model_metrics.csv")
    if os.path.exists(per_run_path):
        per_run_metrics = pd.read_csv(per_run_path)
        rows.extend(build_model_comparisons(per_run_metrics, model_comparisons))

    drift_path = os.path.join(args.results_dir, "drift_recovery_metrics.csv")
    if os.path.exists(drift_path):
        drift_recovery = pd.read_csv(drift_path)
        rows.extend(build_calibration_comparisons(drift_recovery, calib_comparisons))

    result_df = pd.DataFrame(rows)
    if len(result_df):
        result_df["p_value_holm_adjusted"] = holm_correction(result_df["p_value"].fillna(1.0).to_numpy())

    out_path = os.path.join(args.results_dir, "statistical_tests.csv")
    result_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(result_df)} comparisons)")


if __name__ == "__main__":
    main()
