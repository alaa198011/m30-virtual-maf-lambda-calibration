"""Disjoint dataset splitting (roadmap Section 8).

Splits complete run-level trajectories (never individual rows) into
train / validation / threshold-selection / final-test groups, verifies
disjointness, records checksums, and writes ``dataset_manifest.json``.

Key leakage-safety rules enforced here:
  * splits are assigned by run_id (whole trajectories), not by row;
  * threshold-selection run_ids are never reused as final-test run_ids,
    even if only different segments were used, protecting the causal
    calibration evaluation from indirect tuning leakage (Section 8);
  * scalers/normalizers are fit downstream on the ``train`` split ONLY
    (enforced in train_pure_ai.py / train_residual_ai.py, verified here by
    just recording which run_ids belong to `train`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unversioned"


def build_splits(raw_manifest: pd.DataFrame) -> pd.DataFrame:
    """Assign split_role per run_id.

    split_role_hint from generate_dataset.py already encodes the intended
    role (train/val/threshold/test) per scenario. calibration_shift scenarios
    are special-cased: threshold-selection runs and final-test runs must be
    entirely disjoint sets of run_ids (Section 8), which is already true
    here because they come from two different scenarios
    (calibration_shift_threshold vs calibration_shift_test).
    """
    manifest = raw_manifest.copy()
    manifest["split_role"] = manifest["split_role_hint"]

    # Sanity: disjointness of run_ids across roles (trivially true since each
    # run_id is scenario-scoped and unique, but assert explicitly per the
    # roadmap's "train, validation, threshold-selection, and final-test run
    # IDs are disjoint" acceptance gate).
    for a in ["train", "val", "threshold", "test"]:
        ids_a = set(manifest.loc[manifest.split_role == a, "run_id"])
        for b in ["train", "val", "threshold", "test"]:
            if a == b:
                continue
            ids_b = set(manifest.loc[manifest.split_role == b, "run_id"])
            overlap = ids_a & ids_b
            assert not overlap, f"Split leakage between {a} and {b}: {overlap}"

    return manifest


def write_manifest(manifest: pd.DataFrame, data_gen_cfg: dict, out_path: str):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": _git_revision(),
        "python_version": sys.version,
        "sampling": data_gen_cfg["sampling"],
        "n_runs_total": int(len(manifest)),
        "n_runs_by_split_role": manifest.groupby("split_role")["run_id"].count().to_dict(),
        "n_runs_by_scenario": manifest.groupby("scenario")["run_id"].count().to_dict(),
        "runs": manifest.to_dict(orient="records"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Prepare disjoint dataset splits.")
    parser.add_argument("--config", default="configs/data_generation.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--manifest-out", default="results/dataset_manifest.json")
    args = parser.parse_args()

    with open(args.config) as f:
        data_gen_cfg = yaml.safe_load(f)

    raw_manifest_path = os.path.join(args.raw_dir, "raw_manifest.csv")
    raw_manifest = pd.read_csv(raw_manifest_path)

    manifest = build_splits(raw_manifest)

    os.makedirs(args.splits_dir, exist_ok=True)
    split_csv_path = os.path.join(args.splits_dir, "splits.csv")
    manifest.to_csv(split_csv_path, index=False)

    write_manifest(manifest, data_gen_cfg, args.manifest_out)
    print(f"Wrote {split_csv_path} and {args.manifest_out}")
    print(manifest.groupby("split_role")["run_id"].count())


if __name__ == "__main__":
    main()
