"""Section 8 / Section 13 acceptance gate: run-level (not row-level)
disjointness of train/validation/threshold-selection/final-test run IDs,
including the calibration-shift threshold-selection vs. final-test
scenarios never sharing a run_id.
"""
import yaml

from generate_dataset import generate_all
from physics_model import load_engine_config
from prepare_splits import build_splits


def _tiny_config():
    with open("configs/data_generation.yaml") as f:
        cfg = yaml.safe_load(f)
    # Shrink drastically for a fast, structural-only test.
    for sc in cfg["scenarios"].values():
        sc["n_runs"] = 2
        sc["duration_s"] = 4
    return cfg


def test_run_ids_disjoint_across_split_roles(tmp_path):
    data_gen_cfg = _tiny_config()
    engine_cfg = load_engine_config()
    with open("configs/sensors.yaml") as f:
        sensors_cfg = yaml.safe_load(f)

    manifest = generate_all(data_gen_cfg, engine_cfg, sensors_cfg, out_dir=str(tmp_path / "raw"), quick=False)
    splits = build_splits(manifest)

    roles = ["train", "val", "threshold", "test"]
    id_sets = {r: set(splits.loc[splits.split_role == r, "run_id"]) for r in roles}
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            assert id_sets[a].isdisjoint(id_sets[b]), f"{a} and {b} share run_ids: {id_sets[a] & id_sets[b]}"


def test_calibration_shift_threshold_and_test_are_disjoint_scenarios(tmp_path):
    data_gen_cfg = _tiny_config()
    engine_cfg = load_engine_config()
    with open("configs/sensors.yaml") as f:
        sensors_cfg = yaml.safe_load(f)

    manifest = generate_all(data_gen_cfg, engine_cfg, sensors_cfg, out_dir=str(tmp_path / "raw"), quick=False)
    splits = build_splits(manifest)

    threshold_ids = set(splits.loc[splits.scenario == "calibration_shift_threshold", "run_id"])
    test_ids = set(splits.loc[splits.scenario == "calibration_shift_test", "run_id"])
    assert threshold_ids, "expected at least one calibration_shift_threshold run"
    assert test_ids, "expected at least one calibration_shift_test run"
    assert threshold_ids.isdisjoint(test_ids)


def test_no_nan_or_inf_in_generated_runs(tmp_path):
    import numpy as np
    import pandas as pd

    data_gen_cfg = _tiny_config()
    engine_cfg = load_engine_config()
    with open("configs/sensors.yaml") as f:
        sensors_cfg = yaml.safe_load(f)

    manifest = generate_all(data_gen_cfg, engine_cfg, sensors_cfg, out_dir=str(tmp_path / "raw"), quick=False)
    for path in manifest["path"]:
        df = pd.read_parquet(path)
        numeric = df.select_dtypes(include=[np.number])
        assert not numeric.isna().any().any(), f"NaN found in {path}"
        assert np.isfinite(numeric.to_numpy()).all(), f"Inf found in {path}"
