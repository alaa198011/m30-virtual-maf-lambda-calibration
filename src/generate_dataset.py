"""Dataset generation (roadmap Section 8).

Builds complete time-series *runs* (not independent rows) from the hidden
virtual plant + sensor corruption + fuel/lambda path, for every declared
scenario in ``configs/data_generation.yaml``. Each run is written as its
own parquet file under ``data/raw/<scenario>/run_XXXX.parquet`` with the
declared minimum schema, plus a raw manifest CSV consumed by
``prepare_splits.py``.

Design note on RPM: this is a bench-style (engine-in-the-loop) simulation.
RPM is treated as an independently commanded input (as it would be on an
engine dynamometer or a bench rig with a programmable RPM source), while
TPS is the pedal/throttle command that drives the plant's own throttle
actuator and manifold dynamics. This keeps boundary/corner/transient
conditions exactly reproducible, at the cost of not modeling vehicle
inertia/gearing -- an explicitly declared simplification (see
``manuscript/limitations.md``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import yaml

from physics_model import PhysicsModel, load_engine_config
from virtual_engine import VirtualEnginePlant, MismatchConfig
from sensor_models import SensorSuite
from fuel_lambda_path import (
    ActualFuelDeliveryPlant,
    FuelLambdaPath,
    pulse_width_from_fuel_flow_ms,
)

SCHEMA_COLUMNS = [
    "timestamp_s", "run_id", "seed", "scenario", "state",
    "rpm", "map_kpa", "tps_pct", "iat_c", "ect_c",
    "lambda_measured", "lambda_valid", "injector_pw_ms", "battery_v",
    "maf_reference_g_s", "maf_physics_g_s",
    "fault_code", "calibration_allowed",
]

# Hidden-reference / diagnostic columns that must NEVER be fed to a deployed
# estimator. Used by the leakage guard in train_pure_ai.py / train_residual_ai.py
# and by tests/test_feature_leakage.py.
FORBIDDEN_DEPLOYED_FEATURES = {
    "maf_reference_g_s", "maf_physics_g_s", "lambda_measured", "lambda_valid",
    "injector_pw_ms", "run_id", "seed", "scenario", "state", "fault_code",
    "calibration_allowed", "timestamp_s", "lambda_true", "actual_fuel_g_s",
    "ecu_feedforward_airflow_g_s", "pseudo_reference_g_s",
}


def _smooth_random_walk(rng, n, lo, hi, step_std, x0=None):
    x0 = lo + (hi - lo) * 0.3 if x0 is None else x0
    x = np.empty(n)
    x[0] = np.clip(x0, lo, hi)
    for k in range(1, n):
        x[k] = np.clip(x[k - 1] + rng.normal(0, step_std), lo, hi)
    return x


def _classify_state(tps, tps_prev, rpm):
    dtps = tps - tps_prev
    if tps < 8 and rpm < 1100:
        return "idle"
    if dtps > 3.0:
        return "acceleration" if tps > 15 else "tip_in"
    if dtps < -3.0:
        return "deceleration" if tps < 10 else "tip_out"
    if tps > 65:
        return "high_load"
    return "cruise"


def _driver_protocol(kind, n, dt, rng, rpm_range, params=None):
    """Returns per-step dict of arrays: tps_cmd, rpm_cmd, state, load_proxy, flags."""
    t = np.arange(n) * dt
    tps_cmd = np.zeros(n)
    rpm_cmd = np.zeros(n)
    state = np.empty(n, dtype=object)
    fuel_cut = np.zeros(n, dtype=bool)
    enrichment = np.zeros(n, dtype=bool)
    misfire = np.zeros(n, dtype=bool)

    rpm_lo, rpm_hi = rpm_range

    if kind in ("nominal", "mismatch"):
        tps_cmd = _smooth_random_walk(rng, n, 4, 78, 3.0)
        rpm_cmd = _smooth_random_walk(rng, n, max(rpm_lo, 850), min(rpm_hi, 4600), 60.0)
        prev = tps_cmd[0]
        for k in range(n):
            state[k] = _classify_state(tps_cmd[k], prev, rpm_cmd[k])
            prev = tps_cmd[k]

    elif kind == "boundary":
        segs = ["idle_min", "redline", "wot", "vacuum_min", "iat_extreme", "ect_extreme"]
        seg_len = max(1, n // len(segs))
        for i, seg in enumerate(segs):
            s, e = i * seg_len, min(n, (i + 1) * seg_len)
            if seg == "idle_min":
                tps_cmd[s:e], rpm_cmd[s:e] = 2.0, 720
            elif seg == "redline":
                tps_cmd[s:e], rpm_cmd[s:e] = 95.0, rpm_hi - 50
            elif seg == "wot":
                tps_cmd[s:e], rpm_cmd[s:e] = 98.0, 4500
            elif seg == "vacuum_min":
                tps_cmd[s:e], rpm_cmd[s:e] = 0.0, 3200
                fuel_cut[s:e] = True
            elif seg == "iat_extreme":
                tps_cmd[s:e], rpm_cmd[s:e] = 40.0, 2500
            elif seg == "ect_extreme":
                tps_cmd[s:e], rpm_cmd[s:e] = 55.0, 3000
            state[s:e] = seg
        state[state == None] = "boundary"  # noqa: E711 (fill any tail)

    elif kind == "corner":
        segs = ["lugging", "engine_braking", "cold_high_load", "hot_wot", "redline_lowmap"]
        seg_len = max(1, n // len(segs))
        for i, seg in enumerate(segs):
            s, e = i * seg_len, min(n, (i + 1) * seg_len)
            if seg == "lugging":
                tps_cmd[s:e], rpm_cmd[s:e] = 85.0, 900
            elif seg == "engine_braking":
                tps_cmd[s:e], rpm_cmd[s:e] = 0.0, rpm_hi - 200
                fuel_cut[s:e] = True
            elif seg == "cold_high_load":
                tps_cmd[s:e], rpm_cmd[s:e] = 90.0, 3800
            elif seg == "hot_wot":
                tps_cmd[s:e], rpm_cmd[s:e] = 99.0, 4200
            elif seg == "redline_lowmap":
                tps_cmd[s:e], rpm_cmd[s:e] = 12.0, rpm_hi - 100
            state[s:e] = seg
        state[state == None] = "corner"  # noqa: E711

    elif kind == "transient":
        plan = [
            ("idle", 0.10, 5.0, 850),
            ("tip_in", 0.03, 40.0, 1400),
            ("acceleration", 0.20, 75.0, 3600),
            ("cruise", 0.25, 35.0, 2400),
            ("high_load", 0.17, 88.0, 4200),
            ("tip_out", 0.03, 15.0, 2600),
            ("deceleration", 0.22, 0.0, 1400),
        ]
        idx = 0
        for name, frac, tps_target, rpm_target in plan:
            seg_len = max(1, int(round(n * frac)))
            s, e = idx, min(n, idx + seg_len)
            ramp = np.linspace(tps_cmd[s - 1] if s > 0 else tps_target, tps_target, e - s)
            tps_cmd[s:e] = ramp
            rpm_ramp = np.linspace(rpm_cmd[s - 1] if s > 0 else rpm_target, rpm_target, e - s)
            rpm_cmd[s:e] = rpm_ramp
            state[s:e] = name
            if name == "deceleration":
                fuel_cut[s:e] = rpm_cmd[s:e] > 1200
            if name == "tip_in":
                enrichment[s:e] = True
            idx = e
        if idx < n:
            tps_cmd[idx:n] = tps_cmd[idx - 1]
            rpm_cmd[idx:n] = rpm_cmd[idx - 1]
            state[idx:n] = "idle"

    elif kind == "calibration_shift":
        tps_cmd = _smooth_random_walk(rng, n, 15, 75, 4.0)
        rpm_cmd = _smooth_random_walk(rng, n, max(rpm_lo, 1200), min(rpm_hi, 4200), 80.0)
        third = n // 3
        state[:third] = "pre_drift"
        state[third:2 * third] = "adaptation"
        state[2 * third:] = "holdout"

    elif kind == "low_information":
        tps_cmd[:] = 6.0 + rng.normal(0, 0.3, n).cumsum() * 0.0  # deliberately flat
        rpm_cmd[:] = 780.0
        state[:] = "low_info"

    else:
        raise ValueError(f"unknown driver protocol kind: {kind}")

    state = np.where(state == None, "unclassified", state)  # noqa: E711
    load_proxy = np.clip(tps_cmd / 100.0, 0.0, 1.0)
    misfire = misfire | (rng.random(n) < 1e-5)
    return {
        "t": t, "tps_cmd": tps_cmd, "rpm_cmd": rpm_cmd, "state": state,
        "load_proxy": load_proxy, "fuel_cut": fuel_cut,
        "enrichment": enrichment, "misfire": misfire,
    }


def _run_one(run_id, seed, scenario_name, kind, mismatch_cfg, duration_s,
             engine_cfg, plant_cfg, sensors_cfg, dt_s, drift_mismatch_cfg=None):
    rng = np.random.default_rng(seed)
    n = int(round(duration_s / dt_s))
    rpm_range = (engine_cfg["engine"]["rpm_min"], engine_cfg["engine"]["rpm_max"])

    proto = _driver_protocol(kind, n, dt_s, rng, rpm_range)

    plant = VirtualEnginePlant(engine_cfg, plant_cfg, mismatch_cfg, dt_s, rng)
    drift_plant = None
    if kind == "calibration_shift" and drift_mismatch_cfg is not None:
        drift_plant = VirtualEnginePlant(engine_cfg, plant_cfg, drift_mismatch_cfg, dt_s,
                                          np.random.default_rng(seed + 777))
        # keep the drift plant's dynamic state continuous with the primary plant
        drift_plant.state = plant.state

    fuel_path = FuelLambdaPath(plant_cfg, engine_cfg, dt_s)
    actual_fuel_plant = ActualFuelDeliveryPlant(plant_cfg, engine_cfg, dt_s)
    sensors = SensorSuite(sensors_cfg, dt_s, rng)
    physics = PhysicsModel.from_config(engine_cfg)

    rows = []
    cold_segments = {"ect_extreme", "cold_high_load"}
    prev_state = None
    for k in range(n):
        active_plant = plant
        if drift_plant is not None and proto["state"][k] in ("adaptation", "holdout"):
            active_plant = drift_plant

        # Force a cold-engine excursion at the start of dedicated cold
        # boundary/corner segments so cold-start Stage-A blocking is
        # actually exercised in generated data (see virtual_engine.py note).
        if proto["state"][k] in cold_segments and prev_state not in cold_segments:
            active_plant.state.ect_c = 30.0
        prev_state = proto["state"][k]

        clean = active_plant.step(
            tps_command_pct=float(proto["tps_cmd"][k]),
            rpm=float(proto["rpm_cmd"][k]),
            load_proxy=float(proto["load_proxy"][k]),
        )

        fuel_cut = bool(proto["fuel_cut"][k])
        enrichment = bool(proto["enrichment"][k])
        misfire = bool(proto["misfire"][k])
        cold_start = clean["ect_c"] < 40.0

        # The ECU command is generated from its own imperfect speed-density
        # fuel map, never from hidden airflow.  Physical lambda is calculated
        # later from hidden airflow and independently simulated delivered fuel.
        lambda_target = 1.0
        if enrichment:
            lambda_target = 0.86
        if fuel_cut:
            pw_ms = 0.0
            lambda_true = 1.35
            actual_fuel_g_s = 0.0
            ecu_airflow_g_s = 0.0
        else:
            stoich = engine_cfg["constants"]["stoich_afr_gasoline"]
            inj = plant_cfg["injector"]
            fuel_map = inj["ecu_fuel_map"]
            ecu_airflow_g_s = float(physics.airflow_g_s(
                clean["rpm"], clean["map_kpa"], clean["iat_c"]
            ))
            ecu_airflow_g_s = max(
                ecu_airflow_g_s * fuel_map["airflow_scale"] + fuel_map["airflow_bias_g_s"],
                0.0,
            )
            fuel_target_g_s = ecu_airflow_g_s / (stoich * lambda_target)
            pw_ms = pulse_width_from_fuel_flow_ms(
                fuel_target_g_s,
                inj["q_injector_g_s_at_ref_dp"],
                inj["ref_dp_kpa"],
                inj["ref_dp_kpa"],
                inj["dead_time_ms"],
                engine_cfg["engine"]["cylinders"],
                clean["rpm"],
                inj["saturation_pw_ms"],
            )
            actual_fuel_g_s = actual_fuel_plant.step(
                pw_ms, inj["ref_dp_kpa"], clean["rpm"]
            )
            if actual_fuel_g_s <= 1e-9:
                lambda_true = 1.6
            else:
                lambda_true = clean["maf_reference_g_s"] / (stoich * actual_fuel_g_s)
            lambda_true += rng.normal(0, 0.004)

        if misfire:
            lambda_true += 0.15

        fault_code = 1 if misfire else 0
        lambda_true_valid_upstream = not (fuel_cut or cold_start)

        obs = sensors.observe(
            clean={
                "rpm": clean["rpm"], "map_kpa": clean["map_kpa"], "tps_pct": clean["tps_pct"],
                "iat_c": clean["iat_c"], "ect_c": clean["ect_c"], "lambda_true": lambda_true,
                "battery_v_true": 13.8, "injector_pw_ms_true": pw_ms,
            },
            ect_c=clean["ect_c"],
            lambda_true_valid=lambda_true_valid_upstream,
        )

        validity_flags = {
            "fuel_cut": fuel_cut, "enrichment": enrichment, "cold_start": cold_start,
            "misfire": misfire, "dropout": obs["dropout_any"],
            "sensor_fault": obs["sensor_fault_any"], "fault_code": fault_code,
        }
        fp = fuel_path.step(
            pw_ms=obs["injector_pw_ms"], dp_kpa=plant_cfg["injector"]["ref_dp_kpa"],
            rpm=obs["rpm"], lambda_measured=obs["lambda_measured"],
            lambda_valid=obs["lambda_valid"], validity_flags=validity_flags,
        )

        maf_physics = float(physics.airflow_g_s(obs["rpm"], obs["map_kpa"], obs["iat_c"]))

        calibration_allowed = bool(
            fp["pseudo_reference_valid"]
            and clean["ect_c"] >= 60.0
            and not obs["dropout_any"]
            and not obs["stuck_any"]
            and obs["lambda_valid"]
        )

        rows.append({
            "timestamp_s": round(k * dt_s, 4),
            "run_id": run_id,
            "seed": seed,
            "scenario": scenario_name,
            "state": proto["state"][k],
            "rpm": obs["rpm"],
            "map_kpa": obs["map_kpa"],
            "tps_pct": obs["tps_pct"],
            "iat_c": obs["iat_c"],
            "ect_c": obs["ect_c"],
            "lambda_measured": obs["lambda_measured"],
            "lambda_valid": obs["lambda_valid"],
            "injector_pw_ms": obs["injector_pw_ms"],
            "battery_v": obs["battery_v"],
            "maf_reference_g_s": clean["maf_reference_g_s"],
            "maf_physics_g_s": maf_physics,
            "fault_code": fault_code,
            "calibration_allowed": calibration_allowed,
            # extra (non-schema-required, kept for downstream calibration/fuel diagnostics)
            "pseudo_reference_g_s": fp["pseudo_reference_g_s"],
            "pseudo_reference_valid": fp["pseudo_reference_valid"],
            "pseudo_path_warm": fp["pseudo_path_warm"],
            "pseudo_fixed_delay_samples": fp["pseudo_fixed_delay_samples"],
            "pseudo_filter_alpha": fp["pseudo_filter_alpha"],
            "sensor_dropout_any": obs["dropout_any"],
            "sensor_stuck_any": obs["stuck_any"],
            # offline diagnostics only; prohibited from deployed features
            "lambda_true": lambda_true,
            "actual_fuel_g_s": actual_fuel_g_s,
            "ecu_feedforward_airflow_g_s": ecu_airflow_g_s,
        })

    return pd.DataFrame(rows)


def _sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_all(data_gen_cfg, engine_cfg, sensors_cfg, out_dir="data/raw", quick=False):
    os.makedirs(out_dir, exist_ok=True)
    dt_s = data_gen_cfg["sampling"]["dt_s"]
    seed_base = data_gen_cfg["sampling"]["seed_base"]
    mismatch_by_name = {m["name"]: MismatchConfig(**m) for m in data_gen_cfg["mismatch_levels"]}
    quick_divisor = data_gen_cfg.get("quick_divisor", 4) if quick else 1

    manifest_rows = []
    run_counter = 0

    for scenario_name, sc in data_gen_cfg["scenarios"].items():
        n_runs = max(1, sc["n_runs"] // quick_divisor) if quick else sc["n_runs"]
        duration_s = max(dt_s * 20, sc["duration_s"] / quick_divisor) if quick else sc["duration_s"]
        mismatch_cfg = mismatch_by_name[sc["mismatch"]]
        drift_cfg = None
        if sc["kind"] == "calibration_shift":
            drift_cfg = mismatch_cfg
            mismatch_cfg = mismatch_by_name["nominal"]

        scenario_dir = os.path.join(out_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        for i in range(n_runs):
            run_counter += 1
            seed = seed_base + run_counter
            run_id = f"{scenario_name}__run{i:03d}"
            df = _run_one(
                run_id=run_id, seed=seed, scenario_name=scenario_name, kind=sc["kind"],
                mismatch_cfg=mismatch_cfg, duration_s=duration_s, engine_cfg=engine_cfg,
                plant_cfg=data_gen_cfg["plant"], sensors_cfg=sensors_cfg, dt_s=dt_s,
                drift_mismatch_cfg=drift_cfg,
            )
            path = os.path.join(scenario_dir, f"run{i:03d}.parquet")
            df.to_parquet(path, index=False)
            manifest_rows.append({
                "run_id": run_id,
                "scenario": scenario_name,
                "kind": sc["kind"],
                "mismatch": sc["mismatch"],
                "split_role_hint": sc["split_role"],
                "seed": seed,
                "n_samples": len(df),
                "duration_s": duration_s,
                "path": path,
                "sha256": _sha256_of_file(path),
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(out_dir, "raw_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate M30B35 virtual dataset runs.")
    parser.add_argument("--config", default="configs/data_generation.yaml")
    parser.add_argument("--engine-config", default="configs/engine_m30b35.yaml")
    parser.add_argument("--sensors-config", default="configs/sensors.yaml")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        data_gen_cfg = yaml.safe_load(f)
    engine_cfg = load_engine_config(args.engine_config)
    with open(args.sensors_config) as f:
        sensors_cfg = yaml.safe_load(f)

    manifest = generate_all(data_gen_cfg, engine_cfg, sensors_cfg, args.out_dir, quick=args.quick)
    print(f"Generated {len(manifest)} runs -> {args.out_dir}/raw_manifest.csv")


if __name__ == "__main__":
    main()
