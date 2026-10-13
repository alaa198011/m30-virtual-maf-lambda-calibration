"""Exploratory high-load adverse-calibration extension (re-implementation).

Generates 12 new 240 s high-load trajectories, applies seven unflagged lambda-gain
conditions to the MEASURED lambda only, replays C0-C5 with the frozen residual
networks / frozen calibration config, and writes Table-14-style outputs.
Nothing in configs/calibration.yaml, models/ or the policy code is modified.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, sys, copy
import numpy as np, pandas as pd, yaml, joblib

sys.path.insert(0, os.path.dirname(__file__))
from physics_model import PhysicsModel, load_engine_config
from virtual_engine import VirtualEnginePlant, MismatchConfig
from sensor_models import SensorSuite
from fuel_lambda_path import ActualFuelDeliveryPlant, FuelLambdaPath, pulse_width_from_fuel_flow_ms
from generate_dataset import _smooth_random_walk
from online_calibration import run_calibration_on_run
from train_residual_ai import predict_residual_ai
from uncertainty import bootstrap_ci

NORM = 260.0


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def fault_gain_series(t, cond, rng_run):
    g = np.ones_like(t)
    s = cond["schedule"]
    if s == "none":
        return g
    lo, hi = cond["start_s"], cond["end_s"]
    if s == "window":
        g[(t >= lo) & (t < hi)] = cond["gain"]
    elif s == "recurring":
        on, per = cond["on_s"], cond["period_s"]
        phase = 0.0 if cond["phase"] == "aligned_to_proposal_windows" else rng_run.uniform(0.0, per)
        x = np.mod(t - lo - phase, per)
        active = (t >= lo) & (t < hi) & (x < on)
        g[active] = cond["gain"]
    return g


def simulate_trajectory(seed, hl, data_cfg, engine_cfg, sensors_cfg):
    """Healthy base trajectory: returns arrays needed to rebuild the lambda path per condition."""
    dt = data_cfg["sampling"]["dt_s"]
    n = int(round(hl["trajectories"]["duration_s"] / dt))
    rng = np.random.default_rng(seed)
    r_lo, r_hi = hl["trajectories"]["rpm_command_range"]
    t_lo, t_hi = hl["trajectories"]["tps_command_range_pct"]
    tps_cmd = _smooth_random_walk(rng, n, t_lo, t_hi, 3.0)
    rpm_cmd = _smooth_random_walk(rng, n, r_lo, r_hi, 60.0)
    seg = hl["trajectories"]["segments_s"]
    t = np.arange(n) * dt
    state = np.where(t < seg["adaptation"][0], "pre_drift", np.where(t < seg["holdout"][0], "adaptation", "holdout")).astype(object)
    load_proxy = np.clip(tps_cmd / 100.0, 0, 1)

    mm = {m["name"]: MismatchConfig(**m) for m in data_cfg["mismatch_levels"]}
    plant = VirtualEnginePlant(engine_cfg, data_cfg["plant"], mm["nominal"], dt, rng)
    drift = VirtualEnginePlant(engine_cfg, data_cfg["plant"], mm[hl["trajectories"]["plant_mismatch_after_80s"]], dt,
                               np.random.default_rng(seed + 777))
    drift.state = plant.state
    actual = ActualFuelDeliveryPlant(data_cfg["plant"], engine_cfg, dt)
    sc = copy.deepcopy(sensors_cfg)
    if hl["trajectories"]["disable_random_dropout_stuck_misfire"]:
        for ch in sc.values():
            if isinstance(ch, dict):
                ch["dropout_rate_per_s"] = 0.0
                ch["stuck_rate_per_s"] = 0.0
    sensors = SensorSuite(sc, dt, rng)
    physics = PhysicsModel.from_config(engine_cfg)
    stoich = engine_cfg["constants"]["stoich_afr_gasoline"]
    inj = data_cfg["plant"]["injector"]
    fm = inj["ecu_fuel_map"]

    keys = ["rpm", "map_kpa", "tps_pct", "iat_c", "ect_c", "lambda_measured", "lambda_valid", "injector_pw_ms",
            "battery_v", "maf_reference_g_s", "maf_physics_g_s", "dropout", "stuck", "sensor_fault"]
    cols = {k: np.zeros(n) for k in keys}
    for k in range(n):
        ap = drift if state[k] in ("adaptation", "holdout") else plant
        clean = ap.step(tps_command_pct=float(tps_cmd[k]), rpm=float(rpm_cmd[k]), load_proxy=float(load_proxy[k]))
        cold = clean["ect_c"] < 40.0
        ecu_air = max(float(physics.airflow_g_s(clean["rpm"], clean["map_kpa"], clean["iat_c"])) * fm["airflow_scale"] + fm["airflow_bias_g_s"], 0.0)
        pw = pulse_width_from_fuel_flow_ms(ecu_air / (stoich * 1.0), inj["q_injector_g_s_at_ref_dp"], inj["ref_dp_kpa"], inj["ref_dp_kpa"],
                                            inj["dead_time_ms"], engine_cfg["engine"]["cylinders"], clean["rpm"], inj["saturation_pw_ms"])
        fuel = actual.step(pw, inj["ref_dp_kpa"], clean["rpm"])
        lam_true = 1.6 if fuel <= 1e-9 else clean["maf_reference_g_s"] / (stoich * fuel)
        lam_true += rng.normal(0, 0.004)
        obs = sensors.observe(clean={"rpm": clean["rpm"], "map_kpa": clean["map_kpa"], "tps_pct": clean["tps_pct"], "iat_c": clean["iat_c"],
                                     "ect_c": clean["ect_c"], "lambda_true": lam_true, "battery_v_true": 13.8, "injector_pw_ms_true": pw},
                              ect_c=clean["ect_c"], lambda_true_valid=not cold)
        for key in ("rpm", "map_kpa", "tps_pct", "iat_c", "ect_c", "lambda_measured", "injector_pw_ms", "battery_v"):
            cols[key][k] = obs[key]
        cols["lambda_valid"][k] = obs["lambda_valid"]
        cols["maf_reference_g_s"][k] = clean["maf_reference_g_s"]
        cols["maf_physics_g_s"][k] = float(physics.airflow_g_s(obs["rpm"], obs["map_kpa"], obs["iat_c"]))
        cols["dropout"][k] = obs["dropout_any"]; cols["stuck"][k] = obs["stuck_any"]; cols["sensor_fault"][k] = obs["sensor_fault_any"]
    return {"t": t, "state": state, "cols": cols, "dt": dt, "seed": seed}


def build_condition_df(base, gain, data_cfg, engine_cfg, run_id):
    dt, c, t = base["dt"], base["cols"], base["t"]
    n = len(t)
    fp = FuelLambdaPath(data_cfg["plant"], engine_cfg, dt)
    ps = np.zeros(n); pv = np.zeros(n, bool); warm = np.zeros(n, bool)
    for k in range(n):
        out = fp.step(pw_ms=c["injector_pw_ms"][k], dp_kpa=data_cfg["plant"]["injector"]["ref_dp_kpa"], rpm=c["rpm"][k],
                      lambda_measured=c["lambda_measured"][k] * gain[k], lambda_valid=bool(c["lambda_valid"][k]),
                      validity_flags={"fuel_cut": False, "enrichment": False, "cold_start": c["ect_c"][k] < 40.0, "misfire": False,
                                      "dropout": bool(c["dropout"][k]), "sensor_fault": bool(c["sensor_fault"][k]), "fault_code": 0})
        ps[k] = out["pseudo_reference_g_s"]; pv[k] = out["pseudo_reference_valid"]; warm[k] = out["pseudo_path_warm"]
    allowed = pv & (c["ect_c"] >= 60.0) & (~c["dropout"].astype(bool)) & (~c["stuck"].astype(bool)) & (c["lambda_valid"].astype(bool))
    df = pd.DataFrame({
        "timestamp_s": np.round(t, 4), "run_id": run_id, "seed": base["seed"], "scenario": "highload_extension", "state": base["state"],
        "rpm": c["rpm"], "map_kpa": c["map_kpa"], "tps_pct": c["tps_pct"], "iat_c": c["iat_c"], "ect_c": c["ect_c"],
        "lambda_measured": c["lambda_measured"] * gain, "lambda_valid": c["lambda_valid"].astype(bool),
        "injector_pw_ms": c["injector_pw_ms"], "battery_v": c["battery_v"], "maf_reference_g_s": c["maf_reference_g_s"],
        "maf_physics_g_s": c["maf_physics_g_s"], "fault_code": 0, "calibration_allowed": allowed,
        "pseudo_reference_g_s": ps, "pseudo_reference_valid": pv, "pseudo_path_warm": warm,
        "pseudo_fixed_delay_samples": fp.fixed_delay_samples, "pseudo_filter_alpha": fp.filter_alpha,
    })
    return df


def exact_sign_flip_p(d):
    d = np.asarray(d, float); n = len(d)
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    stats = np.abs(signs @ d)
    obs = abs(d.sum())
    return float(np.mean(stats >= obs - 1e-12))


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/highload_extension.yaml")
    ap.add_argument("--out", default="results/highload")
    ap.add_argument("--n-traj", type=int, default=None)
    a = ap.parse_args()
    hl = load_yaml(a.config)
    cfg_hash = hashlib.sha256(open(a.config, "rb").read()).hexdigest()
    data_cfg = load_yaml("configs/data_generation.yaml"); sensors_cfg = load_yaml("configs/sensors.yaml")
    calib_cfg = load_yaml("configs/calibration.yaml"); engine_cfg = load_engine_config("configs/engine_m30b35.yaml")
    os.makedirs(a.out, exist_ok=True)
    n_traj = a.n_traj or hl["trajectories"]["n"]
    seeds = hl["network_seeds"]
    arts = {s: joblib.load(f"models/residual_ai_seed{s}.joblib") for s in seeds}
    conds = hl["conditions"]; modes = hl["policies"]
    rows, events_all, integrity = [], [], {"n_replays": 0, "failures": []}
    base_ref_cache = {}
    for ti in range(n_traj):
        seed = hl["trajectories"]["seed_first"] + ti
        base = simulate_trajectory(seed, hl, data_cfg, engine_cfg, sensors_cfg)
        run_rng = np.random.default_rng(seed + 99)
        dfs = {}
        for cname, cond in conds.items():
            gain = fault_gain_series(base["t"], cond, run_rng)
            dfs[cname] = build_condition_df(base, gain, data_cfg, engine_cfg, f"highload__run{ti:02d}__{cname}")
        # integrity: faults must change only the pseudo-reference (frozen inputs identical)
        h = dfs["healthy"]
        for cname, df in dfs.items():
            same = all(np.array_equal(df[k].to_numpy(), h[k].to_numpy()) for k in ("rpm", "map_kpa", "tps_pct", "iat_c", "ect_c", "maf_reference_g_s"))
            if not same: integrity["failures"].append(f"{ti}:{cname}:frozen_inputs_changed")
        phis = {s: predict_residual_ai(arts[s], h, engine_cfg)[0] for s in seeds}
        for cname, df in dfs.items():
            adapt = (df.state == "adaptation").to_numpy(); hold = (df.state == "holdout").to_numpy()
            ref = df.maf_reference_g_s.to_numpy()
            for s in seeds:
                phi = phis[s]
                for mode in modes:
                    res = run_calibration_on_run(df, phi, mode, calib_cfg["modes"][mode], calib_cfg, base["dt"], df.run_id.iloc[0], model_seed=s)
                    cs = res["committed_s"]
                    ev = pd.DataFrame(res["events"])
                    acc = ev[ev.decision == "accept"] if len(ev) else ev
                    rows.append({
                        "traj": ti, "seed_traj": seed, "condition": cname, "net_seed": s, "mode": mode,
                        "adapt_rmse_g_s": float(np.sqrt(np.mean((cs[adapt] * phi[adapt] - ref[adapt]) ** 2))),
                        "holdout_nrmse": float(np.sqrt(np.mean((res["final_s"] * phi[hold] - ref[hold]) ** 2)) / NORM),
                        "final_s": res["final_s"],
                        "n_accepted": int(len(acc)),
                        "n_rejected": int((ev.decision == "reject").sum()) if len(ev) else 0,
                        "n_false_accepted": int(acc.false_update.fillna(False).sum()) if len(acc) else 0,
                        "valid_adapt_fraction": float(df.calibration_allowed.to_numpy()[adapt].mean()),
                        "s_min": float(cs.min()), "s_max": float(cs.max()),
                    })
                    integrity["n_replays"] += 1
                    if len(ev):
                        e = ev.copy(); e["traj"] = ti; e["condition"] = cname; events_all.append(e)
        print(f"traj {ti+1}/{n_traj} done", flush=True)
    R = pd.DataFrame(rows); R.to_csv(f"{a.out}/per_replay_metrics.csv", index=False)
    if events_all: pd.concat(events_all).to_csv(f"{a.out}/calibration_events.csv", index=False)
    # C0 / C4 invariance across conditions
    for m in ("C0",):
        g = R[R["mode"] == m].groupby(["traj", "net_seed"]).adapt_rmse_g_s.nunique()
        if (g > 1).any(): integrity["failures"].append(f"{m}_not_invariant")
    # Table 14
    run = R.groupby(["condition", "mode", "traj"]).agg(adapt=("adapt_rmse_g_s", "mean"), hold=("holdout_nrmse", "mean")).reset_index()
    t14 = []
    for cname in conds:
        r = {"condition": cname}
        for m in ("C0", "C1", "C2", "C3", "C4", "C5"):
            r[f"{m}_adapt_rmse"] = run[(run.condition == cname) & (run["mode"] == m)].adapt.mean()
        for m in ("C2", "C3"):
            sub = R[(R.condition == cname) & (R["mode"] == m)]
            r[f"{m}_false"] = int(sub.n_false_accepted.sum()); r[f"{m}_accepted"] = int(sub.n_accepted.sum())
        t14.append(r)
    pd.DataFrame(t14).to_csv(f"{a.out}/table14_reimpl.csv", index=False)
    # Paired tests
    tests = []
    for cname in conds:
        for (m1, m2) in [tuple(c) for c in hl["statistics"]["comparisons"]]:
            x = run[(run.condition == cname) & (run["mode"] == m1)].sort_values("traj").adapt.to_numpy()
            y = run[(run.condition == cname) & (run["mode"] == m2)].sort_values("traj").adapt.to_numpy()
            d = x - y; mean, lo, hi = bootstrap_ci(d, n_boot=hl["statistics"]["bootstrap"]["resamples"], seed=0)
            tests.append({"condition": cname, "a": m1, "b": m2, "mean_diff_g_s": mean, "ci_lo": lo, "ci_hi": hi, "p_raw": exact_sign_flip_p(d)})
    T = pd.DataFrame(tests); T["p_holm"] = holm(T.p_raw.to_numpy()); T.to_csv(f"{a.out}/paired_tests.csv", index=False)
    integrity["config_sha256"] = cfg_hash; integrity["bound_violations"] = int(((R.s_min < 0.8 - 1e-9) | (R.s_max > 1.2 + 1e-9)).sum())
    json.dump(integrity, open(f"{a.out}/integrity.json", "w"), indent=2)
    print(pd.DataFrame(t14).round(3).to_string()); print(T.round(4).to_string()); print(integrity)


if __name__ == "__main__":
    main()
