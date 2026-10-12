"""Physics-informed residual-AI estimator (roadmap Section 9).

maf_hat = maf_physics + residual_ai(features)   [additive, frozen -- see
configs/model_physics.yaml "correction_form"]

The residual target is generated from the HIDDEN reference airflow ONLY for
offline training-target construction (explicitly permitted by Section 3:
"may be used during development, training-target generation"). The hidden
reference is never a feature and never reaches the deployed inference path
(FEATURES below is the same frozen deployed list as Pure-AI).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from generate_dataset import FORBIDDEN_DEPLOYED_FEATURES
from physics_model import PhysicsModel, load_engine_config
from train_pure_ai import load_split, count_params  # reuse loader + param counter


def assert_no_leakage(features):
    leaked = set(features) & FORBIDDEN_DEPLOYED_FEATURES
    assert not leaked, f"Hidden-reference/diagnostic feature(s) leaked into deployed inputs: {leaked}"


def train_one_seed(X_train, y_train, X_val, y_val, arch_cfg, seed):
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_val)

    model = MLPRegressor(
        hidden_layer_sizes=tuple(arch_cfg["hidden_layer_sizes"]),
        activation=arch_cfg["activation"],
        solver=arch_cfg["solver"],
        alpha=arch_cfg["alpha"],
        learning_rate_init=arch_cfg["learning_rate_init"],
        max_iter=1,
        warm_start=True,
        random_state=seed,
        batch_size=min(arch_cfg.get("batch_size", 256), max(8, len(Xtr))),
    )

    best_val_mae = np.inf
    best_state = None
    no_improve = 0
    n_iter_no_change = arch_cfg.get("n_iter_no_change", 15)
    max_epochs = arch_cfg.get("max_iter", 400)
    epoch = 0

    for epoch in range(max_epochs):
        model.partial_fit(Xtr, y_train)
        val_pred = model.predict(Xva)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_state = {
                "coefs_": [c.copy() for c in model.coefs_],
                "intercepts_": [b.copy() for b in model.intercepts_],
            }
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= n_iter_no_change:
            break

    if best_state is not None:
        model.coefs_ = best_state["coefs_"]
        model.intercepts_ = best_state["intercepts_"]

    return model, scaler, best_val_mae, epoch + 1


def main():
    parser = argparse.ArgumentParser(description="Train the physics-informed residual-AI estimator.")
    parser.add_argument("--config", default="configs/model_residual_ai.yaml")
    parser.add_argument("--engine-config", default="configs/engine_m30b35.yaml")
    parser.add_argument("--splits-csv", default="data/splits/splits.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    features = cfg["model"]["features"]
    target = cfg["model"]["target"]
    anchor_col = cfg["model"]["physics_anchor"]
    assert_no_leakage(features)

    engine_cfg = load_engine_config(args.engine_config)
    physics = PhysicsModel.from_config(engine_cfg)

    train_roles = cfg["training"]["train_split_roles"]
    val_roles = cfg["training"]["val_split_roles"]
    train_df = load_split(args.splits_csv, train_roles)
    val_df = load_split(args.splits_csv, val_roles)

    # Recompute the physics anchor from the estimator's OWN baseline (not
    # trusting the stored maf_physics_g_s column's provenance blindly, but
    # they should match since generate_dataset.py used the same PhysicsModel).
    for df in (train_df, val_df):
        recomputed = physics.airflow_g_s(df["rpm"].to_numpy(), df["map_kpa"].to_numpy(), df["iat_c"].to_numpy())
        df["_physics_anchor"] = recomputed

    clip_lo, clip_hi = cfg["training"]["residual_target_clip_g_s"]

    X_train = train_df[features].to_numpy(dtype=float)
    residual_train = np.clip(train_df[target].to_numpy(dtype=float) - train_df["_physics_anchor"].to_numpy(), clip_lo, clip_hi)
    X_val = val_df[features].to_numpy(dtype=float)
    residual_val = np.clip(val_df[target].to_numpy(dtype=float) - val_df["_physics_anchor"].to_numpy(), clip_lo, clip_hi)

    os.makedirs(args.models_dir, exist_ok=True)
    summary = []
    for seed in cfg["training"]["seeds"]:
        model, scaler, val_mae, n_epochs = train_one_seed(
            X_train, residual_train, X_val, residual_val, cfg["architecture"], seed
        )
        n_params = count_params(model)
        artifact = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "target": target,
            "correction_form": cfg["model"]["correction_form"],
            "engine_config_path": args.engine_config,
            "seed": seed,
            "val_residual_mae_g_s": val_mae,
            "n_epochs": n_epochs,
            "n_parameters": n_params,
            "scikit_learn_version": sklearn.__version__,
        }
        out_path = os.path.join(args.models_dir, f"residual_ai_seed{seed}.joblib")
        joblib.dump(artifact, out_path)
        summary.append({"seed": seed, "val_residual_mae_g_s": val_mae, "n_epochs": n_epochs, "n_parameters": n_params})
        print(f"[residual_ai] seed={seed} val_residual_MAE={val_mae:.3f} g/s epochs={n_epochs} params={n_params} -> {out_path}")

    budget = cfg.get("resource_budget", {})
    if budget:
        max_p = budget.get("max_parameters")
        if max_p is not None and any(s["n_parameters"] > max_p for s in summary):
            print(f"WARNING: residual-AI parameter count exceeds declared budget {max_p}")

    with open(os.path.join(args.models_dir, "residual_ai_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def predict_residual_ai(artifact, df: pd.DataFrame, engine_cfg=None) -> np.ndarray:
    """Inference helper: maf_hat = physics_anchor + residual_model(features).

    Reused by evaluate.py / online_calibration.py so the composition rule
    (Section 5) is defined in exactly one place.
    """
    if engine_cfg is None:
        engine_cfg = load_engine_config(artifact["engine_config_path"])
    physics = PhysicsModel.from_config(engine_cfg)
    anchor = physics.airflow_g_s(df["rpm"].to_numpy(), df["map_kpa"].to_numpy(), df["iat_c"].to_numpy())
    X = df[artifact["features"]].to_numpy(dtype=float)
    Xs = artifact["scaler"].transform(X)
    residual = artifact["model"].predict(Xs)
    return np.asarray(anchor) + np.asarray(residual), np.asarray(anchor)


if __name__ == "__main__":
    main()
