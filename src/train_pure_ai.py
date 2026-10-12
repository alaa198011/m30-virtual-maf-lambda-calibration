"""Direct Pure-AI airflow estimator (roadmap Section 9).

Trains a compact MLP (default 16-16-8, frozen in configs/model_pure_ai.yaml)
to predict airflow directly from the FROZEN deployed feature list (RPM,
MAP, TPS, IAT, ECT). The scaler is fit on the training split ONLY. Trains
one model per declared seed (>=5, Section 13 acceptance gate).

Hard leakage guard: asserts the feature list never contains a hidden-
reference or diagnostic-bus column (Section 3 / Section 7: "The pipeline
must fail if a hidden reference feature is passed to the deployed
estimator").
"""
from __future__ import annotations

import argparse
import glob
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

# We intentionally drive MLPRegressor one epoch at a time (max_iter=1,
# warm_start=True) to implement manual early stopping against the
# dedicated validation split; sklearn's per-call convergence warning is
# expected and not actionable here.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from generate_dataset import FORBIDDEN_DEPLOYED_FEATURES


def assert_no_leakage(features):
    leaked = set(features) & FORBIDDEN_DEPLOYED_FEATURES
    assert not leaked, f"Hidden-reference/diagnostic feature(s) leaked into deployed inputs: {leaked}"


def load_split(splits_csv, roles):
    splits = pd.read_csv(splits_csv)
    paths = splits.loc[splits.split_role.isin(roles), "path"].tolist()
    if not paths:
        raise RuntimeError(f"No runs found for split roles {roles} in {splits_csv}")
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


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

    for epoch in range(max_epochs):
        # partial_fit preserves Adam optimizer state across epochs; repeatedly
        # calling fit(max_iter=1) resets optimizer moments in some sklearn versions.
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


def count_params(model: MLPRegressor) -> int:
    return int(sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_))


def main():
    parser = argparse.ArgumentParser(description="Train the Pure-AI direct airflow estimator.")
    parser.add_argument("--config", default="configs/model_pure_ai.yaml")
    parser.add_argument("--splits-csv", default="data/splits/splits.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    features = cfg["model"]["features"]
    target = cfg["model"]["target"]
    assert_no_leakage(features)

    train_roles = cfg["training"]["train_split_roles"]
    val_roles = cfg["training"]["val_split_roles"]

    train_df = load_split(args.splits_csv, train_roles)
    val_df = load_split(args.splits_csv, val_roles)

    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    X_val = val_df[features].to_numpy(dtype=float)
    y_val = val_df[target].to_numpy(dtype=float)

    os.makedirs(args.models_dir, exist_ok=True)
    summary = []
    for seed in cfg["training"]["seeds"]:
        model, scaler, val_mae, n_epochs = train_one_seed(
            X_train, y_train, X_val, y_val, cfg["architecture"], seed
        )
        n_params = count_params(model)
        artifact = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "target": target,
            "seed": seed,
            "val_mae_g_s": val_mae,
            "n_epochs": n_epochs,
            "n_parameters": n_params,
            "scikit_learn_version": sklearn.__version__,
        }
        out_path = os.path.join(args.models_dir, f"pure_ai_seed{seed}.joblib")
        joblib.dump(artifact, out_path)
        summary.append({"seed": seed, "val_mae_g_s": val_mae, "n_epochs": n_epochs, "n_parameters": n_params})
        print(f"[pure_ai] seed={seed} val_MAE={val_mae:.3f} g/s epochs={n_epochs} params={n_params} -> {out_path}")

    with open(os.path.join(args.models_dir, "pure_ai_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
