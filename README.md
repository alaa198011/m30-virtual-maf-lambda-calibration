# M30 Self-Calibrating Virtual MAF — Reference Implementation

Reproducible Debian 13 / Python research codebase implementing the roadmap
*"Identifiability-Aware, Lambda-Guided Calibration of a Physics-Informed
Virtual Airflow Sensor"* (BMW M30B35 virtual case study, Article 1 scope).

This repository is a **simulation-level feasibility study**. It does **not**
establish physical engine accuracy, real-time MCU operation, physical ECU
acceptance, universal engine compatibility, or real-world fuel-economy
improvement. See `manuscript/limitations.md`.

## What this is

A compact, physics-informed virtual airflow estimator (speed-density
baseline + residual MLP) that adapts a **bounded global airflow scale**
using conditioned wideband-lambda / injector-command feedback, gated by a
**frozen scalar Fisher-information threshold** and accepted only after a
**strictly subsequent, causal acceptance window**. It is compared against a
Pure-AI direct estimator, an ungated/validity-only calibration ablation, an
oracle (non-deployable) upper bound, and a matched one-state recursive
least-squares (RLS) comparator.

## Terminology

- **Reference airflow**: hidden clean airflow from the virtual plant. Used
  only for training-target generation and offline evaluation — **never** a
  deployed estimator feature.
- **Estimator inputs**: RPM, MAP, TPS, IAT, ECT (direct simulated electrical
  signals only — no OBD-II / CAN diagnostic PIDs).
- **Calibration feedback**: conditioned wideband-lambda + injector command,
  turned into a delayed *pseudo-reference* (not a calibrated measurement).
- **Deployed estimator**: the frozen model, excluding hidden reference
  airflow.
- **Legacy output emulation**: software-only voltage/frequency conversion.
  Not physical ECU compatibility.

Preferred phrase: **"reference-MAF-free field operation"**, not "operation
without any reference" — field calibration still depends on conditioned
feedback and declared fuel/engine assumptions.

## Project layout

```
m30_self_calibrating_maf/
├── configs/            declarative, frozen-before-final-test parameters
├── src/                pipeline modules (plant, sensors, models, gate, eval)
├── tests/              unit tests (units, leakage guard, gate, RLS, splits)
├── data/               generated runs (raw/) and disjoint splits (splits/)
├── models/             trained model artifacts (.joblib) per seed
├── results/            CSV/JSON outputs + figures/ (required software outputs)
└── manuscript/         article scaffolding, limitations, Article 2 handoff
```

## Quickstart (Debian 13)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .

make pipeline-quick     # smoke-sized end-to-end run (minutes)
make pipeline           # full study-sized run per configs/*.yaml (slow)
make test                # unit tests
```

`make pipeline-quick` and `make pipeline` both call
`src/run_pipeline.py`; the `--quick` flag shrinks run counts, trajectory
durations and training epochs so the *entire* pipeline (generate → split → train → calibrate
→ evaluate → statistics → figures) can be smoke-tested in a few minutes
without changing calibration windows, thresholds, bounds, or endpoints.

This repository ships with `data/`, `models/`, and `results/` already
populated from one such `--mode quick` run (~20 seconds on a single core),
so `results/figures/` and every CSV in `README.md#required-software-outputs`
can be inspected immediately without first setting up a virtual
environment. Run `make clean` before `make pipeline` to regenerate
everything at full, roadmap-declared scale.

## Pipeline stages

1. `generate_dataset.py` — independent hidden plant (`virtual_engine.py`,
   structurally different from the estimator's speed-density baseline) +
   sensor corruption (`sensor_models.py`) + fuel/lambda path
   (`fuel_lambda_path.py`) → per-run parquet files under `data/raw/`.
2. `prepare_splits.py` — disjoint train/validation/threshold-selection/
   final-test run IDs, scalers fit on training data only, checksums,
   `dataset_manifest.json`.
3. `train_pure_ai.py` / `train_residual_ai.py` — 5-seed compact MLP
   (16-16-8) training with a frozen, explicit, leakage-checked feature list.
4. `scalar_fisher_gate.py` + `calibration_gate.py` + `online_calibration.py`
   + `classical_rls.py` — the four-stage calibration gate (validity → scalar
   Fisher information → bounded proposal → causal accept/rollback) run in
   modes C0–C5 (see `configs/calibration.yaml`).
5. `evaluate.py` + `statistical_analysis.py` — required CSV outputs, paired
   run-level statistics (Wilcoxon/paired-t + Holm correction).
6. `make_publication_figures.py` — the 9 essential figures, each built from
   a stored CSV/parquet file saved beside the figure.

## Required software outputs (Section 14)

Written under `results/`: `environment.json`, `dataset_manifest.json`,
`test_metrics_all_models.csv`, `test_predictions.parquet`,
`transient_state_metrics.csv`, `calibration_events.csv`,
`scalar_fisher_trace.csv`, `calibration_data_efficiency.csv`,
`classical_comparator_metrics.csv`, `mismatch_crossover.csv`,
`drift_recovery_metrics.csv`, `information_gate_report.csv`,
`legacy_output_metrics.csv`, `parameter_trajectory.csv`,
`acceptance_report.json`, `statistical_tests.csv`.

## Acceptance gates before publication (Section 13)

`tests/test_acceptance_gates.py` checks the mechanical/leakage-safety
subset of Section 13 automatically (disjoint run IDs, no hidden-reference
feature leakage, scalers fit on train only, bounded parameters, logged
gate decisions with reasons, five AI seeds, host-timing labeling). The
scientific subset (frozen thresholds, predeclared NRMSE target and margin,
false-update definition) is evaluated explicitly in
`results/acceptance_report.json`. Configurations are treated as frozen
inputs: **do not edit thresholds after inspecting final-test results** —
`run_pipeline.py --full` refuses to re-run the final-test stage once
`results/test_metrics_all_models.csv` exists unless `--allow-refreeze` is
passed explicitly, to make post-hoc threshold changes an auditable,
deliberate act rather than an accident.

The included results are a **quick-mode verification run**, not final article
evidence. A failed quick-mode acceptance item must be reported rather than
hidden or tuned using final-test data. Run full mode on the declared Debian 13
host before drawing publication conclusions.

## What Article 1 may and may not claim

See `manuscript/claims.md` for the frozen permitted/excluded claim lists
carried over verbatim from the roadmap (Section 1).

## Article 2 handoff

See `manuscript/article2_handoff.md` for the freeze checklist (features,
architecture, weights, I/O contract, calibration parameterization, RLS
definition, legacy-output profile, PIL test vectors, learning budget,
configs/checksums) required before hardware porting.
