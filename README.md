# Lambda-Guided Bounded Scale Adaptation for a Physics-Informed Virtual Mass-Airflow Estimator

Reference implementation of the BMW M30B35 simulation study by Alaa A. Aldhalemi (University of Kufa).

> **Scope.** This is a simulation-level feasibility study. It does not establish physical engine accuracy, real-time microcontroller operation, ECU compatibility, compatibility with other engines, or any real-world fuel-economy gain. Engine and hardware validation are future work.

| | |
|---|---|
| Paper | *Lambda-Guided Bounded Scale Adaptation for a Physics-Informed Virtual Mass-Airflow Estimator: A BMW M30B35 Simulation Study* (submitted; link to be added) |
| Release | `v1.0.0` |
| Archive (DOI) | `<ZENODO-DOI>` |
| Licence | `<LICENSE>` |

## What it is

A virtual mass-airflow (MAF) estimator that combines a speed-density physics baseline with a small residual neural network (16-16-8, 513 parameters). One bounded global scale `s` in [0.80, 1.20] is adapted online from a delayed pseudo-reference built from wideband lambda and the injector command. Every update passes a four-stage supervisor:

1. **A, validity:** simulator health mask and operating-condition checks.
2. **B, excitation:** a frozen scalar Fisher-information gate.
3. **C, bounded proposal:** deadband, per-update, absolute and rate limits.
4. **D, causal acceptance:** a candidate is judged on a strictly later window and rolled back if it does not improve.

Six policies are compared:

| Policy | Description |
|---|---|
| C0 | Frozen estimator, no calibration |
| C1 | Ungated adaptation |
| C2 | Validity-only adaptation |
| C3 | Full supervisor (A to D) |
| C4 | Oracle using hidden airflow (not deployable) |
| C5 | Matched one-state recursive least squares (RLS) |

## Main results (full mode)

| Result | Value |
|---|---|
| Post-shift holdout NRMSE, C0 to C3 | 0.0152 to 0.0079 (48.2 % lower) |
| Original acceptance check (`results/acceptance_report.json`) | **FAIL**. C3 and C1 both had zero false accepted updates, so the strict "lower than C1" criterion cannot be met |

C3 did not outperform the simpler adaptive policies on the original final test. See the paper for all tables and the statistical analysis.

### Exploratory high-load extension

Twelve high-load trajectories, seven unflagged ±15 % lambda-fault conditions, five frozen networks and policies C0 to C5 (2520 replays). The values in the paper come from a re-implementation that was run once from the protocol description.

| | |
|---|---|
| Code | `src/highload_extension.py` |
| Configuration | `configs/highload_extension.yaml` |
| Outputs | `results/highload/` (`table14_reimpl.csv`, `paired_tests.csv`, `per_replay_metrics.csv`, `calibration_events.csv`, `integrity.json`) |

```bash
python3 src/highload_extension.py   # about 3 minutes on one core
```

## Terminology

- **Reference airflow:** hidden clean airflow from the virtual plant. Used only for training targets and offline evaluation, never as an estimator input.
- **Estimator inputs:** RPM, MAP, TPS, IAT and ECT (simulated electrical signals; no OBD-II or CAN PIDs).
- **Calibration feedback:** conditioned wideband lambda and injector command, turned into a delayed *pseudo-reference*. It is not a calibrated measurement.
- **Legacy output emulation:** software-only voltage/frequency conversion, not ECU compatibility.

## Repository layout

```text
configs/   frozen parameters (engine, plant, sensors, models, calibration, experiments)
src/       plant, sensors, fuel/lambda path, models, gates, calibration, evaluation
tests/     unit and integration tests
data/      generated runs (raw/) and disjoint splits (splits/)
models/    trained model files (pure-AI and residual-AI, five seeds each)
results/   CSV/JSON outputs, figures/, and highload/
```

The raw simulation data and files over 50 MB (for example `results/parameter_trajectory.csv`) are archived on Zenodo, not in git.

## Quickstart (Debian 13, Python 3.13)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .

make test              # unit and integration tests
make pipeline-quick    # smoke run (minutes); software check only
make pipeline          # full study (slow)
```

`--mode quick` shrinks run counts, durations and training epochs. It never changes thresholds, bounds, windows or endpoints. **Do not quote quick-mode results.** The results archived with this repository are from full mode (`results/environment.json`, `mode: full`).

`make pipeline` refuses to overwrite existing full-mode results unless `--allow-refreeze` is passed. `make clean` deletes generated data, models and results, so run it only on a copy.

## Pipeline stages

1. `generate_dataset.py`: hidden plant, sensor corruption and fuel/lambda path, written as per-run files.
2. `prepare_splits.py`: disjoint train, validation, threshold-selection and final-test runs, with checksums.
3. `train_pure_ai.py`, `train_residual_ai.py`: five seeds each, with leakage-checked feature lists.
4. `scalar_fisher_gate.py`, `calibration_gate.py`, `online_calibration.py`, `classical_rls.py`: policies C0 to C5.
5. `evaluate.py`, `statistical_analysis.py`: metrics and paired run-level tests with Holm correction.
6. `make_publication_figures.py`: figures, each saved with its data CSV.

## Outputs in `results/`

| File | Content |
|---|---|
| `environment.json`, `dataset_manifest.json` | Run environment and dataset checksums |
| `test_metrics_all_models.csv`, `per_run_model_metrics.csv`, `test_predictions.parquet` | Estimator accuracy |
| `transient_state_metrics.csv`, `boundary_corner_metrics.csv`, `mismatch_crossover.csv` | Transient, boundary and mismatch results |
| `calibration_events.csv`, `parameter_trajectory.csv`, `drift_recovery_metrics.csv`, `calibration_data_efficiency.csv`, `classical_comparator_metrics.csv` | Calibration policies |
| `scalar_fisher_trace.csv`, `information_gate_report.csv` | Excitation gate |
| `legacy_output_metrics.csv` | Software output-mapping check |
| `statistical_tests.csv`, `acceptance_report.json` | Statistics and predeclared acceptance check |
| `figures/` | Figures with their data files |
| `highload/` | Exploratory high-load extension |

## Reproducibility notes

- Recorded environment: Debian 13.7, Python 3.13.5, packages as in `requirements-lock.txt`.
- The archived full run did not record a git revision; the released tag identifies the code snapshot.
- Trained models were pickled with scikit-learn 1.9.1. Load them with the locked version.

## Citation

See `CITATION.cff` (to be added with the release DOI).

## Contact

Alaa A. Aldhalemi, University of Kufa, alaaa.aldhalemi@uokufa.edu.iq, alaamh198012@gmail.com
# m30-virtual-maf-lambda-calibration
