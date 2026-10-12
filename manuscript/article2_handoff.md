# Article 2 handoff freeze checklist (roadmap Section 18)

Freeze the following at the end of Article 1, using the artifacts this
repository produces:

- [ ] Residual-model features, architecture, parameter count, normalization,
      and weights — `models/residual_ai_seed*.joblib` + `configs/model_residual_ai.yaml`.
- [ ] Direct-wired sensor I/O contract, explicit exclusion of OBD-II and CAN
      diagnostic PIDs, and validity rules — `configs/sensors.yaml`,
      `src/sensor_models.py`, `src/calibration_gate.py` Stage A.
- [ ] Calibration parameterization, bounds, delay model, gate conditions, and
      rollback logic — `configs/calibration.yaml`, `src/calibration_gate.py`,
      `src/online_calibration.py`.
- [ ] Scalar Fisher-information definition, fixed valid-sample count N,
      pseudo-reference variance source, and frozen I_s,min threshold —
      `configs/calibration.yaml#stage_b_information`, `src/scalar_fisher_gate.py`.
- [ ] One-state RLS definition and matched comparator protocol —
      `src/classical_rls.py`, `results/classical_comparator_metrics.csv`.
- [ ] One complete software legacy-output profile —
      `configs/output_profiles.yaml`, `src/legacy_output.py`,
      `results/legacy_output_metrics.csv`.
- [ ] Recorded input sequences and expected outputs for processor-in-the-loop
      testing — `results/test_predictions.parquet` + `data/splits/`.
- [ ] The accepted-informative-data learning budget —
      `results/calibration_data_efficiency.csv`.
- [ ] All configuration files, checksums, source revision, and final
      metrics — `results/environment.json`, `results/dataset_manifest.json`,
      `results/test_metrics_all_models.csv`.

## Article 2 should then

1. Port the frozen model to STM32H7 and ESP32-S3.
2. Measure worst-case latency, RAM, flash, CPU load, and energy on target
   hardware (never host-timing figures from this repository).
3. Design protected direct-wired automotive signal-conditioning interfaces,
   including the wideband-lambda controller/interface.
4. Validate airflow against a temporary calibrated MAF or flow bench.
5. Test the legacy output with HIL and a target ECU.
6. Conduct controlled real-engine learning-drive and fault tests.
7. Evaluate a physically different second engine before making portability
   claims.
