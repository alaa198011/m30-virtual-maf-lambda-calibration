# Limitations (roadmap Sections 1, 9, 18, Final recommendation)

This repository establishes **simulation-level feasibility only**. It does
not establish:

- physical engine accuracy (the plant is virtual, not a physical M30B35);
- real-time operation on an STM32H7 or ESP32-S3 (all timings recorded by
  `src/legacy_output.py` / `results/legacy_output_metrics.csv` are HOST
  timing, explicitly labeled as such — never MCU timing);
- physical ECU acceptance (the legacy-output transfer function is a
  software emulation, not a validated hardware interface);
- universal engine compatibility (only the BMW M30B35 virtual case is
  studied; portability to a second, physically different engine is
  reserved for Article 2);
- improved real-world fuel economy, torque, emissions, knock margin, or
  drivability (no combustion, torque, or emissions model is included);
- independent identification of VE error, MAP gain, lambda bias,
  injector-flow error, and fuel-pressure error from lambda feedback alone
  (only a single bounded global scale is adapted — see
  `configs/calibration.yaml#parameterization` and the observability
  boundary statement in Section 4 of the roadmap).

The calibration layer's scalar Fisher-information gate (`src/scalar_fisher_gate.py`)
is a practical information-sufficiency test under the declared
nuisance-parameter assumptions in `configs/calibration.yaml`. It is **not**
proof of global identifiability.
