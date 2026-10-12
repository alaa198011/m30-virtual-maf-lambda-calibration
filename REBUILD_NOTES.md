# Corrected v2 rebuild

This package is a non-destructive rebuild of the original virtual-MAF study.
It implements the code-review recommendations without overwriting the source
archive.

## Main corrections

1. The simulated ECU injector command is generated from an imperfect nominal
   speed-density fuel map, not hidden airflow.
2. Physical lambda is calculated from hidden cylinder airflow and an
   independent actual injector/wall-film plant.
3. The calibration observer uses separate nominal injector parameters.
4. Fixed delay and controller low-pass filtering are modeled separately; the
   FIFO has an exact sample delay and an explicit warm-up validity period.
5. The base estimator and offline hidden comparator are conditioned by the
   same delay/filter path before proposal and acceptance calculations.
6. Fisher information is reset at proposal boundaries and on invalid samples;
   its score therefore belongs to the candidate's current proposal window.
7. C4 calculates its proposal from aligned hidden airflow and is now a real
   non-deployable oracle comparator.
8. Rate limiting uses actual elapsed time since the last accepted commit.
9. Learning begins only in the declared post-shift adaptation segment.
10. E6 budgets cumulative Stage-A-valid samples rather than update count.
11. Dropout uses sample-and-hold with an explicit status bit; permanent stuck
    faults are exposed and close Stage A. Fault rates are interpreted per
    second rather than once per simulation sample.
12. NRMSE uses a declared 260 g/s normalization range. Missing transient,
    adaptation, pseudo-reference-quality, parameter-trajectory, and acceptance
    outputs are now generated.
13. Model resource bytes include scaler mean/scale parameters, and legacy
    output shaping is evaluated on deployed model estimates rather than hidden
    airflow.
14. Training uses `partial_fit` epochs so Adam optimizer state is preserved.
    Artifacts record their scikit-learn version; Python 3.13 is permitted.

## Verification completed

- Clean locked dependency environment: passed `pip check`.
- Source compilation: passed.
- Unit/integration tests: 51 passed.
- Coverage: 51% overall; `online_calibration.py` 73%, sensor model 100%, fuel
  and delay path 97%.
- End-to-end quick pipeline: passed all seven stages and regenerated data,
  five-seed models, metrics, statistics, acceptance report, and figures.
- Quick-run pseudo-reference delay-consistent R2: approximately 0.99.
- Quick-run C3 improvement over C0 on final-test shift trajectories: about
  21.4%; this is diagnostic only and must not replace the full experiment.

## Honest acceptance status

The quick-mode `acceptance_report.json` is intentionally allowed to report
`FAIL`. In the included run, C1 and C3 both had zero false accepted updates, so
the strict requirement that C3 be *lower* than C1 cannot pass (0 is not lower
than 0). Do not change this rule after inspecting final-test outputs. The full
study must determine whether the strict comparison is supported; otherwise the
article must report the tie and revise the hypothesis before a new frozen run.

## Publication procedure

Create a fresh Debian 13 virtual environment, install
`requirements-lock.txt`, run the tests, archive the frozen configurations,
then execute `make pipeline`. The provided quick outputs demonstrate software
operation only.
