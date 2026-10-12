# Article 1 structure (roadmap Section 16)

1. **Abstract** — problem, direct sensor inputs, M30B35 virtual case,
   residual-AI method, calibration gate, principal numerical results,
   simulation limitation.
2. **Introduction** — motivation for virtual airflow sensing; narrow
   research gap; the contribution is *identifiability-aware,
   reference-MAF-free bounded adaptation*, not speed-density estimation or
   neural prediction alone.
3. **Related work** — speed-density models, virtual sensors, residual
   learning, adaptive observers, lambda-guided calibration, embedded
   inference; distinguish simulation evidence from physical-engine evidence.
4. **Problem definition and observability boundary** — estimator inputs,
   calibration feedback, hidden reference airflow, unknown parameters,
   fixed assumptions, non-identifiable confounders.
5. **BMW M30B35 virtual plant** — engine assumptions, displacement, VE
   surface, manifold/thermal dynamics, fuel/lambda path, sensor faults,
   mismatch cases.
6. **Estimator and calibration gate** — physics baseline derivation,
   residual model, scale correction, scalar Fisher-information score, fixed
   valid-sample window, threshold, delay model, proposal rule, validation
   buffer, rollback mechanism.
7. **Experimental design** — datasets, run-level splits, baselines, seeds,
   threshold-freeze rule, calibration modes, mismatches, metrics,
   statistical analysis.
8. **Results** — E1-E7 in order; E5-E6 as the central evidence for the
   proposed novelty.
9. **Discussion** — model mismatch, calibration observability, failure
   modes, data requirements, RLS comparison, legacy-output emulation limits.
10. **Limitations and Article 2 plan** — see `limitations.md` and
    `article2_handoff.md`.
11. **Conclusion** — virtual-study conclusions only; no universal,
    real-time, or physical-compatibility claims.

Figures and result files referenced throughout this outline are produced by
`src/make_publication_figures.py` from the CSV/parquet files listed in
`README.md#required-software-outputs`.
