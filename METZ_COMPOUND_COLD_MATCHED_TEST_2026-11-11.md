# Metz compound-cold: first matched held-out test

**Outcome (11 November 2026):** DynaCPI v5 two-seed ensemble achieved lower macro and pooled RMSE than two original unmodified PerceiverCPI seeds on the exact same **6,117 held-out pairs with unseen compounds**. Both paired 95% descriptive cluster-bootstrap intervals cross zero; this is a favorable point estimate, not a statistically robust win.

| Metric (lower better) | DynaCPI v5 | Original PerceiverCPI | Dyna minus original | Paired descriptive 95% interval |
|---|---:|---:|---:|---|
| Per-protein macro RMSE | **0.748509** | 0.764061 | −0.015552 | [−0.051546, +0.024340] |
| Pooled RMSE | **0.781367** | 0.804360 | −0.022993 | [−0.065268, +0.021794] |

Relative RMSE improvements (point estimates): macro approximately **2.0%**, pooled approximately **2.9%**. Both bootstrap confidence intervals span zero. Never call this statistically conclusive.

## Methodology and provenance

- Source cohort: frozen filtered Metz from PADME, labels retain Metz-source affinity convention, not DAVIS pKd or KIBA composite scores.
- Split: preregistered `metz/compound_cold` train/validation/test partitions in `external_benchmarks_20261110/locked_external_protocol_v1`; exact compound grouping based on Murcko scaffolds, not a guaranteed low-Tanimoto chemical holdout. The observed held-out test comprises **6,117 protein-compound pairs**.
- Original PerceiverCPI: official pinned unchanged source SHA `755f076ef51099a90332fd1ec8437f94e3750c0b`, 30 epochs, original model/training algorithm, seeds 2931 and 2932, checkpoints selected on training/validation only, frozen before the individual test.
- DynaCPI: pretrained frozen ESM-2 embeddings + Morgan chemical two-tower LowRankDynaCPI v5. Both selected two-seed checkpoints were frozen before any KIBA/Metz held-out test was opened. Original PerceiverCPI does not share DynaCPI's ESM-2 feature pretraining, so the result is a matched evaluation of *complete systems*, not proof of architecture-specific novelty.
- One-time evaluation: `evaluate_external_ready_scenario_once_20261111.py` performed full six-split integrity preflight (no test labels), verified the two Metz compound-cold official checkpoints, replayed DynaCPI validation, then wrote a new once-only scored result. No retraining or hyperparameter tuning took place. Scheduling amendment was committed before any external test was opened: `EXTERNAL_INCREMENTAL_ONCE_TEST_PROTOCOL_2026-11-11.md`.
- Per-protein macro RMSE and pooled RMSE computed on identical pair rows. All results and paired raw test predictions retained **privately, gitignored**, with SHA256 manifest; never expose raw test labels in the public repo.
- Paired uncertainty: **5,000** fixed-seed, predeclared cluster bootstrap resamples over exact Murcko scaffold groups, using common sampled observations for both model errors. Both intervals include zero, so the direction remains uncertain across resampled chemical groups.
- No claims of paper-fold reproducibility, universal CPI affinity superiority, biologically curated holdouts, clinical prediction, or 3D/equivariant architecture effects.

## Overall external held-out progress at this checkpoint

| Scenario | Held-out macro RMSE (Dyna / Original) | Held-out pooled RMSE (Dyna / Original) | Current conclusion |
|---|---|---|---|
| KIBA unseen proteins | **0.647370** / 0.753914 | **0.608841** / 0.694440 | Lower both; paired intervals favor DynaCPI |
| Metz unseen proteins | **0.751161** / 0.777072 | **0.735788** / 0.772058 | Lower both; intervals cross zero |
| Metz unseen compounds | **0.748509** / 0.764061 | **0.781367** / 0.804360 | Lower both; intervals cross zero |

Remaining: KIBA unseen-compound, KIBA double-cold, Metz double-cold. The original all-six evaluation gate waits for their predeclared official 2-seed checkpoints; it is configured to import existing individual once-only results unchanged, with provenance checks and no rescoring.
