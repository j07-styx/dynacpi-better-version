# KIBA / Metz incremental sealed evaluation amendment — 11 November 2026

## Status and timing
This is an explicit **prospective evaluation-scheduling amendment** created **before any KIBA/Metz held-out test results were scored**. The original six-scenario test plan waited for all twelve original PerceiverCPI checkpoints to finish before permitting a single six-scenario held-out evaluation. That delay was not necessary for the mathematical same-split comparison of independently frozen scenarios. The amendment permits one-time scoring of a scenario once its two original PerceiverCPI validation-selected checkpoints are complete. It does not change any train/validation/test assignments, feature preprocessing, model, seed, hyperparameter, model-selection rule, or metric.

**Scientific limitation:** Individual test outcomes can become visible before other official baseline training is finished. This deviates from the stronger all-at-once blinding protocol. All six DynaCPI two-seed checkpoints and corresponding validation selections were already frozen before this change, and the original PerceiverCPI training commands for each scenario were already fixed. No test outcome may be used to select/retrain a model or alter any remaining evaluation. Report this amendment in any manuscript.

## Eligible frozen scenarios now
- **Metz / protein_cold:** original PerceiverCPI seeds 2931 and 2932 completed; original source and frozen split verified before test.
- **KIBA / protein_cold:** original PerceiverCPI seeds 2931 and 2932 completed; original source and frozen split verified before test.
- The other four KIBA/Metz scenarios remain sealed until both original checkpoints for that scenario are finished.

## No-test validation comparisons completed before amendment
These are *not* sealed-test results. Both models were selected on validation, so the validation differences are descriptive and may be optimistic.

| Split | DynaCPI 2-seed macro RMSE | Official PerceiverCPI 2-seed macro RMSE | DynaCPI 2-seed pooled RMSE | Official PerceiverCPI 2-seed pooled RMSE |
|---|---:|---:|---:|---:|
| Metz protein-cold (4,814 validation pairs) | 0.707470 | 0.748272 | 0.690724 | 0.748820 |
| KIBA protein-cold (16,548 validation pairs) | 0.657419 | 0.749951 | 0.630686 | 0.723876 |

Source code: `audit_original_external_validation_against_frozen_dyna_20261111.py`. Exact scored validation JSON, per-seed checkpoint SHA256s and private predictions are stored under each gitignored dataset/split directory in `ORIGINAL_VS_DYNA_VALIDATION_ONLY_20261111`.

## Per-scenario once-only protocol
1. Verify the pinned unmodified original source SHA, external source feature hashes, immutable six-way split manifests, CSV SHA256 values and all six original DynaCPI validation freezes. Do not inspect any test label.
2. Verify both *specific* original PerceiverCPI seeds have fully finished their original 30-epoch train-and-validation runs, with exact checkpoint hashes and validation-proxy CSV, and replay that scenario's DynaCPI selected validation predictions.
3. Refuse operation if a scenario's individual output directory exists, whether it is finished or incomplete; refuse if the final six-scenario aggregate has already begun.
4. Create its unique ignored private result directory, then and only then read that scenario's held-out indices once, run exact original `predict.py` for two official seeds and frozen DynaCPI inference on identical observed pair rows, and compute macro-per-protein RMSE and pooled RMSE.
5. Preserve checkpoint, data, predictions and test-row hashes, raw paired scores (private) and immutable `TEST_RESULTS.json`. Never use scored test labels to tune a model. A failed partial test is preserved and not restarted automatically.
6. The *existing* all-six gate continues waiting for all 12 official checkpoints. Its evaluator has been amended to verify and **copy previously scored individual results without reopening or re-scoring their test rows**; remaining unscored scenarios are evaluated by the original all-six code. Its paired-bootstrap summaries are then applied uniformly to the completed six-scenario output.

Code: `evaluate_external_ready_scenario_once_20261111.py`; updated `evaluate_external_kiba_metz_all_six_once_20261110.py`. Both were syntax checked, and the two eligible-scenario **no-test-label preflights passed** before the individual scoring procedure.

## Reporting rules
- Negative deltas mean DynaCPI has lower error than the official original PerceiverCPI on the matched split.
- Do not report validation improvements as independent held-out evidence.
- Do not compare KIBA composite-score RMSE numerically to Metz endpoint RMSE or the original DAVIS pKd RMSE.
- Original PerceiverCPI is not supplied the ESM-2 pretrained feature bank used by DynaCPI; the comparison of complete systems does not isolate architecture novelty or equal pretrained compute.
- KIBA and Metz cohorts and deterministic groups are not claimed to reproduce a paper's original evaluation folds or guarantee disjoint biological families.
- Where paired uncertainty has not yet been computed, describe point-estimate results as preliminary, not conclusive universal improvement.
