# Matched official PerceiverCPI vs LowRankDynaCPI v5 — FINAL single-split test (10 November 2026)

**Status:** SAME-SPLIT HELD-OUT TEST COMPLETE. **DynaCPI v5 has lower error than the tested official PerceiverCPI checkpoint** on the frozen 81-protein DAVIS evaluation. The result does NOT establish universal superiority over PerceiverCPI across seeds, hyperparameters, family groupings or benchmark folds.

## Exact protocols and provenance

- Dataset: original public DeepDTA DAVIS kinase–inhibitor matrix with 442 protein reference entries, 68 compounds, and 30,056 transformed affinity labels in pKd. Frozen 3-mer-similarity-proxy protein partitions: 302 training proteins (20,536 pairs), 59 validation (4,012 pairs), 81 independent test (5,508 pairs). All compounds occur during training. This is a **cold protein / warm compound** protocol, not a chemistry OOD or double-cold test.
- **PerceiverCPI:** official `dmis-lab/PerceiverCPI` unmodified `InteractionModel`, original source commit `755f076ef51099a90332fd1ec8437f94e3750c0b`; one run seed 2931, batch 128, 30 epochs, original LAMB optimizer, validation-based best-checkpoint selection (best validation pooled MSE 0.556168, at epoch 20). Trained for 30 epochs and completed. Checkpoint SHA256: `07345889ba947e4994c70827ee89491e66283cbbe18cc13fff1e9be4d3a6bf5f`. Actual held-out prediction executed separately with original upstream `predict.py`, after training; training's mandatory intermediate test path was a duplicate of the validation CSV, **not the actual 81-protein test**.
- **LowRankDynaCPI v5:** separately frozen validation-selected 2-seed ensemble trained on 302 protein entries; selected checkpoint epochs 29 and 24. ESM-2 frozen 650M protein features + radius-2 chiral Morgan fingerprints + 32D dual tower. Neither implementation has an equivalent model architecture or feature pretraining budget. Both are evaluated using the exact same 81 DAVIS protein entries and labels, 5,508 pairs. Original pre-frozen v5 results reused; the v6 all-DAVIS refit is **not** included.
- **Exact one-time holdout run:** local guarded `evaluate_matched_perceivercpi_20261110.py`, which verified source commit, SHA256s of raw DAVIS labels, frozen split manifest, training/validation CSVs and original checkpoint. It refused to open the heldout set before all 30 epochs ended. Local output `matched_perceivercpi_20261110/official_matched_v3_HOLDOUT_ONCE/METRICS_MATCHED_TEST.json` has `evaluation_status = SINGLE_FROZEN_MATCHED_HELDOUT_TEST`. Raw held-out CSV, official predictions, model checkpoint and training logs are local only and gitignored.

## Independent 81-protein TEST results

| Test metric, lower better | DynaCPI v5 (2-seed) | Official PerceiverCPI (one seed) | Relative DynaCPI advantage |
|---|---:|---:|---:|
| Target-macro RMSE | **0.5852030569843681** | 0.8893046479245454 | **34.20% lower** |
| Pooled RMSE | **0.6561323403688101** | 0.9355491608712206 | **29.87% lower** |
| Pooled MSE (computed as pooled RMSE²) | **0.430510** | **0.875252** | approximately 50.81% lower |
| Target-centered MSE | **0.3788165324639691** | 0.7357082915323715 | approximately 48.51% lower |

Target-macro RMSE averages each protein's RMSE across its 68 compounds; pooled RMSE aggregates all 5,508 measured pairs. Since every protein has 68 compounds, both are meaningful but distinct metrics. For a literature comparison reporting MSE, square the pooled RMSE—not the target-macro average.

PerceiverCPI's printed training-run 'test mse = 0.556168' was actually its validation-proxy MSE and MUST NOT be labeled real test accuracy. Its true one-time held-out pooled test MSE is approximately **0.875252**, pooled RMSE **0.935549**. DynaCPI v5's 0.656132 is independent held-out pooled RMSE, not final v6 training fit.

## Interpretation and caveats

**Supported finding:** On this one pre-existing, approximate protein-similarity-grouped DAVIS partition, a validation-selected 2-seed DynaCPI v5 ensemble scored substantially lower MSE/RMSE than one validation-selected official PerceiverCPI 30-epoch checkpoint on the same 5,508 untouched held-out pairs. This is an actual matched-split experimental finding, not just numerical juxtaposition with a published baseline from a different test.

**Caveats:**
1. PerceiverCPI currently has **one official seed** while DynaCPI has **two seeds ensembled**. To compare identical compute/ensemble settings, run another predeclared PerceiverCPI seed and ensemble prediction before an additional independent test, rather than optimizing on this exposed test.
2. One approximate similarity-group split is **not** multiple independent curated kinase-family partitions. The 3-mer similarity proxy can leave correlated kinases in train vs held out. No family-stratified bootstrap confidence interval or error significance was established.
3. Official PerceiverCPI hyperparameters were largely defaults, not independently optimized on this split; the result may partly reflect architecture adaptation, compute, pretraining, or tuning choices.
4. All chemical identities were warm. There is no cold-compound generalization result from the final v6 model and no claim that the proposed GVP or K-shot components improved the result.
5. Affinity pKd lower clamp/censoring heuristics and assay conventions remain limitations; no external binding-label validation exists for the six new ligands in the separate unseen-compound demo.
6. This is NOT the same protocol as the published PerceiverCPI DAVIS five-fold figures, so do not state that the published paper's reported benchmark has been reproduced or universally surpassed.

## Reproduction and sources

- Code and status: `prepare_matched_perceivercpi_20261110.py`, `evaluate_matched_perceivercpi_20261110.py`, `analyze_official_perceivercpi_validation_20261110.py`, `MATCHED_PERCEIVERCPI_STATUS_2026-11-10.md`, official upstream https://github.com/dmis-lab/PerceiverCPI.
- Private outputs on research PC: `C:\Users\J0124\DynaCPI-Backend\matched_perceivercpi_20261110\official_matched_v3_HOLDOUT_ONCE\METRICS_MATCHED_TEST.json`, `predictions_official.csv`, `test_exact_once.csv`, `prediction.log`, and `PREDICTION_COMMAND.json`.
- Model checkpoint: `C:\Users\J0124\DynaCPI-Backend\matched_perceivercpi_20261110\official_matched_v3_seed2931_30ep\fold_0\model_0\model.pt` (SHA256 above).
- All raw data, private split manifests, checkpoints, test prediction rows and logs deliberately excluded from Git.
