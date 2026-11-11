# KIBA + Metz external model generalization — 10 November 2026

**HISTORICAL SNAPSHOT (10 November 2026).** The status below was true before the first held-out scores were opened. **Updated 11 November 2026:** Both KIBA and Metz protein-cold matched two-seed held-out comparisons have completed and favor DynaCPI on macro and pooled RMSE point estimates. KIBA descriptive paired 95% intervals favor DynaCPI; Metz intervals include zero. **Metz compound-cold also completed a frozen held-out comparison on 11 November, DynaCPI winning both RMSE point estimates but bootstrap intervals crossing zero.** Three remaining external splits are sealed: KIBA compound-cold, KIBA double-cold, Metz double-cold. See [new external test report](KIBA_METZ_MATCHED_PROTEIN_COLD_TEST_RESULTS_2026-11-11.md) and [transparent one-time protocol amendment](EXTERNAL_INCREMENTAL_ONCE_TEST_PROTOCOL_2026-11-11.md).

STATUS AS OF 10 NOVEMBER: All six LowRankDynaCPI external two-seed VALIDATION branches completed. No KIBA/Metz original PerceiverCPI held-out comparison had yet completed at that time. All six newly frozen tests were SEALED.

## Source provenance
KIBA: hkmztrk/DeepDTA git a546a8433a6822e958f36171c4356ad6f414d623, 229 protein sequences x 2,111 source compounds, 118,254 observed KIBA composite scores, NOT pKd. This differs from the published PerceiverCPI-filtered subset of 2,068 compounds / 117,657 scores; no claim of published-fold reproduction.

Metz: simonfqy/PADME git e01c592cc06c4de04b3ed6db35da5af5ff7f863f; filtered triplet_split_warm_filtered.csv 170 proteins x 1,423 compounds, 35,259 nonduplicate observed numeric-affinity pairs, not the unfiltered 35,307-row table. Endpoint units retained as provided. Unobserved pairs were NEVER imputed as zeros or negative interactions.

Exact original source SHA256 files, frozen train/validation CSV hashes, label-scale convention, sequence/chemical reference order and immutable split manifests live under the ignored local external_benchmarks_20261110/locked_external_protocol_v1 directory. The pre-existing authorized PC's frozen ESM-2 650M sequence-hash cache ALREADY contained embeddings for ALL KIBA and ALL Metz proteins. New chemical features are radius-2 chiral Morgan 2,048-bit, no preprocessing fitted on test labels.

## Predeclared six splits with test labels not exported

| Dataset / scenario | TRAIN observed pairs | VALID observed pairs | SEALED TEST observed pairs |
|---|---:|---:|---:|
| KIBA protein-cold | 80,951 | 16,548 | 20,755 |
| KIBA compound-cold | 75,367 | 18,988 | 23,899 |
| KIBA double-cold | 53,045 | 3,166 | 3,442 |
| Metz protein-cold | 24,464 | 4,814 | 5,981 |
| Metz compound-cold | 23,378 | 5,764 | 6,117 |
| Metz double-cold | 15,320 | 658 | 1,416 |

Protein groups are connected TF-IDF character 3-mer cosine >= 0.35, NOT certified biological family boundaries. Compound groups use RDKit Bemis–Murcko scaffold, not guaranteed low fingerprint similarity. Splits were fixed prior to ALL new training in prepare_external_kiba_metz_frozen_benchmarks_20261110.py. Neither paper fold equivalence nor universal generalization is claimed.

## Frozen two-seed DynaCPI VALIDATION performance (not test!)

Adaptation preserves original LowRankDynaCPI v5 neural architecture, train-only mean-centered 1280D frozen ESM-2 protein features, 2048D Morgan, 32D two-tower interaction head; uses fixed two seeds 2931/2932, AdamW LR 0.001 WD 0.03, batch1024, max45 epochs, validation macro-RMSE selected checkpoints. KIBA and Metz use standard uncensored MSE on each dataset's own numeric scale and NO DAVIS pKd=5 clamp. Chemical anchor is calculated only from OBSERVED TRAINING compound affinity means; novel ligands use a predeclared top-16 squared-Morgan-Tanimoto nearest-neighbor prior. There are no test-label queries in training, preprocessing, prior computation or checkpoint selection.

| Dataset / scenario | Dyna macro RMSE | Train-only chemical prior macro | Dyna pooled RMSE | Prior pooled | Chosen epochs |
|---|---:|---:|---:|---:|---|
| KIBA protein-cold | 0.657419 | 0.691480 | 0.630686 | 0.682226 | 6, 3 |
| KIBA compound-cold | 0.642766 | 0.730280 | 0.627424 | 0.727890 | 7, 5 |
| KIBA double-cold | 0.781430 | 0.789900 | 0.758663 | 0.755423 | 1, 2 |
| Metz protein-cold | 0.707470 | 0.738776 | 0.690724 | 0.728157 | 1, 2 |
| Metz compound-cold | 0.758222 | 0.907310 | 0.784928 | 0.948809 | 3, 10 |
| Metz double-cold | 0.777782 | 0.789940 | 0.778389 | 0.803480 | 3, 1 |

NEGATIVE: On KIBA double-cold validation, DynaCPI's pooled RMSE 0.758663 is slightly WORSE than training-only prior's 0.755423. Retain this result, do not choose another split or model after test exposure. Relative to original published PerceiverCPI, NO external comparison yet exists. KIBA and Metz numerical errors are on different label scales; never compare their magnitudes directly.

## The original original-code PerceiverCPI benchmark

Unmodified dmis-lab/PerceiverCPI source SHA 755f076ef51099a90332fd1ec8437f94e3750c0b, 30 epochs, original InteractionModel, original LAMB, batch128, dropout0.1 and fixed two training seeds 2931/2932. Exactly the SAME train/validation pair rows for each external scenario. The original train.py program requires an intermediate test input; it is supplied only a byte-identical copy of VALIDATION, not the protected unseen test pairs. Its printed training-run test MSE is therefore a VALIDATION PROXY, NOT generalization evidence.

The first Metz protein-cold original baseline seed 2931 started at ~20:37 IST on 10 November 2026 and was active at epoch2 during the last check. Generic audited original runner: run_external_original_perceiver_frozen_20261110.ps1. A local Windows sequential queue, run_remaining_external_perceiver_queue_20261110.ps1, was launched at ~20:40 IST to train the other 11 original external baseline seeds after the first is complete, using a >=2.80GB free SYSTEM RAM guard. A second LOCAL evaluation gate, run_external_sealed_test_gate_20261110.ps1, was launched at ~20:49 IST to wait for all 12 original model checkpoints, then run a strict no-test-labels preflight, replay all six frozen DynaCPI validations, and only then run the once-only six-scenario matched original-model test. The public evaluate_external_kiba_metz_all_six_once_20261110.py script can also be run in preflight/validate-only modes. The validate-only replay was executed and exactly reproduced ALL six frozen DynaCPI validation macro RMSEs before any test opening. These OS processes may run beyond this conversation, but they do not create automatic ChatGPT messages or guarantee completion if the PC disconnects, sleeps, runs out of RAM or the code fails; any failure stops and preserves logs. This is parallel to the existing DAVIS official training queue, not additional simultaneous heavy models. Each original PerceiverCPI Python process consumed approximately 1.5–1.7GB RAM. One connected GPU (RTX 4060 Laptop, 8GB) is accessible on this PC. Last observed GPU load varied 4–60%; low momentary utilization is NOT proof of safe extra RAM or throughput.

## Validation-only representation sanity checks (10 Nov)

In addition to model-vs-chemical-prior VAL comparisons above, we ran five fixed protein-feature permutations and five fixed learned-chemical-fingerprint feature permutations on each of the six original checkpoint-selected validation splits, leaving true assay scores, pair IDs and training-derived chemical anchors untouched. Both permutations increased mean validation macro RMSE on all six cohorts; the full technical explanation, exact numbers and retained weak double-cold observations are in EXTERNAL_TWO_TOWER_FEATURE_RELIANCE_2026-11-10.md. This is only an out-of-distribution perturbation negative control, not a causal demonstration or independent test. All raw sealed test labels remain untouched.

## Claim rules
Only claim a KIBA/Metz win after completing validation-selected TWO official PerceiverCPI checkpoints for the SAME frozen scenario, then evaluating BOTH two-seed model ensembles once on the EXACT SAME SEALED test rows. Report macro RMSE per protein over OBSERVED pairs and pooled RMSE, with paired uncertainty and every negative result. Do not compare to original published PerceiverCPI figures from different KIBA filters or folds as if they were matched. Multiple positive public benchmark results can support specific conditional generalization, NEVER universal superiority. There are no clinically validated claims. Separate original DAVIS v5 win and full-data v6 train-fit as distinct evidence.
