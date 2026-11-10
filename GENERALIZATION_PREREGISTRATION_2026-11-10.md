# DynaCPI versus PerceiverCPI — generalization pre-registration, 10 November 2026

## Goal and claim standards

Evaluate whether LowRankDynaCPI has repeatable generalization advantages against the unmodified published PerceiverCPI implementation, using identical data splits, equal ensemble sizes, multiple chemical/protein novelty conditions and unbiased test reporting. "Universally superior" cannot be proven from finite benchmark experiments; negative comparisons are required to be kept and explained.

This document clearly separates (1) POST-HOC examination of a previously exposed DAVIS held-out split from (2) FROZEN new partitions with test labels not accessed.

## 1. Existing single-split win and post-hoc robustness

Original DAVIS 302 training / 59 validation / 81 held-out protein reference entries with all 68 compounds warm, 5,508 held-out pairs. DynaCPI v5 two-seed ensemble: target-macro RMSE 0.5852031, pooled RMSE 0.6561323. Original PerceiverCPI one seed 2931, 30 epochs, validation-selected epoch 20: target-macro RMSE 0.8893046, pooled RMSE 0.9355492. Same exact 81 proteins and 5,508 labels.

The floor-bias audit reconstructed v5 test outputs, aligned every original Perceiver prediction row by compound SMILES, and calculated:

- 3,868 / 5,508 (70.225%) held-out labels equal pKd=5 (weak binding floor).
- Clamping original PerceiverCPI predictions below 5 reduces its pooled RMSE to 0.928311, versus unchanged DynaCPI 0.656132.
- On 1,640 non-floor affinity measurements only: DynaCPI RMSE 1.134500; official original PerceiverCPI 1.441167; floor-clamped PerceiverCPI 1.433809.
- On 3,868 floor-valued pairs: DynaCPI RMSE 0.259473; official original PerceiverCPI 0.604762; floor-clamped PerceiverCPI 0.596234.
- Protein-wise macro RMSE better for DynaCPI on 69/81 entries against original PerceiverCPI; 68/81 after PerceiverCPI floor clamping.
- Paired, 12,000-resample 57-sequence-group bootstrap produced DynaCPI-minus-PerceiverCPI target-macro delta descriptive 95% percentile CI [−0.4492, −0.1810]; after floor clamping [−0.4391, −0.1679].

ALL RESULTS ABOVE are EXPLORATORY SENSITIVITY ANALYSES on the SAME exposed held-out test, not further independently verified tests. These post-hoc confidence intervals are descriptive, not predeclared hypothesis testing. Source: audit_matched_test_floor_bias_20261110.py; full results stored locally in ignored POSTHOC_FAIR_FLOOR_AUDIT_20261110.json.

## 2. Equal ensemble-size replication — current original split

Official PerceiverCPI is unmodified dmis-lab/PerceiverCPI source at SHA 755f076ef51099a90332fd1ec8437f94e3750c0b, using official LAMB optimizer, dropout 0.1, batch 128, 30 epochs, best epoch selected on validation pooled MSE only. First official seed 2931 is complete. Second original official seed 2932 STARTED at 19:18:53 IST, 10 Nov 2026; running on the local Windows research PC in matched_perceivercpi_20261110/official_matched_v4_seed2932_30ep. Only the original fixed 20,536 train and 4,012 validation affinity pairs are provided to upstream training; the apparent training "test" is a byte-identical validation proxy.

When BOTH official training runs finish, evaluate_perceiver_2seed_parity_20261110.py is ready to compute a two-seed official ensemble and compare it to the frozen two-seed v5 ensemble on the previously used 81-protein holdout. It refuses to run without completed 30-epoch training and checkpoint SHA checks. This removes seed-count asymmetry, but the test is already exposed and cannot be relabeled a new independent test. No tuning using this test.

## 3. NEW outcome-blind DAVIS test partitions

Predeclared in prepare_preregistered_davis_generalization_20261110.py. Each suite partition has a separate immutable local split manifest, SHA256, training CSV and validation CSV; test-label CSVs were NEVER EXPORTED. Suite index: generalization_suite_20261110/SUITE_LOCKED.json (local only). Raw public DAVIS Y SHA256 9855c4f234ec6dd18295b0e0cd6ee54214cafa908c6eb1d447b47d454fc0294a.

Scenario | Train affinity pairs | Validation pairs | Unscored test pairs
- protein_strict_20261113 | 20,332 | 5,236 | 4,488 (66 proteins x 68 warm compounds)
- protein_strict_20261114 | 19,584 | 4,624 | 5,848
- compound_scaffold_20261115 | 19,890 | 4,420 | 5,746
- double_cold_20261116 | 12,780 | 550 | 1,339

Protein novelty proxy is connected 3-mer TF-IDF cosine >= 0.35, creating 268 similarity groups, instead of the older threshold 0.90 creating 378 groups. This is STILL NOT verified curated kinase-family grouping. RDKit exact Bemis-Murcko scaffold grouping finds 68 distinct identifiers for the 68 DAVIS compounds, so scaffold-disjoint here is effectively compound-identity disjoint, NOT a guarantee of chemical Tanimoto dissimilarity. Stronger chemical-neighbor exclusions and external cohorts remain necessary.

Fixed learning plan: DynaCPI v5 previously selected centered_esm_censored variant (frozen), two seeds 2931/2932, AdamW LR 0.001 and WD 0.03, batch 1024, at most 45 epochs, validation target-macro RMSE checkpoint/early stop. Unmodified PerceiverCPI, two fixed seeds 2931/2932, same data pairs, original LAMB optimizer, 30 epochs, batch 128, validation MSE selection. Baseline tuning budgets and pretraining costs differ; do not claim method-level fairness beyond data split and ensemble count without further controls.

PROGRESS: First new stricter protein split DynaCPI training FINISHED with both locked validation-selected seeds, epochs 16 and 11, ensemble validation target-macro RMSE 0.693429727 and pooled RMSE 0.744957173 on 77 validation proteins (5,236 affinity pairs). No new test labels consulted. Source: train_preregistered_dynacpi_v5_protein_20261110.py; checkpoint and frozen selection local only. Official unchanged PerceiverCPI training on this exact split (first seed 2931) STARTED at 19:28:18 IST, 10 Nov. Its validation and test outcomes were not yet available when this pre-registration was written.

## 4. Cold chemistry implementation barrier

Existing DynaCPI v5 code anchors each compound on its TRAINING chemical mean, which cannot be calculated for a held-out scaffold. Before evaluating the scaffold or double-cold scenarios, implement a TRAINING-COMPOUND-ONLY chemical anchor and validation-selected Morgan-Tanimoto extrapolation prior, then retrain both models using strictly disjoint chemical IDs. Never query test-compound affinity labels for prior construction. Report weak-affinity floor/non-floor strata and different dataset endpoints.

## 5. External datasets and decision logic

Prioritize additional independent kinase families, KIBA and METZ where labels and authorized dataset sources are checked. KIBA is a composite score, not pKd; do not compare its numeric RMSE directly with DAVIS pKd RMSE. Ensure consistent train/validation/test partitions and report each endpoint independently.

Primary metric per split: target-macro RMSE; co-primary sensitivity: pooled RMSE. Secondary: non-floor RMSE, target-wise wins/losses, CI, MAE, cold-chemistry checks, and group-resampled paired confidence intervals. A positive result on a split requires both macro and pooled RMSE improvements on the SAME independently trained test labels, after fixed validation-only selection. Present every positive and negative result, no cherry-picking folds or seeds. Finite evidence supports "lower error on these tested protocols," not "universally superior."

Original pretrained ESM-2 and unequal training budget versus PerceiverCPI may account for part of the effect; isolate through ablations before claiming novel architectural mechanisms. Proposed GVP 3D geometry and K-shot support routing are NOT in these trained low-rank models and cannot be credited for the current results.

## Reproducibility status

Live training tasks are processes on the authorized Windows PC, not a guarantee of future push notifications or unattended ChatGPT delivery. The repository includes code and this public-safe protocol. Exact raw DAVIS features, immutable split CSVs, checkpoints, prediction rows and full logs remain locally protected and gitignored.
