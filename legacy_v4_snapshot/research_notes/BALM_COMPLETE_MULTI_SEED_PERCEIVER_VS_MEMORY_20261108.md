# DynaCPI v4 — completed matched PerceiverCPI and memory evaluation

**2026-11-08 | Experimental development evidence, NOT a valid prospective-test or paper-level win.**

## Confirmed matched inputs and reproducibility

- The same 3,891 existing BALM inner-dev examples and 173 original protein 3-mer proxy groups; SHA256 61895efb20aa386c3b59ddf81bff98da9fd2d708c9dda0a7ac60d44c91fcc2b1.
- All 8 completed PerceiverCPI runs share this exact row hash. PerceiverCPI is the original InteractionModel, with a 500-token protein input and fixed training configuration.
- Six distinct 10-epoch seeds (2103, 2601–2605), and two 30-epoch seeds (2601–2602). The first 10 epochs for matching seeds 2601 and 2602 have exactly identical saved predictions between 10-epoch and 30-epoch runs.
- No reserved prospective test data was loaded. Both ensembles fit only the 15,538 original TRAIN records (12,315 mean-collapsed exact pairs).

## Core matched results (lower RMSE is better)

| Candidate | Pooled RMSE | Protein-group macro RMSE |
|---|---:|---:|
| Locked Ridge + explicit interactions | **1.383743** | **1.247098** |
| PerceiverCPI 10 ep, mean over 6 final seeds | 1.512232 ± 0.077057 (seed SD) | 1.372680 ± 0.068211 |
| PerceiverCPI 30 ep, mean over 2 final seeds | 1.589725 ± 0.028047 (seed SD) | 1.412739 ± 0.052934 |
| Exploratory ESM2 8M + fixed joint memory k64 | 1.382733 | 1.239977 |
| Exploratory ESM2 8M base without memory | 1.383848 | 1.233368 |
| Mean training-label prediction | 1.589844 | 1.464093 |

The Perceiver values above are averages of **individual seed RMSEs**, NOT RMSE of averaged predictions. Error bars represent seed-level sample standard deviation. Seed-averaged-prediction ensembles are evaluated separately below.

## Paired 5,000-draw protein-group bootstrap: locked ensemble minus PerceiverCPI

| PerceiverCPI comparator | Δ pooled RMSE [95% group interval] | Δ macro RMSE [95% group interval] |
|---|---|---|
| 10ep final seed mean, PRIMARY | -0.128489 [-0.208675, -0.061076] | -0.125583 [-0.165532, -0.084237] |
| 10ep dev-best epoch, OPTIMISTIC | -0.114538 [-0.202646, -0.041857] | -0.122294 [-0.166229, -0.077673] |
| 10ep averaged seed predictions | -0.068999 [-0.146171, -0.006192] | -0.060410 [-0.103381, -0.016194] |
| 30ep final seed mean, PRIMARY | -0.205982 [-0.346238, -0.092791] | -0.165641 [-0.222981, -0.106889] |
| 30ep dev-best epoch, OPTIMISTIC | -0.172153 [-0.290762, -0.073329] | -0.148460 [-0.201140, -0.092912] |
| 30ep averaged seed predictions | -0.180456 [-0.315603, -0.069805] | -0.145149 [-0.202996, -0.086075] |

Negative deltas favor our ensemble. Both prespecified PRIMARY 95% intervals exclude zero for pooled and protein-group macro RMSE at both training budgets. Even optimistic selection of PerceiverCPI's best development epoch per seed leaves an observed gap in this specific exploratory split.

## Individual neural-seed training behavior

| Epoch budget | Seed | Final pooled RMSE | Final macro RMSE | Best DEV epoch / pooled RMSE |
|---:|---:|---:|---:|---|
| 10 | 2103 | 1.449942 | 1.292871 | 7 / 1.431752 |
| 10 | 2601 | 1.555695 | 1.413269 | 6 / 1.549967 |
| 10 | 2602 | 1.627243 | 1.465610 | 5 / 1.590455 |
| 10 | 2603 | 1.436256 | 1.324985 | 7 / 1.413254 |
| 10 | 2604 | 1.453571 | 1.323210 | 10 / 1.453571 |
| 10 | 2605 | 1.550687 | 1.416138 | 10 / 1.550687 |
| 30 | 2601 | 1.569893 | 1.375309 | 12 / 1.539217 |
| 30 | 2602 | 1.609557 | 1.450169 | 14 / 1.572575 |

The 30-epoch runs did not improve on their DEV-best scores; seed 2601 best epoch 12 and seed 2602 best epoch 14. Both 30-epoch runs have identical predictions to their corresponding 10-epoch runs through epoch 10. This is late-training DEV degradation at these seeds/settings; it does not prove the original architecture is intrinsically weaker after tuning.

## Strong alternative control: DeepDTA-inspired CNN

The prior **DeepDTA-inspired single seed** on this same existing DEV partition recorded final epoch 10 RMSE 1.436926 / macro 1.293077. Its best DEV-selected epoch 8 pooled RMSE was **1.371757**, numerically BETTER than our locked ensemble's 1.383743. This is a DEV-optimized epoch selection, not a locked fair winner, and per-row DeepDTA predictions are unavailable for the matched group bootstrap. Do NOT claim all baselines are beaten.

## Exploratory memory model on exact same DEV rows

Frozen ESM2 8M representations were generated for 209 DEV-only protein sequences (30 truncated at 1,022 residues); neither dev target labels nor reserved test inputs were used for feature extraction. The trained model itself was already fitted only to original TRAIN. Memory+ESM2 fixed k64 yields 1.382733 pooled and 1.239977 macro.

| Exploratory paired comparison (new memory minus reference) | Δ pooled RMSE [95% group interval] | Δ macro RMSE [95% group interval] |
|---|---|---|
| Joint ESM memory vs locked classic | -0.001010 [-0.018954, +0.016191] | -0.007120 [-0.021391, +0.006997] |
| Joint ESM memory vs chemical-only memory | +0.002149 [-0.004665, +0.008037] | -0.000699 [-0.005387, +0.004283] |

**Neither memory-vs-locked nor joint-protein-vs-chemical-only paired interval excludes zero on this DEV split.** The memory augmentation did help relative to old architecture in strict TRAIN-only alignment+scaffold cold CV, but this development run does not establish a reproducible advantage. Do not choose from the variants based on this DEV view and then claim an independent win.

## Publication blockers and next decisions

1. The current DEV protein groups use an earlier approximate 3-mer grouping; a separate TRAIN-only Smith–Waterman audit found many cross-group alignments. The DEV comparison also lacks explicit Murcko scaffold exclusions, so it is NOT the same as alignment-and-scaffold-cold TRAIN cross-validation.
2. BALM source labels have unresolved assay provenance, methods, protein constructs and repeated-input measurement disagreements. An independently reviewed assay-harmonized scientific cohort and locked homology/scaffold split are required.
3. The model family and selection thresholds have been inspected repeatedly using development outcomes. Resampling the same 173 groups cannot undo adaptive model selection.
4. The DeepDTA-inspired best-DEV-epoch counterexample precludes blanket claims of superiority; a fair DeepDTA/PerceiverCPI replication requires matched hyperparameter selection and compute budgets.
5. The scientific cohort gate remains BLOCKED_UNAPPROVED_NOT_TRAINING_READY. **Do not score or tune on the reserved prospective test** until approvals and pre-registration exist.

## Reproduction

- Source seed files: research/balm_perceiver_multiseed_dev_20261108/perceiver_seed*_ep*.json (8 reports, 6 ten-epoch and 2 thirty-epoch).
- Prior frozen classical reference: research/balm_ensemble_dev_predictions_20261108.json.
- Python paired comparison: scripts/compare_balm_perceiver_vs_ensemble_dev.py --epochs 10; repeat --epochs 30.
- Novel pretrained DEV inputs feature cache: scripts/cache_balm_frozen_esm8m_innerdev_inputs_dev.py. Binary NPZ kept locally; checksum in research/balm_esm2_t6_8m_innerdev_input_only_manifest_dev_20261108.json.
- Exact same rows memory evaluation: scripts/evaluate_balm_memory_vs_perceiver_matched_innerdev_dev.py and research/balm_memory_vs_perceiver_matched_innerdev_20261108.json.
- Report generator: scripts/summarize_balm_perceiver_multiseed_complete_dev.py.
