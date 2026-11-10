# Matched DAVIS PerceiverCPI experiment — status 10 November 2026

**Status: official PerceiverCPI training in progress, MATCHED TEST NOT YET EVALUATED.** This document is a progress record and does NOT establish a head-to-head test win.

## Frozen experimental protocol
- Same original public DeepDTA DAVIS labels and exact DynaCPI v5 protein roles. The SHA256 of the original affinity matrix and split manifest are verified in \`prepare_matched_perceivercpi_20261110.py\`.
- 302 training protein reference entries (20,536 pairs), 59 validation entries (4,012 pairs), 81 held-out protein entries (5,508 pairs); all 68 DAVIS compounds occur in training. Approximate protein 3-mer similarity groups, not curated independent protein families.
- Official original dmis-lab/PerceiverCPI source checkout at commit \`755f076ef51099a90332fd1ec8437f94e3750c0b\`. Its original \`InteractionModel\` has 4,672,073 parameters and is not edited.
- Single official seed 2931, 30 planned epochs, batch 128, original LAMB optimizer, dropout 0.1, training/validation MSE metric. This is NOT matched hyperparameter compute or identical architecture/feature capacity to DynaCPI, just matched data and metrics.
- Source directory / execution checkpoint remain LOCAL ONLY in \`matched_perceivercpi_20261110/\`. Training uses a copy of validation as \`--separate_test_path\` because official upstream requires an intermediate test path; this must NEVER be mistaken for independent test accuracy. The actual 81-protein held-out labels were not exported to the training process.
- Official 1-epoch training-only smoke test passed using the original code and tiny train/validation subsets.

## Interim VALIDATION observations ONLY

Best official PerceiverCPI checkpoint is currently selected on 59 validation protein entries (NOT held-out test). Early training history:

| Epoch | Official PerceiverCPI validation pooled MSE |
|---:|---:|
| 0 | 1.019160 |
| 1 | 0.991014 |
| 2 | 0.866547 |
| 3 | **0.612706** |
| 4 | 0.618980 |
| 5 | 0.617962 |
| 6 | 0.595888 |
| 7 | 0.719226 |

In a separately executed full validation-only prediction check, the frozen epoch-3 official PerceiverCPI checkpoint (SHA256 744CC97AFFC34986BDF3D65E7EA4E99C2C0C7504EB7B0761E083828E453025FB) was run through the ORIGINAL upstream predict.py on all 4,012 validation rows. The script analyze_official_perceivercpi_validation_20261110.py confirmed exact SMILES row alignment, read the original 59×68 labels, and calculated target-macro RMSE **0.7532018**, pooled RMSE **0.7827557**, and target-centered MSE **0.4451006**. The original DynaCPI v5 two-seed ensemble on those SAME validation labels scored target-macro RMSE **0.5169132**, pooled RMSE **0.5417198**, and centered MSE **0.2703707** (31.37% lower macro and 30.79% lower pooled RMSE than the frozen epoch-3 checkpoint). The chemical-only validation prior had pooled RMSE **0.6776341**. Official training has since improved its BEST validation pooled MSE to **0.595888** at epoch 6; this one-off snapshot comparison is not the finalized 30-epoch model result.

**Critical caveat:** The official PerceiverCPI run is NOT COMPLETE (only 5 epochs were verified with validation metrics); the 30-epoch run can improve. Moreover, validation was used to SELECT both models, so the observed difference is NOT an independent generalization estimate. Do not compare the values above against DynaCPI's 81-protein TEST score or present this as a publication-ready PerceiverCPI win.

## Matched test procedure (not yet executed)

\`evaluate_matched_perceivercpi_20261110.py\` is a one-time guarded evaluator. It REFUSES to create or access the real 81-protein test CSV until the original full 30-epoch job has completed, its validation-selected \`model.pt\` exists, and its training arguments, upstream source commit, frozen train/validation exports and dataset hashes match. Then it exports the untouched 81×68 test pairs in exactly the v5 order and runs the official \`predict.py\` on the selected checkpoint. It computes pooled RMSE/MSE, target-macro RMSE, target-centered MSE, and differences from DynaCPI v5 on the SAME 5,508 samples. Its output directory refuses to overwrite a prior test result. The safety guard was executed while epoch 6 was running and correctly blocked all access to test labels: \`PASS_TEST_NOT_OPENED_BEFORE_TRAINING_COMPLETE\`.

**When the original 30-epoch run actually completes, run:** \`C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe C:\Users\J0124\DynaCPI-Backend\evaluate_matched_perceivercpi_20261110.py\`.

A single official seed versus a two-seed DynaCPI ensemble is not fully matched. Additional predeclared official seeds and multiple frozen protein partitions would be required for a strong superiority claim, along with scrutiny of censoring assumptions and family grouping.

## Actual held-out DynaCPI v5 reference, for future matched comparison
The **separately tested** DynaCPI v5 ensemble already attained target-macro RMSE **0.58520305698**, pooled RMSE **0.656132** on 81 held-out protein entries. Chemical-only test baseline target-macro RMSE **0.82359633006**, pooled RMSE **0.871410**. The new official PerceiverCPI values remain pending until the 30-epoch training and once-only test run are finished.
