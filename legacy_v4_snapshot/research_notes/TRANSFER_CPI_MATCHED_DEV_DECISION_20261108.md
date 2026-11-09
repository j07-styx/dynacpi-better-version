# DynaCPI / TransferCPI — matched development decision checkpoint

**Date:** 2026-11-08. **Status:** completed local exploratory analysis; no prospective scientific evaluation.
**Repository:** `j07-styx/Demo-smth`, `dynacpi-v4`, observed HEAD `44a56820164e6e67df9ebbd749b60f782a3a3d39`.
**Frozen provisional split:** `research/bindingdb_frozen_component_split_manifest_dev.json` (SHA256 `4f66961f2854123346f598b8816344118e7ff764a1389c8f36d505cc66fe4412`), 82 train / 20 inner-dev / 33 reserved prospective-test protein components, 27,118 TRAIN and 7,387 inner-DEV source pairs. Reserved prospective-test components were not retained in train/DEV, used for optimization, or scored; the existing loader streams a shared raw candidate file before role filtering, so this is not a claim that raw test-label bytes were never read.

## Completed matched experiment

The five candidate arms use saved seed-2601, five-epoch training runs on exactly the same train/inner-DEV split, with matched batch size, optimizer, row order, learning rate, weight decay and label scaling. The first model has frozen ESM2-650M compressed residue tokens, molecular graphs, fingerprints/descriptors and bidirectional cross-attention; the dual version additionally uses frozen ChemBERTa molecular representations. The PerceiverCPI arm uses its original 500-token sequence interface. Match in epochs does not equal match in GPU compute or published-paper tuning.

| Arm (checkpoint selected by minimum inner-DEV component-macro RMSE) | Best epoch | Pooled RMSE | Component-macro RMSE |
|---|---:|---:|---:|
| Train-label mean, no fitted neural model | — | 1.803902 | 1.698289 |
| Original PerceiverCPI | 1 | 2.019847 | **1.325136** |
| TransferCPI | 1 | 2.588441 | 1.508015 |
| Chemical-only TransferCPI | 4 | 2.791769 | 1.560025 |
| Dual-pretrained TransferCPI + ChemBERTa | 2 | 2.605404 | 1.600903 |
| Dual-pretrained chemical-only | 4 | 2.755706 | 1.620020 |

The best TransferCPI macro-RMSE difference versus PerceiverCPI is **+0.182879**, with the saved 10,000-draw paired-component bootstrap 95% interval **[-0.041657,+0.449973]** (20 components). The dual-pretrained difference versus PerceiverCPI is **+0.275767**, interval **[+0.062123,+0.515568]**. Positive differences favor PerceiverCPI. The first interval includes zero; these are exploratory, correlated, development-set bootstrap intervals, not independent hypothesis tests.

For pooled error the train-mean reference (1.803902) beats every candidate above. PerceiverCPI has the strongest component-macro point estimate, but no evaluated arm demonstrates satisfactory dominance on both metrics. The two neural training schedules are short and some components are disproportionately large, so avoid definitive architectural claims.

## Mechanistic and numerical checks

- The TransferCPI + chemical-only + PerceiverCPI comparison confirms identical row order and training schedule in `comparison_transfer_seed2601.json`.
- The ChemBERTa variant comparison confirms the same in `comparison_transfer_clm_seed2601.json`.
- The saved perturbation report `perturbation_diagnostics.json` tests shuffled protein features and removing interaction; these are out-of-distribution diagnostics, not causal evidence.
- On this checkpoint, frozen TransferCPI predictions have mean 5.663862 versus mean development label 7.263588, supporting a bias/assay-shift investigation without selecting a new correction on the already inspected DEV.
- Three focused, read-only validation scripts were rerun on the authorized machine and returned exit code zero: `tests_v4/transfer_cpi_test.py` (572,226 parameters), `tests_v4/dual_transfer_cpi_test.py` (619,170 parameters), and `tests_v4/transfer_feature_integrity_test.py` (full ESM coverage and reserved-test exclusion). These demonstrate engineering consistency only.

## Decision and next scientific gate

**HOLD: do not promote TransferCPI, dual TransferCPI or a new model architecture on this evidence.** Preserve all completed runs/checkpoints and the original sequence/geometry work. Do not restart completed output folders or fit an exploratory correction on the repeatedly inspected DEV as if it were independent.

Priority actions: (1) independent source and protein-construct/assay provenance adjudication, (2) publication- and homology-linked leakage-aware split approval, (3) train-only controlled diagnostics of target-wise label distribution and model biases, (4) a locked compute-matched benchmark plan with PerceiverCPI, DeepDTA and mean/chemical baselines, then (5) one reserved prospective-test evaluation only after review and preregistration. No human approvals, approved Kd scientific cohort, or independently validated model superiority are claimed.

Local source-of-truth artifacts: `research/transfer_cpi_dev_20261108/*/COMPLETE.json`, `comparison_transfer_seed2601.json`, `comparison_transfer_clm_seed2601.json`, `perturbation_diagnostics.json`. Large binary checkpoints, per-row predictions and private source worksheets stay local. Formal gate: `BLOCKED_UNAPPROVED_NOT_TRAINING_READY`.
