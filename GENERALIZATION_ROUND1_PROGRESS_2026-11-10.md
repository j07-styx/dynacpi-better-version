# DynaCPI versus PerceiverCPI — experimental results and current suite (10 November 2026)

This report is **not** a universal superiority claim. It records every completed checkpoint-backed result obtained so far, both positive and weak, and distinguishes the previously exposed held-out DAVIS split from newer **VALIDATION-ONLY** results whose sealed tests have not been accessed.

## 1. Equal 2-seed head-to-head on original DAVIS 302/59/81 split: COMPLETE

- Both use EXACTLY the same original 81-protein test set, all 68 compounds warm, 5,508 pairs.
- DynaCPI v5 two-seed frozen checkpoints 2931 + 2932, as originally selected on 59 validation proteins.
- PerceiverCPI original UNMODIFIED source at SHA 755f076ef51099a90332fd1ec8437f94e3750c0b, two separate 30-epoch training seeds 2931 and 2932, best checkpoint per seed selected on the same 59 validation proteins, equal-weight 2-seed prediction ensemble, **no test-based selection**.
- Original PerceiverCPI seed 2931 held-out macro RMSE 0.889304648, pooled RMSE 0.935549161. Seed 2932 held-out macro 0.986225169, pooled 1.076548831; second seed worsened original model on the pre-exposed protein heldout.
- **Two-seed comparison:** DynaCPI v5 macro RMSE **0.585203057**, pooled RMSE **0.656132340**. Official PerceiverCPI 2-seed ensemble macro RMSE **0.890663077**, pooled RMSE **0.950052280**. DynaCPI lowers target-macro RMSE **34.296%**, pooled RMSE **30.937%**.
- This removes the prior 2-vs-1 seed number asymmetry and confirms DynaCPI is ahead on this PARTICULAR original split. The test is ALREADY EXPOSED from the previous run and CANNOT count as independent replication.
- Training-run "test MSE" at 0.556168 and 0.563030 are the official source's REQUIRED **validation proxy**, NOT the 81-protein independent test. Correct official held-out ensemble MSE is pooled RMSE squared.
- Provenance: ignored local matched_perceivercpi_20261110/parity_2seed_heldout/PARITY_METRICS.json; SHA verified against the frozen official seed2931 checkpoint and previously generated 5,508 original predictions. Public evaluator: evaluate_perceiver_2seed_parity_20261110.py. Source for original 1-seed heldout: MATCHED_PERCEIVERCPI_FINAL_HELDOUT_2026-11-10.md.

## 2. NEW locked DAVIS validations; no sealed test opened

Fixed data splits and metric priorities are in GENERALIZATION_PREREGISTRATION_2026-11-10.md, with frozen local SHA256 manifests and no test-label CSVs exported. The low-rank architecture and training seeds are frozen independently of new test labels.

Scenario | DynaCPI 2-seed VALIDATION macro RMSE | Pooled RMSE | Training-only chemical prior macro RMSE | Validation pairs
- Strict protein split A (20261113): 0.693429727 | 0.744957173 | not separately scored | 5,236 (77 validation proteins x 68 warm compounds)
- Strict protein split B (20261114): 0.648593048 | 0.734122859 | not separately scored | 4,624 (68 proteins x 68 warm compounds)
- Cold compound scaffold / chemical identity (20261115): 0.629825048 | 0.728496810 | 0.745633628 | 4,420 (442 known proteins x 10 validation compounds)
- Double cold compound + protein (20261116): 0.704524644 | 0.788427148 | 0.715481167 | 550 (55 validation proteins x 10 validation compounds)

The first strict protein split DynaCPI validation-selected seed checkpoints were epochs 16, 11. Second strict protein split checkpoints selected solely on validation: first seed 23, second seed 35. Cold-chemistry checkpoint epochs: compound 14, 7; double-cold 1, 1. The latter selected extremely early epochs and barely improved on the chemical prior, which is a SIGNIFICANT RESEARCH LIMITATION, not a model success to hide.

For all cold-chemistry experiments, DynaCPI chemical anchor values for unknown compounds come **only from training compounds' TRAINING-protein mean pKd**, extrapolated with a fixed ridge-0.1 Morgan-Tanimoto kernel. No validation/test chemical labels are used to construct the training prior. Two checkpoint weights were frozen on VALIDATION for both cold scenarios, no sealed tests accessed. A NumPy int32 metadata compatibility issue during the first cold-compound finalization was fixed with explicit safe torch weights-only loading; the original two checkpoint weights were preserved rather than retrained, and their SHA256s verified.

**Next required comparison:** original PerceiverCPI 2-seed training on each new split, exactly the same train/validation rows. Running on strict split A: seed 2931 has reached epoch 21 and seed 2932 started 19:56 IST, 10 Nov. No strict-split test score exists. Generic official run script: run_preregistered_official_perceiver_generic_20261110.ps1; originally launched first seed with run_preregistered_official_perceiver_strict_seed2931.ps1. New guarded test evaluators will refuse test access until both original official model runs complete and frozen DynaCPI validation replays match.

## 3. Blind, label-free chemical similarity audit

The fixed Morgan radius-2 2,048-bit fingerprint test-to-training maximum Tanimoto statistics were computed WITHOUT consulting test labels or modifying splits. Public audit code: audit_preregistered_davis_chemical_novelty_20261110.py; full local ignored CHEM_NOVELTY_DIAGNOSTIC.json.

- Cold compounds: 45 training ligands, 10 validation, 13 held-out TEST. Median maximum Tanimoto to any training ligand is **0.3010** for validation, **0.2887** for test; maximum test nearest similarity **0.5974**. Two of thirteen test compounds have nearest Tanimoto ≥0.5.
- Double cold: 45 training ligands, 10 validation, 13 held-out TEST. Median maximum similarity is **0.3104** validation, **0.2340** test; maximum test nearest similarity **0.6974**, two of thirteen ≥0.5.
- All 68 DAVIS compounds have distinct exact Murcko scaffold identifiers. Here "scaffold cold" implies chemical-identity disjointness and exact Murcko-scaffold disjointness, NOT uniformly low pharmacophore/fragment similarity. Even the strict protein 3-mer threshold 0.35 does not guarantee curated kinase family separation.

## 4. Previously exposed split floor sensitivity (post hoc)

On old 81-protein test, **70.225%** of pairs have pKd exactly 5. DynaCPI remains better if the official Perceiver prediction is also clamped at 5 (pooled RMSE 0.928311 versus DynaCPI 0.656132). On non-floor pairs (1,640), DynaCPI RMSE 1.134500 versus official 1.441167. This is meaningful diagnostic evidence on an OLD exposed test, NOT a fresh confirmatory test.

## 5. Predeclared failure criteria and follow-up

Success on each NEW frozen split requires two-seed DynaCPI lower target-macro AND lower pooled RMSE than the two-seed official Perceiver ensemble on identical sealed test pairs. Secondary non-floor RMSE is reported even when adverse. Every negative result is retained. Confidence intervals are descriptive unless a prespecified independent family-level resampling protocol is supported. Repeat beyond DAVIS across original endpoint definitions (KIBA composite score, METZ if verified); no claims beyond tested protocols.

**Model-provenance caveats:** DynaCPI v5's frozen 650M-parameter ESM-2 protein encoder pretraining differs from PerceiverCPI's architecture. The 2-seed-versus-2-seed comparison matches seed count and data split, NOT pretrained representation quality, model complexity, training compute, hyperparameter optimization or publication's original fivefold procedure. The completed results concern compact LowRankDynaCPI v5, not unimplemented GVP geometry/support routing; the separate full-refit v6 score of 0.298741 is TRAIN-FIT only.

## Code and private artifacts

Public repository: https://github.com/j07-styx/dynacpi-better-version. Private raw DAVIS data, frozen training/validation CSVs, hidden test indices and labels, official checkpoint weights, inference rows and metric JSON are ignored locally. New scripts:
- train_preregistered_dynacpi_v5_protein_20261110.py (frozen strict A/B)
- train_preregistered_dynacpi_cold_chem_20261110.py (fixed training-only chemical prior for compound/double)
- finalize_preregistered_dynacpi_cold_chem_20261110.py (weights-only safe recovery, no test)
- audit_preregistered_davis_chemical_novelty_20261110.py (no label access)
- run_preregistered_official_perceiver_generic_20261110.ps1
- evaluate_preregistered_strict_protein_parity_20261110.py (guarded)
- evaluate_preregistered_cold_chem_parity_20261110.py (guarded)
