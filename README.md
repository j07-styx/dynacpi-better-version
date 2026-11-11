# DynaCPI better version — consolidated research repository

**LATEST VERIFIED RESULTS — 11 NOVEMBER 2026, 07:08 IST:** DynaCPI v5 beat the original two-seed PerceiverCPI on the independently held-out KIBA and Metz **protein-cold** splits, both macro and pooled RMSE. The KIBA paired cluster-bootstrap 95% intervals favor DynaCPI on both metrics; Metz point estimates favor DynaCPI but both intervals include zero. See [KIBA and Metz test results, uncertainty, provenance and limitations](KIBA_METZ_MATCHED_PROTEIN_COLD_TEST_RESULTS_2026-11-11.md) and [incremental one-time testing protocol amendment](EXTERNAL_INCREMENTAL_ONCE_TEST_PROTOCOL_2026-11-11.md). **New Metz compound-cold held-out result**: DynaCPI macro RMSE **0.748509** vs original **0.764061**, pooled **0.781367** vs original **0.804360** on 6,117 unseen-compound pairs; descriptive paired intervals include zero. See [Metz unseen-compound once-test report](METZ_COMPOUND_COLD_MATCHED_TEST_2026-11-11.md). KIBA compound-cold and both double-cold external tests remain pending. The four new DAVIS tests are complete: DynaCPI has both-metric wins on strict-protein A/B and cold compounds, with mixed double-cold results. **Older dated status sections below are historical snapshots, not the current benchmark status.**

**NEW 11 NOV RESEARCH AND SAFE SCHEDULING:** [Post-hoc KIBA/Metz double-cold *validation-only* novelty analysis](KIBA_METZ_DOUBLE_COLD_VAL_NOVELTY_DIAGNOSTIC_2026-11-11.md) replays all frozen checkpoints without accessing double-cold held-out outcomes. KIBA's train-only chemical prior slightly beats DynaCPI on pooled double-cold VAL, particularly for less chemically similar unseen ligands; Metz shows a modest positive DynaCPI pooled improvement. The [source code](audit_external_double_cold_novelty_validation_20261111.py) makes no training adjustments. The [optional intermediate once-test gate](run_incremental_external_ready_gate_20261111.ps1) only processes **KIBA compound-cold and Metz double-cold** once both original seed checkpoints are ready, leaving final KIBA double-cold to the existing all-six gate to prevent a race. Every per-scenario run retains its unique private test output; no test is repeated or used for retuning. This local Windows watchdog cannot send ChatGPT notifications and stops if the computer sleeps or its underlying process fails.

**START WITH [HANDOFF_COMPLETE_2026-11-09.md](HANDOFF_COMPLETE_2026-11-09.md).** It summarizes all completed experiments (including negative results), the former v4 code snapshot, datasets and split boundaries, checkpoint locations, reproducibility and the handoff instructions to another AI.

**Current status (2026-11-09):** v4.3 anchored sequence+chemical-prior backend runs on localhost:8781. The previous **v4.6 exploratory nested stacking** reached **0.607092** double-cold target-macro RMSE versus chemical-only **0.608048** on the previously inspected historical folds; its paired confidence interval includes zero, so this is not a validated win. The earlier v4.4 protein-conditioned kernel alone scored 0.616156. No verified improvement over original published PerceiverCPI. **The old PerceiverCPI/DynaCPI checkout remains separate and scientifically unapproved BindingDB cohorts must stay blocked.**

This is a **public** repository requested as `dynacpi better version`. It contains legacy v4 and v4.2/v4.3 source files, public-safe experimental summaries, technical handoffs and tests. Model checkpoint binaries, fitted kernel coefficients, protected source records, and raw datasets remain only on the original computer.

## Two-tower signal diagnostic: protein and chemical features both matter on validation (10 Nov 2026)

[Six-way KIBA/Metz two-tower feature reliance audit](EXTERNAL_TWO_TOWER_FEATURE_RELIANCE_2026-11-10.md). Replaying the SAME frozen two-seed models on their original VAL samples, five fixed protein-feature shuffles and five fixed chemical-tower Morgan feature shuffles per scenario showed positive mean target-macro RMSE degradation on **ALL six** KIBA/Metz splits for BOTH representation branches. Protein shuffle macro penalty ranged **+0.0190 to +0.2386**, chemical-head fingerprint shuffle **+0.0233 to +0.1339**. The chemical-prior anchor, opposite-modality features, true labels and sample IDs were preserved. Much weaker DOUBLE-COLD penalties and KIBA double-cold underperformance versus the prior on pooled validation warn against overclaiming biology. Both shuffles are **VALIDATION-ONLY OOD diagnostic controls**, neither new TEST data nor proof of model superiority. Public code: [protein reliance diagnostic](audit_external_protein_feature_shuffle_val_20261110.py), [molecular residual reliance diagnostic](audit_external_chemical_feature_shuffle_val_20261110.py).

## External KIBA and Metz generalization: six validations done, original baseline training active (10 Nov 2026)

[External benchmark source provenance, six frozen unseen-group splits, all validation scores and preserved negative results](KIBA_METZ_EXTERNAL_VALIDATION_STATUS_2026-11-10.md). Both public source datasets are downloaded locally and SHA-pinned: raw KIBA **229 proteins / 2,111 compounds / 118,254 observed scores** (KIBA composite score, NOT DAVIS pKd), and filtered Metz **170 proteins / 1,423 compounds / 35,259 numeric affinity observations**. All **399 protein sequences already had exact cached ESM-2 650M embeddings**. [Source audit/split builder](prepare_external_kiba_metz_frozen_benchmarks_20261110.py) has locked three outcome-blind splits per dataset (cold protein, cold compound, double cold), without exporting any sealed test labels. [Fixed sparse DynaCPI trainer](train_external_sparse_dynacpi_v5_20261110.py) completed TWO seeds on ALL SIX scenarios, selecting checkpoints on validation only. DynaCPI improves over its chemical-only anchor on most validation metrics, but its **KIBA double-cold pooled validation RMSE 0.758663 is slightly WORSE than the anchor 0.755423**. No external PerceiverCPI superiority claim is justified yet.

The official unchanged original PerceiverCPI baseline is now training on Metz protein-cold seed2931 through [frozen original-model runner](run_external_original_perceiver_frozen_20261110.ps1). The [other 11 official baseline seeds/scenarios](run_remaining_external_perceiver_queue_20261110.ps1) are queued SEQUENTIALLY behind it with a 2.80GB free-system-RAM check, alongside the existing DAVIS study. The laptop's **16GB RAM is the principal concurrency bottleneck**, despite spare 8GB GPU VRAM. All six external sealed tests remain unopened. The [external all-six once-only comparison evaluator](evaluate_external_kiba_metz_all_six_once_20261110.py) has **reconstructed and verified all six DynaCPI frozen validation metrics**, and the [external test gate](run_external_sealed_test_gate_20261110.ps1) is launched on the authorized Windows machine. It waits until all 12 official PerceiverCPI seed/scenario checkpoints are complete and source hashes match, replays frozen validation, and only then opens and scores the six independently sealed tests exactly once; failures stop without cherry-picking or retraining. No ChatGPT background alerts or immediate results are promised. These are different KIBA/Metz data variants and splits from the published paper: do not numerically compare unaligned metrics.

## All four DAVIS held-out tests guarded by checkpoint replay audit (10 Nov 2026)

[Validation-only four-scenario checkpoint replay](audit_davis_four_frozen_validation_replay_20261110.py) successfully reproduced the previously SHA-locked two-seed validation macro/pooled RMSEs for strict-protein A, strict-protein B, unseen-compound and double-cold partitions **without reading any test labels**. The [four-test once-only evaluator](evaluate_all_four_preregistered_davis_once_20261110.py) now mandates this whole-suite replay BEFORE opening ANY sealed test. Both original official PerceiverCPI seeds on strict A are complete; the additional six seed/split official models are running in the existing fail-closed queue. No new sealed DAVIS test score yet.

## Latest: 2-seed head-to-head confirmed, 4 cold/strict validations completed (10 Nov 2026)

[Full protocol, score ledger and negative observations](GENERALIZATION_ROUND1_PROGRESS_2026-11-10.md). On our original 81-protein / 5,508-pair **HELD-OUT** DAVIS split, the validation-frozen **two-seed versus two-seed** comparison has now completed: **LowRankDynaCPI v5 macro RMSE 0.585203, pooled RMSE 0.656132** versus **original PerceiverCPI macro RMSE 0.890663, pooled RMSE 0.950052** (34.3% and 30.9% lower). This eliminates the initial 2-versus-1 seed-count imbalance on the SAME ALREADY-EXPOSED split but does NOT constitute a new independent family-level test or show identical encoder pretraining budget.

**New, independently frozen, TEST-SEALED validations completed:** strict protein A macro RMSE **0.693430**, strict protein B **0.648593**; cold compound **0.629825** versus training-only chemical-prior **0.745634**; double-cold **0.704525** versus prior **0.715481** (very small gain). These are **VALIDATION scores**, NOT matched PerceiverCPI tests. All four DynaCPI two-seed checkpoint sets are saved locally. Two original PerceiverCPI seeds are currently training on strict split A, and a fail-closed sequential OS training queue is launched to run six remaining official seed/scenario combinations once strict A completes. All four new test partitions stay sealed and the original code, source features, checkpoints and protected raw outputs remain uncommitted.

[Original seed-parity evaluator](evaluate_perceiver_2seed_parity_20261110.py), [strict protein guarded evaluator](evaluate_preregistered_strict_protein_parity_20261110.py), [cold-compound/double-cold guarded evaluator](evaluate_preregistered_cold_chem_parity_20261110.py), [generic official training runner](run_preregistered_official_perceiver_generic_20261110.ps1), and [local sequential queue](run_remaining_preregistered_perceiver_queue_20261110.ps1). The blind evaluators refuse test access unless both official seeds have finished and model/data hashes plus frozen DynaCPI validation replay match. The [four-scenario all-model preflight and single-use evaluator](evaluate_all_four_preregistered_davis_once_20261110.py) is now syntax-validated and confirms all four DynaCPI validation selections are frozen; the new test sets remain sealed. The [local evaluation gate](run_sealed_benchmark_evaluation_gate_20261110.ps1) is running on the authorized PC, waiting for the existing official training queue to complete before launching the one-time test suite; a failed training/checkpoint gate stops without accessing test labels. The [paired protein/chemical-group bootstrap](paired_bootstrap_preregistered_davis_20261110.py) passed synthetic-data tests and provides descriptive, not definitive, uncertainty intervals. **No new blinded test outcomes have been observed yet.** Neither process guarantees future ChatGPT notifications, and the PC must remain running.

## Stronger validation underway: split robustness, floor sensitivity, seed parity (10 Nov 2026)

[Preregistered robustness protocol](GENERALIZATION_PREREGISTRATION_2026-11-10.md) and [post-hoc paired floor audit code](audit_matched_test_floor_bias_20261110.py). Our original matched-split win remains after giving PerceiverCPI the same pKd>=5 output floor (Perceiver pooled RMSE **0.928311** vs DynaCPI **0.656132**). On **1,640 non-floor affinity pairs** the original official model scored RMSE **1.441167**, DynaCPI **1.134500**; a descriptive 57-sequence-group bootstrap for the old exposed split also favors DynaCPI, but this is **NOT a new independent test**. An additional official Perceiver seed (2932) has **finished 30 epochs** on the original 302/59 split. Its validation-selected checkpoint is independently scored through the [2-seed parity evaluator](evaluate_perceiver_2seed_parity_20261110.py). This is the same already-exposed original split; full results appear in the latest section above.

Four outcome-blind new DAVIS splits were frozen by [generalization split builder](prepare_preregistered_davis_generalization_20261110.py): two stricter protein-similarity-group splits, one Murcko/compound-identity cold and one double cold. The [fixed-architecture DynaCPI trainer](train_preregistered_dynacpi_v5_protein_20261110.py) has FINISHED two seeds on **both stricter protein partitions**, selecting solely on validation. The [cold-chemistry trainer](train_preregistered_dynacpi_cold_chem_20261110.py) has also completed both compound-cold and double-cold validations with training-only chemical priors. Matching official PerceiverCPI two-seed runs are still required on every new split; **no new split's sealed test has been evaluated**. All raw data, test indices, validation predictions and trained checkpoints stay ignored and local.

**No universal superiority claim:** our original 0.9 cosine protein groups are not curated families; 70% DAVIS floor labels and ESM-2 pretraining are confounds; chemistry-specific extrapolation and KIBA/METZ validations remain future steps. Report negative results as well as positive ones.

## Official PerceiverCPI matched-DAVIS comparison — independent held-out result (10 Nov 2026)

The [original dmis-lab/PerceiverCPI](https://github.com/dmis-lab/PerceiverCPI) `InteractionModel` (unchanged upstream source at 755f076) was trained for **all 30 epochs** on our exact v5 frozen DAVIS 302/59/81 protein-grouped split; validation-selected best model was epoch 20 (validation pooled MSE 0.556168, but that is NOT the true test). Its once-only independent **81-protein / 5,508-pair matched held-out TEST** produced target-macro RMSE **0.889305**, pooled RMSE **0.935549**. Our earlier validation-selected two-seed DynaCPI v5 ensemble had target-macro RMSE **0.585203** and pooled RMSE **0.656132** on **exactly the same test pairs**: **34.2% lower target-macro RMSE** and **29.9% lower pooled RMSE** in this single-split, 2-seed-vs-1-seed comparison. [Complete official baseline report, hashes and limitations](MATCHED_PERCEIVERCPI_FINAL_HELDOUT_2026-11-10.md); [interim validation analysis](MATCHED_PERCEIVERCPI_STATUS_2026-11-10.md); [one-time guarded test evaluator](evaluate_matched_perceivercpi_20261110.py). This is a legitimate **matched-split win against one official PerceiverCPI seed**, not a multiple-seed, multi-split or published fivefold benchmark superiority claim. Raw input CSVs, test predictions, original clone and checkpoints remain LOCAL ONLY.

## Latest unseen-ligand inference check (10 Nov 2026)

[12 true v6 predictions on six unseen compounds versus ABL1 and EGFR](UNSEEN_COMPOUND_DEMO_V6_2026-11-10.md): all six ligands confirmed absent from the 68 original DAVIS training ligands; direct frontend-proxy checkpoint-backed pKd predictions succeeded; maximum Tanimoto to any training ligand ranged **0.086–0.250**, an out-of-distribution warning. These values are **not experimentally validated** and there is no unseen-compound RMSE or matched PerceiverCPI comparison. Reproduction script: [analyze_unseen_compounds_v6_20261110.py](analyze_unseen_compounds_v6_20261110.py).

## Latest: v6 final all-DAVIS trained model and real React frontend (10 Nov 2026)

Full report: [FULL_DAVIS_V6_FINAL_APP_2026-11-10.md](FULL_DAVIS_V6_FINAL_APP_2026-11-10.md). Original Midnight & Violet Lovable-style React frontend: [frontend_v6/](frontend_v6/) on local port **5173**, connected to real **v6 full-data inference** on port **8783**. Accepts user SMILES + protein sequences, not only fixed benchmark IDs. Original v5 held-out API on port **8782** remains unchanged. v6 refit uses **all 442 DAVIS protein reference entries × 68 compounds = 30,056 pairs**, 2 seeds, 29 and 24 preselected epochs. Verified full-data ensemble **TRAIN-FIT** macro RMSE **0.298741** vs chemical-only **0.737121**, which is **IN-SAMPLE and NOT a generalization score**. Prior **v5** approximately protein-grouped **held-out** result **0.585203 vs 0.823596** is from **DIFFERENT earlier checkpoints**. All 13 frontend tests, TypeScript and production build pass. A separately trained matched original PerceiverCPI comparator for **v5** has completed (see above); **v6** remains a full-data deployment refit without independent test. Local source data and weights stay uncommitted.

## Full DAVIS v5 — TRAINED, SERVING, HELD-OUT RESULTS (9 Nov 2026)

**Demo (on the authorized Windows PC): http://127.0.0.1:8782/**. See [FULL_DAVIS_V5_DEMO_REPORT_2026-11-09.md](FULL_DAVIS_V5_DEMO_REPORT_2026-11-09.md) for the full methodology, exact training data and source hashes, architecture, training curves, validation selection, held-out evaluation and demo endpoints. A small **new LowRankDynaCPI v5** uses train-only-centered ESM, a 32D protein/compound bottleneck and censored-aware training. On the full public DeepDTA DAVIS benchmark **442 protein reference entries × 68 warm compounds = 30,056 affinity pairs**, trained on 302 entries, selected on 59 validation entries, evaluated on 81 held-out protein entries: the validation-selected **two-seed** ensemble scored **0.585203 target-macro RMSE vs 0.823596 train-compound-mean prior**, a **28.95% reduction** on this protein-grouped, warm-compound split. Best checkpoints were from epochs 29 and 24; separate older v4.3 server remains online port 8781.

**Strict limitations:** This is **NOT** PerceiverCPI; although a **single original-model matched-split TEST comparison was completed on 10 November 2026 and favors v5**, two PerceiverCPI seeds and one approximate similarity-group split have been evaluated in the original matched test, so no generalized published-paper superiority claim is warranted. The protein grouping is a 3-mer similarity **proxy**, not curated family disjointness. The heuristic DAVIS pKd floor handling is not assay-level censor annotation. This cannot establish a published PerceiverCPI win, real-world drug affinity performance, or double-cold generalization. Code and public-safe summaries only are in Git; original data, ESM caches and checkpoint weights stay local.

## Latest read-only diagnostic (v4.8, 9 Nov 2026)

[Zero-residual training-loss and protein-neighbor diagnostic](DIAGNOSTIC_ZERO_RESIDUAL_PROTEIN_LOO_2026-11-09.md) reconstructed all three original chemical-prior fold scores exactly. Anchored model's mean standardized TRAIN MSE is **0.74320 after five epochs vs 0.79001 with zero neural residual** (5.9% lower); this suggests limited residual fit but does **not** establish undertraining. In a 24-protein leave-one-protein-out selectivity diagnostic with all 54 compounds warm, centered MSE is **0.40770** chemical mean, **0.32134** label-informed best single protein, and **0.63087** nearest mean-pooled ESM cosine. This is TRAIN-set analysis, not independent science or a paper benchmark. The raw ESM nearest-neighbor method is worse than the chemical mean and motivates investigating better protein similarity on a NEW approved evaluation.

## Latest tested experiment

**v4.7 (9 Nov 2026):** [Scaffold-excluded out-of-fold chemical residual study](EXPERIMENT_OOF_RESIDUAL_V47_2026-11-09.md). The chemical prior was more optimistic on its seen training compounds than on held-out chemical scaffold groups. Out-of-fold residual training with four inner folds improved the **cold-compound** macro RMSE (0.696806 vs 0.708662 chemical-only, and 0.700078 matched in-sample neural), but **worsened double-cold** RMSE (0.615673 vs 0.608048 chemical-only and 0.607088 matched in-sample). This is NOT a general model improvement or a paper benchmark win. Deterministic torch/CUBLAS settings enforced. Neither backend nor original checkpoints changed.

**v4.6 (earlier):**

Read [EXPERIMENT_ENSEMBLE_STACK_SHRINK_2026-11-09.md](EXPERIMENT_ENSEMBLE_STACK_SHRINK_2026-11-09.md) for the three-seed prediction ensemble, nested chemical/stacked residual blend, optional chemical/protein gates, all six comparator metrics, per-fold results, descriptive paired bootstrap, numerical replay differences, and reproduction instructions. All experiment code and public-safe summaries are present; pair-level prediction caches and model weights remain local.

## Historical v4.2 notes

**9 November 2026.** Separate backend/training workspace. This is **not** the `Demo-smth` repository and does not modify its tracked source, original checkpoints, frontend or model defaults.

## What works

- **Trained from scratch:** two-path sequence-only candidate (fingerprint residual + direct compound/protein interaction MLP), across seeds 2831 / 2832 / 2833, five epochs each on 24 DAVIS engineering training proteins (1,296 pairs).
- **Fairer matched control:** original DynaCPI SequenceFoundation trained from fresh matched seed, same 24 proteins/1,296 pairs and same five-epoch budget.
- **Metrics:** repaired 0.75141 vs original 0.80942 target-macro RMSE averaged over seeds, on eight **previously inspected** DAVIS development proteins. Important: training-compound average 0.70166 beats both; no independent generalization is established.
- **Separate local inference API:** `http://127.0.0.1:8781`. No frontend code in this workspace.
- **Frozen serving:** choose `repaired_sequence_2831`, `repaired_sequence_2832`, or `repaired_sequence_2833` by explicit `model_id`; legacy `sequence_baseline` remains default. The geometry-only variants from the old v4 backend remain available as historical engineering choices, not improvements.
- **Provenance:** model source, dataset and fingerprint hashes checked against checkpoint before loading. All source and training endpoints are local; no public deployment.

## Requirements / current data dependency

The training code and inference service intentionally **reuse read-only source data, cached ESM-2 residue embeddings, Morgan fingerprints, and legacy model modules** from:

`C:\Users\J0124\dynacpi_v4`

They also reuse its Python virtual environment and shared GPU lease. This repository **is not runnable on a clean machine without migrating those dependencies and obtaining the same licensed datasets**. No raw BindingDB data, private source-review artifacts, protected test labels or large ESM caches are copied here. Local checkpoints live under `checkpoints/` and are intentionally not pushed to Git.

The model requires CUDA because the original compound MPN implementation uses CUDA internally. Train and inference share the existing `.gpu.lock` lease to avoid interrupting other GPU activity.

## Run the backend

PowerShell:

```powershell
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' 'C:\Users\J0124\DynaCPI-Backend\app.py'
```

Endpoint summary:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Backend availability and version |
| GET | `/api/v1/models` | Saved checkpoints, provenance, baseline/default status |
| GET | `/api/v1/targets` | Four mapped DAVIS development targets: ADCK4, FRK, MEK5, OSR1 |
| GET | `/api/v1/training/repaired` | Repaired model training run summary |
| GET | `/api/v1/training` | Historical geometry experiment results |
| POST | `/api/v1/predict` | Frozen sequence or geometry inference, optional support adaptation |

POST example, for a **model estimate** not an experimentally measured affinity:

```json
{
  "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
  "target": "FRK",
  "support_k": 0,
  "model_id": "repaired_sequence_2833"
}
```

Expected response fields: `model_id`, `protein`, `canonical_smiles`, `support_k`, `geometry_available`, `predictions` containing `sequence_pkd`, `geometry_pkd` and A/B/C/D support-calibrated variations, `scope`, and `warnings`. The ordinary candidate is **sequence-only**: `geometry_available=false`. Experimental 3D checkpoint models remain distinct models; do not imply that the repaired candidate has geometry.

The service permits localhost origins at ports 3000/4173/5173/8080. Use `DYNACPI_ALLOWED_ORIGINS` to configure another origin. It binds only `127.0.0.1`. Do not expose this research Flask server to the internet.

## Reproduce training

To avoid accidentally overwriting finished results, **training scripts refuse to overwrite checkpoints**. Use a new experiment directory for a repeat or intentionally archive/move existing artifacts first. Previous results should remain immutable.

```powershell
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' train_repaired_full.py
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' train_original_matched.py
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' audit_simple_baselines.py
```

- `train_repaired_full.py` fits three 5-epoch two-bypass models, computes frozen-label diagnostics after saving, writes `checkpoints/SUMMARY.json` and each seed's local `checkpoint.pt`.
- `train_original_matched.py` fits matched original architectures, writes `checkpoints/MATCHED_COMPARISON.json`.
- `audit_simple_baselines.py` computes global and train-compound-average priors on exact matching historical DEV panels.
- `causal_diagnosis_20261109/` contains original supplied causal probes, the 18/6 internal split comparison and its record.
- `repaired_model.py` implements the same two bypasses as the original supplied experiment. `test_model_parity.py` confirms equality under identical loaded state tensors.

Tests:

```powershell
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' test_model_parity.py
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' test_api.py
```

## Research limitations

This does **not** establish superiority to PerceiverCPI or generalization to unseen compounds. Original and repaired models have different parameter counts. The eight-protein DEV was previously inspected, all 54 compounds are shared between training and DEV, and primary assay construct/source review remains incomplete. Floor-bound DAVIS pKd should not be conflated with uncensored experimental measurements. The original graph/geometry and original checkpoints are not overwritten. BindingDB candidate tasks and reserved tests are not used in any new fit.

**Do not promote repaired model as the production default.** A scientifically independent source-family/chemistry-disjoint cohort, a compound-generalization comparator, and a new geometry ablation must precede such a claim.
