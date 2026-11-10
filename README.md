# DynaCPI better version — consolidated research repository

**START WITH [HANDOFF_COMPLETE_2026-11-09.md](HANDOFF_COMPLETE_2026-11-09.md).** It summarizes all completed experiments (including negative results), the former v4 code snapshot, datasets and split boundaries, checkpoint locations, reproducibility and the handoff instructions to another AI.

**Current status (2026-11-09):** v4.3 anchored sequence+chemical-prior backend runs on localhost:8781. The previous **v4.6 exploratory nested stacking** reached **0.607092** double-cold target-macro RMSE versus chemical-only **0.608048** on the previously inspected historical folds; its paired confidence interval includes zero, so this is not a validated win. The earlier v4.4 protein-conditioned kernel alone scored 0.616156. No verified improvement over original published PerceiverCPI. **The old PerceiverCPI/DynaCPI checkout remains separate and scientifically unapproved BindingDB cohorts must stay blocked.**

This is a **public** repository requested as `dynacpi better version`. It contains legacy v4 and v4.2/v4.3 source files, public-safe experimental summaries, technical handoffs and tests. Model checkpoint binaries, fitted kernel coefficients, protected source records, and raw datasets remain only on the original computer.

## Latest unseen-ligand inference check (10 Nov 2026)

[12 true v6 predictions on six unseen compounds versus ABL1 and EGFR](UNSEEN_COMPOUND_DEMO_V6_2026-11-10.md): all six ligands confirmed absent from the 68 original DAVIS training ligands; direct frontend-proxy checkpoint-backed pKd predictions succeeded; maximum Tanimoto to any training ligand ranged **0.086–0.250**, an out-of-distribution warning. These values are **not experimentally validated** and there is no unseen-compound RMSE or matched PerceiverCPI comparison. Reproduction script: [analyze_unseen_compounds_v6_20261110.py](analyze_unseen_compounds_v6_20261110.py).

## Latest: v6 final all-DAVIS trained model and real React frontend (10 Nov 2026)

Full report: [FULL_DAVIS_V6_FINAL_APP_2026-11-10.md](FULL_DAVIS_V6_FINAL_APP_2026-11-10.md). Original Midnight & Violet Lovable-style React frontend: [frontend_v6/](frontend_v6/) on local port **5173**, connected to real **v6 full-data inference** on port **8783**. Accepts user SMILES + protein sequences, not only fixed benchmark IDs. Original v5 held-out API on port **8782** remains unchanged. v6 refit uses **all 442 DAVIS protein reference entries × 68 compounds = 30,056 pairs**, 2 seeds, 29 and 24 preselected epochs. Verified full-data ensemble **TRAIN-FIT** macro RMSE **0.298741** vs chemical-only **0.737121**, which is **IN-SAMPLE and NOT a generalization score**. Prior **v5** approximately protein-grouped **held-out** result **0.585203 vs 0.823596** is from **DIFFERENT earlier checkpoints**. All 13 frontend tests, TypeScript and production build pass. No matched original PerceiverCPI comparator. Local source data and weights stay uncommitted.

## Full DAVIS v5 — TRAINED, SERVING, HELD-OUT RESULTS (9 Nov 2026)

**Demo (on the authorized Windows PC): http://127.0.0.1:8782/**. See [FULL_DAVIS_V5_DEMO_REPORT_2026-11-09.md](FULL_DAVIS_V5_DEMO_REPORT_2026-11-09.md) for the full methodology, exact training data and source hashes, architecture, training curves, validation selection, held-out evaluation and demo endpoints. A small **new LowRankDynaCPI v5** uses train-only-centered ESM, a 32D protein/compound bottleneck and censored-aware training. On the full public DeepDTA DAVIS benchmark **442 protein reference entries × 68 warm compounds = 30,056 affinity pairs**, trained on 302 entries, selected on 59 validation entries, evaluated on 81 held-out protein entries: the validation-selected **two-seed** ensemble scored **0.585203 target-macro RMSE vs 0.823596 train-compound-mean prior**, a **28.95% reduction** on this protein-grouped, warm-compound split. Best checkpoints were from epochs 29 and 24; separate older v4.3 server remains online port 8781.

**Strict limitations:** This is **NOT** PerceiverCPI and PerceiverCPI has **NOT** been run on this split. The protein grouping is a 3-mer similarity **proxy**, not curated family disjointness. The heuristic DAVIS pKd floor handling is not assay-level censor annotation. This cannot establish a published PerceiverCPI win, real-world drug affinity performance, or double-cold generalization. Code and public-safe summaries only are in Git; original data, ESM caches and checkpoint weights stay local.

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
