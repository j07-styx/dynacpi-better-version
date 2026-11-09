# DynaCPI better version — consolidated research repository

**START WITH [HANDOFF_COMPLETE_2026-11-09.md](HANDOFF_COMPLETE_2026-11-09.md).** It summarizes all completed experiments (including negative results), the former v4 code snapshot, datasets and split boundaries, checkpoint locations, reproducibility and the handoff instructions to another AI.

**Current status (2026-11-09):** v4.3 anchored sequence+chemical-prior backend runs on localhost:8781, while v4.4 protein-conditioned kernel correction did **not** beat chemical-only double-cold. No verified improvement over original published PerceiverCPI. **The old PerceiverCPI/DynaCPI checkout remains separate and scientifically unapproved BindingDB cohorts must stay blocked.**

This is a **public** repository requested as `dynacpi better version`. It contains legacy v4 and v4.2/v4.3 source files, public-safe experimental summaries, technical handoffs and tests. Model checkpoint binaries, fitted kernel coefficients, protected source records, and raw datasets remain only on the original computer.

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
