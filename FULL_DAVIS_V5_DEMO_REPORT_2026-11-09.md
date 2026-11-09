# DynaCPI v5 full DAVIS — trained working demonstration

**Date:** 9 November 2026
**Status:** FULL-DAVIS TRAINING COMPLETED, VALIDATION SELECTION FROZEN, TEST EVALUATED, CHECKPOINTS RECOVERED, LOCAL FLASK DEMO RUNNING
**Demo URL on connected Windows PC:** http://127.0.0.1:8782
**Previous v4.3 backend unchanged:** http://127.0.0.1:8781

## 1. Achievement and research boundaries

We trained a **new small two-tower LowRankDynaCPI model** on the **full public DeepDTA DAVIS benchmark**, with 442 protein reference entries, 68 ligand reference entries and 30,056 nominal affinity observations. Frozen residue-level ESM-2 protein caches from the earlier machine were mean-pooled and converted into 1280-dimensional per-protein vectors; 2048-bit Morgan radius-2 chiral fingerprints were regenerated from public canonical ligand SMILES. The new network is NOT the earlier DynaCPI v4.3/4.7 sequence-attention encoder and NOT a reimplementation of original PerceiverCPI.

All runs are local research demos, not clinical decision tools or independently validated ligand ranking algorithms. Do not infer model superiority against published PerceiverCPI: that architecture has **NOT** yet been trained and evaluated on the identical new split.

The model was trained on ALL 68 compounds, but distinct protein entries. It has NOT been validated on unseen compounds or a double-cold split.

## 2. Dataset provenance

Source: the canonical `hkmztrk/DeepDTA` public benchmark `data/davis` files (`Y`, `proteins.txt`, `ligands_can.txt`). The downloaded protein and ligand files are byte-identical to the existing `dynacpi_v4/data/references` copies. The raw `Y` SHA256 is:

`9855c4f234ec6dd18295b0e0cd6ee54214cafa908c6eb1d447b47d454fc0294a`

The binary `Y` matrix is 68 × 442 and was deserialized via a restricted NumPy-only unpickler. pKd labels were calculated as `9 - log10(Kd in nM)`. There are 379 unique full-length protein sequences among the 442 entries. All these sequences have frozen ESM features available in the local cache; no expensive feature extraction was needed.

**69.64% of rows have Kd = 10,000 nM / nominal pKd=5.** The censored-loss arm heuristically interprets these as threshold observations rather than exact pKd values. This is NOT validated against individual original assay records. Full raw source, cached features, and training binary checkpoints remain on the user's computer, excluded from public Git.

Split: fixed seed `20261109`; identical sequences and connected groups with TF-IDF character-trigram cosine ≥0.90 were kept in the same partition. **That is a sequence-neighbor proxy, not curated kinase-family or homology identity annotation.**

| Partition | Proteins | Pair observations |
|---|---:|---:|
| Training | 302 | 20,536 |
| Validation | 59 | 4,012 |
| Held-out test | 81 | 5,508 |
| **Total** | **442** | **30,056** |

378 sequence-neighbor groups in the full reference, 57 of them in the test partition. Max train/test trigram cosine was **0.7168** (but this metric is not biological proof of family independence). The model-family and epoch selection used VALIDATION only. The TEST metrics were calculated only after `SELECTION_FROZEN.json` had been written.

## 3. Model architecture and what fixes were tested

The model inputs are pretrained **mean-pooled ESM-2 protein features** and **Morgan fingerprint features**. The neural network consists of a 1280→96→32 protein projection and a 2048→128→32 compound projection, with LayerNorm, GELU and dropout (0.15). For one target–compound pair, concatenate the two 32D features, their elementwise product, and their absolute difference; a 128→64→1 MLP predicts a residual added to the training-only compound mean prior. Output residual head initialized to zero. This is a lightweight, regularized protein–compound interaction predictor, not a 3D graph network.

Three configurations were predeclared, trained for up to **45 epochs**, and compared using the same frozen validation set and two seeds (2931 and 2932):

1. **Raw** L2-normalized ESM features plus standard MSE.
2. **Centered/standardized** ESM features (mu/sigma fitted ONLY on the 302 training proteins) plus standard MSE.
3. Same centered/standardized ESM plus **censored-aware loss**, using a one-sided hinge penalty for observations at nominal pKd=5. Evaluation still uses nominal pKd RMSE for fair common-score comparison.

All arms share the same two-tower architecture, chemical-prior construction, AdamW, 0.001 learning rate, 0.03 weight decay, nominal pKd metric, and fixed training/batch sequence. Two independently selected seeds were ensembled at prediction level.

**No double-cold OOF residual training, expert routing, 3D geometry, or original PerceiverCPI training took place in this v5 test.**

## 4. VALIDATION (selected without test results)

Validation target-macro RMSE (lower better):

| Model | Validation macro RMSE |
|---|---:|
| Frozen chemical-only (TRAIN compound mean) | 0.656721 |
| Raw ESM + standard MSE, 2-seed ensemble | 0.528493 |
| Train-centered ESM + standard MSE, 2-seed ensemble | 0.522935 |
| **Train-centered ESM + censored-aware loss, 2-seed ensemble** | **0.516913** |

The chosen configuration is `centered_esm_censored`. Its best seed checkpoints were independently selected on validation at **epoch 29** for seed 2931 and **epoch 24** for seed 2932. This is direct evidence that a 5-epoch cutoff would prematurely terminate this particular *new* low-rank architecture, but it does not prove the old v4.3 architecture would benefit by the same amount.

## 5. HELD-OUT TEST (one selected configuration only)

| Metric | Chemical-only prior | Selected low-rank DynaCPI v5 | Relative change |
|---|---:|---:|---:|
| Target-macro RMSE ↓ | 0.823596 | **0.585203** | **−28.95%** |
| Pooled RMSE ↓ | 0.871410 | **0.656132** | −24.70% |
| Target-centered MSE ↓ | 0.675195 | **0.378817** | −43.89% |
| Protein bias MSE ↓ | 0.084161 | **0.051693** | −38.58% |

The model improves per-target RMSE for **68/81** individual held-out target entries and worsens it for **13/81**. A 10,000-resample **descriptive** bootstrap over 57 held-out proxy protein groups yielded a group-weighted paired RMSE-difference interval of **[−0.24024, −0.11769]**. This is a summary for one fixed data split, NOT a confidence interval for independently verified kinase families or other datasets.

The result is a **concrete improvement versus the chemical-only baseline ON THIS EXACT SPLIT**. It is *not* comparable to the old v4.6 double-cold score 0.607 because split and training population are different. No PerceiverCPI head-to-head exists yet.

## 6. Reproducibility and demo files

Sources in this repository:
- `prepare_full_davis_v5.py` — validates public full DAVIS source files, restricts pickle globals, loads 379 cached ESM feature vectors, regenerates fingerprints, freezes grouped split.
- `train_full_davis_v5.py` — deterministic GPU training, per-epoch train and validation diagnostics, early stopping; 3 variants × 2 seeds, non-destructive checkpoints.
- `finalize_full_davis_v5.py` — recovers completed checkpoints after an initial local serialization error; ensemble VALIDATION selection, freeze on disk, then score TEST once.
- `verify_full_davis_v5.py` — independently reloads the two selected checkpoints and reproduces all headline test metrics, paired error and bootstrap.
- `demo_full_davis_v5.py` — Flask on port 8782, experiment dashboard and a single fixed anonymized benchmark-pair inference with real checkpoint execution only; no arbitrary molecule search or medical recommendations.
- `launch_full_davis_v5.bat` — restart launcher on Windows.
- `test_full_davis_v5.py` — local endpoint, split and result checks.

Local-only outputs under `DynaCPI-Backend/full_davis_demo_20261109/` include the raw public benchmark files, prepared feature arrays, `train_v5/checkpoints/*`, frozen validation selection and all per-epoch trajectories. **Do not upload these binaries to GitHub.**

Public-safe summaries will be copied to `full_davis_v5_public_20261109/`:
- `split_manifest.json` — frozen index split and provenance / limitations
- `PROTOCOL.json` — training policy and fixed hyperparameters
- `RESULTS.json` — training curves, configuration comparison, final test aggregate
- `SELECTION_FROZEN.json` — selected model BEFORE test scoring
- `VERIFICATION.json` — independently recomputed metrics and descriptive group bootstrap

**Known implementation correction:** initial training script completed training all six checkpoints but failed at their restricted deserialization because NumPy arrays in our own checkpoint were not initially allowlisted. We introduced a restricted `torch.serialization.safe_globals` load rather than unsafe arbitrary pickle, preserved the frozen checkpoint files and training histories, and completed the validation/test selection in `finalize_full_davis_v5.py` without retraining. This interruption did not use test scores or change hyperparameter selection.

## 7. Presentation walkthrough

1. Open **http://127.0.0.1:8782/** in Microsoft Edge on the training PC.
2. Show *full DAVIS* / 30,056 observations / 442 protein reference entries / separate training, validation and held-out protein groups.
3. Show test macro RMSE **0.8236 vs 0.5852** and explicitly state **the comparison is versus chemical-only, NOT PerceiverCPI**.
4. Use the **Live model inference** panel to run a fixed anonymized held-out example. This executes both actual saved neural checkpoints, not a mock API. Individual examples may have large errors; the aggregate is not guaranteed for each case.
5. Show validation selection: raw vs centered ESM vs censored-aware objective, then the per-epoch training plot (best checkpoint after 5 epochs).
6. Close with limitations: all compounds warm, proxy protein grouping, censoring heuristic, missing same-split published baseline.

Endpoints:
```
GET  /api/v1/health
GET  /api/v1/metrics
GET  /api/v1/training
GET  /api/v1/examples
POST /api/v1/predict {"demo_id":"DV-01"}  # fixed, anonymized benchmark pair only
```

The older v4.3 API at `127.0.0.1:8781` and current project files remain unmodified. **Do not present this demo as a deployed medical model.**

## 8. Next scientific task

A valid PerceiverCPI claim requires rerunning **original published PerceiverCPI** with the identical newly frozen public DAVIS source, protein-grouped split, label transform, train–val selection, test scoring, model capacity/compute accounting and seed policy. Stronger protein-family curation and independent release-level external validation are still missing. Do not tune on the newly inspected test set.

This completed demonstration is for the upcoming presentation, not a paper-ready head-to-head superiority result.
