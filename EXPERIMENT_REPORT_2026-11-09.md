# Matched two-path chemistry repair — executed results

**Date:** 2026-11-09
**Status:** Finished engineering development experiment; no independent scientific validation
**Code:** `repaired_model.py`, `train_repaired_full.py`, `train_original_matched.py`, `audit_simple_baselines.py`
**Source:** Local legacy `dynacpi_v4` DAVIS pilot TRAIN only, precomputed ESM-2 features

## Experimental question

Does restoring direct molecular fingerprint and compound–protein interaction paths improve our v4 model's chemical discrimination and internal target-generalization compared with the identical training budget for the original architecture?

Candidate: `compound = LayerNorm(MPN(SMILES) + FP(Morgan))`. A second bypass gives the regression head `[compound, protein, compound*protein, |compound - protein|]`. The inherited compound/query attention and ESM-projection weights remain trainable, but the two original channel-attention bottlenecks are bypassed for the candidate. Neither arm receives 3D geometry.

## Execution and provenance

- 24 original DAVIS **TRAIN** protein targets, 54 unique known molecules, **1,296 supervised training pairs**.
- Same original DAVIS **DEV** partition: 8 targets × 54 molecules = **432 pairs**, already inspected in previous experiments.
- Seeds: **2831, 2832, 2833** for both arms, matched initial shared foundation parameters, same optimizer AdamW learning rate 0.0003, weight decay 0.0001, **5 epochs**.
- Epoch count chosen *before fitting this full training cohort* using the earlier internal training-only 18/6 protein split (candidate best epoch 10/5/5; median five).
- All original 24 TRAIN targets—including the prior six inner-validation targets—are used to fit the final checkpoint; therefore the prior inner split is NOT a post-fit holdout for these final models.
- Eight DEV targets scored **only after saving the fixed five-epoch checkpoint** and never used to select the shown epochs; nevertheless that panel was repeatedly consulted by earlier work, so it is not independent.
- No BindingDB unreviewed source task, protected TEST label, or geometry feature was used in training.
- Feature, training-row, and architecture SHA-256 hashes are stored inside each checkpoint. Local GPU lock respected.

## Results — target-macro RMSE (smaller is better)

| Seed | Original after same five epochs | Repaired after same five epochs | Relative RMSE reduction |
|---|---:|---:|---:|
| 2831 | 0.808063 | 0.763457 | 5.52% |
| 2832 | 0.810654 | 0.750148 | 7.46% |
| 2833 | 0.809548 | 0.740614 | 8.52% |
| **Mean** | **0.809422** | **0.751406** | **7.17%** |

Changes are paired by seed and evaluation panel. Three random seeds do not create three independent experiments. The candidate has more parameters; the result is **not capacity matched**.

## Crucial simple baselines — same 24 TRAIN proteins and 8 previously inspected DEV proteins

| Predictor | Target-macro DEV RMSE |
|---|---:|
| Global training mean | 0.808113 |
| Original five-epoch model (three-seed mean) | 0.809422 |
| Repaired five-epoch model (three-seed mean) | 0.751406 |
| **Per-compound training mean across all 24 proteins** | **0.701664** |

The simple compound-average baseline **beats the repaired neural model**. This baseline can be strong because the exact same 54 chemical identities occur in the training and DEV panels; it is not a cold-chemistry method. This materially limits the claim: **the repair improves our original architecture but not the strongest simple comparator in this target-holdout/known-compound setting.**

## Chemical discrimination result

Mean pairwise cosine similarity across the 54 training compound embeddings, after fixed five epochs:

| Seed | Original | Repaired |
|---|---:|---:|
| 2831 | 0.999990 | 0.367937 |
| 2832 | 0.999643 | 0.427199 |
| 2833 | 0.997779 | 0.404538 |

The repaired representation is much less collapsed, consistent with the earlier original holdout's compound-identity shuffle sensitivity. These are representation statistics, **not** ligand affinity performance metrics.

The new `test_model_parity.py` verified that under identical state tensors the separately ported repaired class matches the prior experimental `Repair` class in both chemical embeddings and output. All standalone HTTP integration checks passed, including each new checkpoint, K=5 support calibration, and localhost frontend CORS.

## Next controlled experiments

1. Run a **new-scaffold holdout** to test genuine chemical novelty, not just a known-54-compound transfer. The per-compound mean baseline cannot predict an unseen chemistry ID without an explicitly specified fallback.
2. Add training-only **target-centered auxiliary loss**, and compare against plain MSE at equal updates, with chemically deranged controls. Confirm the compound-aware gains persist rather than merely increasing output variance.
3. Test simple molecular fingerprints with protein features, regularized shrinkage, and a capacity-matched neural alternative. Inspect per-target harms, observed floor-valued measurements, RMSE and MAE.
4. Reintroduce GVP geometry only after a stable repaired sequence baseline passes simple controls. For genuine geometry contribution, compare real/misaligned/edge-corrupted geometry with matching protein and ligand support information.
5. Require source-reviewed, assay-compatible, protein-homology/publication-family disjoint data and an untouched final evaluation to claim PerceiverCPI improvements.

**Do not advertise 7.17% as a published or independent biological benchmark gain.** It is an engineering comparison on a reused development cohort.

## Result artifacts

- `checkpoints/SUMMARY.json`: completed repaired checkpoints and hashes.
- `checkpoints/MATCHED_COMPARISON.json`: matched original versus repaired results, per seed.
- `checkpoints/MATCHED_SIMPLE_BASELINES.json`: global and compound mean performance, per target.
- `checkpoints/seed_283{1,2,3}/COMPLETE.json`: seed-specific metrics and checkpoint digest.
- `checkpoints/original_seed_283{1,2,3}/COMPLETE.json`: matched original metrics.
- `causal_diagnosis_20261109/`: independently supplied original diagnostic and inner TRAIN split investigation.
- Local `checkpoints/**/checkpoint.pt`: trained binary weights, excluded from version control.
