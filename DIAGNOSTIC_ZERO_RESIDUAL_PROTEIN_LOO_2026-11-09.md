# DynaCPI v4.8 read-only diagnostic — zero residual loss and protein selectivity neighbors

**Date:** 9 November 2026
**Status:** Complete, exact old chemical-prior score reconstruction; no new model training.
**Executable diagnostic:** [diagnose_zero_residual_protein_loo_20261109.py](diagnose_zero_residual_protein_loo_20261109.py)
**Numeric provenance:** [RESULTS.json](interaction_study_20261109/zero_residual_protein_loo_diagnostic_20261109/RESULTS.json)

## Goal

Answer the external reviewer's two zero-compute questions:

1. Is the five-epoch anchored neural training loss close to the loss of simply predicting **zero residual** on the original **outer fitting rectangles**?
2. Is there an affinity-selectivity-near protein in the original 24 TRAIN kinases, and does raw **mean-pooled, uncentered, L2-normalized ESM-2 cosine** choose it?

**All numeric values below were recomputed from the saved historical experiment's TRAIN protein/compound matrix and frozen ESM caches.** The original DEV source file was read and immediately filtered to `split=='train'`; no DEV labels were used in calculations. The existing saved three-fold v4.3 `study_results.json` supplies the already-selected chemical ridge parameters, fold indices, and training histories. The analysis is not a new independent benchmark and should not be used for extra model shopping on the same folds.

## 1. Anchored neural zero-residual TRAIN baseline

For outer fold f, use only the original 16 × 36 fitting rectangle. Refit the exact original chemical-only `AffinityProfile` prior with the ridge parameters already selected by nested training CV. Let `p_c` be the fitted prior for training compound c and `sigma` the observed TRAIN label std over that rectangle:

```
zero_residual_standardized_train_mse =
    mean_{p,c in outer FIT} [ ((y_{p,c} - p_c) / sigma)^2 ]
```

This is the MSE of an anchored network that **predicts zero for every residual**, not a validation loss. We compare it to `training_history.anchored_neural_seed*` from the saved v4.3 study (three seeds, first/fifth epochs):

| Outer fold | Zero residual TRAIN MSE | Mean epoch-1 TRAIN MSE | Mean epoch-5 TRAIN MSE |
|---|---:|---:|---:|
| 0 | 0.84305850 | 0.87034763 | 0.76841818 |
| 1 | 0.76059418 | 0.77345217 | 0.76216929 |
| 2 | 0.76637172 | 0.79015577 | 0.69900319 |
| **Mean** | **0.79000813** | **0.81131852** | **0.74319688** |

- Training MSE improvement from zero residual after 5 epochs: **−0.04681125 standardized MSE**, around **5.93%** of zero-residual baseline.
- Fold 1 stays **0.001575 MSE worse** than zero-residual at epoch 5; folds 0 and 2 improve.
- Epoch-1 recorded loss is an epoch-average over sequential optimizer-updated minibatches, not the initial untrained model's exact full-fit zero-residual loss.
- All results are **TRAIN fit diagnostics**: they do not establish undertraining, optimization convergence, protein-generalization or any unseen-compound benefit.
- The neural architecture does optimize residuals somewhat; it is false to say it never learns anything.

Reconstruction sanity check: original saved chemical-only double-cold macro RMSE and recomputed prior match for every fold:
- Fold 0: original and recomputed **0.5623630427927879**
- Fold 1: original and recomputed **0.7526730805341569**
- Fold 2: original **0.5091067453138612**, recomputed **0.5091067453138614** (floating-point rounding).

## 2. LOPO warm-chemical selectivity diagnostic

Use exactly the 24 original TRAIN proteins and 54 TRAIN compounds. For heldout protein q, let `c(x) = x - mean_compound(x)`. Compare:

- **Chemical baseline**: `mean((c(mean(y_other_proteins, axis=0)) - c(y_q))²)`.
- **Label-informed optimistic single donor**: `min_{p != q} mean((c(y_p) - c(y_q))²)`, selected after observing *heldout* `y_q`.
- **ESM nearest donor**: choose `argmax_{p != q} cosine(mean_residue_ESM_q, mean_residue_ESM_p)`, then compute the centered affinity MSE of that donor. All 54 chemical identities are warm, unlike the main double-cold evaluation.

Mean squared error, averaged over 24 held-out TRAIN proteins:

| Diagnostic | Target-centered MSE (lower better) |
|---|---:|
| Chemical mean | **0.4076998968** |
| Label-informed best SINGLE donor ("oracle") | **0.3213377099** |
| Raw mean-pooled ESM cosine nearest SINGLE donor | **0.6308665743** |

ESM nearest-neighbor equals the label-informed optimal donor in **1/24** cases. Mean nearest-protein raw ESM cosine **0.96672263** (range **0.78846426–0.98624950**).

Interpretation:
- The best single target, chosen using heldout affinity labels, beats the chemical mean by ~**21.2% MSE**, so some targets have potentially useful selectivity neighbors.
- Raw uncentered mean-pooled ESM cosine nearest donor is ~**54.7% WORSE MSE** than chemical mean. Raw cosine is therefore a poor selectivity-neighbor selector in this small set.
- This is **not a prospective external test**. The oracle can use the withheld protein's true affinities: it is an optimistic *label-informed diagnostic*, not a deployable model or a formal upper bound on every weighted ensemble/kernel model. A mixture of multiple training proteins may outperform the best single protein.
- High raw ESM cosine is consistent with shared protein embedding components dominating similarity, but does **not prove** mean pooling or a particular feature fix caused the predictive failure.

## 3. Direction for further research

The **ESM similarity–selectivity mismatch** is currently the more informative finding than the 5-epoch train-loss magnitude. Investigate protein features only with a **predeclared, genuinely new protein-family-disjoint evaluation**, comparing raw L2-normalized ESM, **training-only centered** and **training-only standardized** variants and, if annotation allows, kinase-domain/site features. Include chemical-only, ESM-KNN, additive ESM profile and matched PerceiverCPI controls. Do not optimize these choices on the three historical outer folds or this repeatedly inspected 24-protein matrix and then claim a paper improvement.

A fixed-budget inner-only 5/20/50-epoch probe could still determine whether optimization converges, but avoid choosing a model with the old outer fold errors.

## 4. Reproduction / limits

Run on the original connected Windows PC with local untouched data/caches:

```powershell
$py = 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
& $py 'C:\Users\J0124\DynaCPI-Backend\diagnose_zero_residual_protein_loo_20261109.py'
```

The script refuses to overwrite an existing results directory. To reproduce independently, use a fresh directory while retaining the original data SHA256 and result SHA256. Public Git includes **only** the script and summary JSON, not original raw assays, label matrices or ESM embedding caches.
