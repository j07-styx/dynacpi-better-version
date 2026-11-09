# DynaCPI v4.7 — scaffold-excluded OOF chemical-prior residual labels

**Executed:** 2026-11-09, NVIDIA RTX 4060 Laptop GPU, PyTorch 2.5.1+cu124.
**Scientific status:** exploratory historical DAVIS TRAIN-only re-evaluation. **No new independent test, no win over published PerceiverCPI.**
**Hypothesis:** the old in-sample fitted chemical profile makes neural residual targets artificially small on seen training compounds. Training residuals against a chemical prior that excludes each compound's full Murcko/near-neighbor group could improve cold-compound generalization.
**Code:** `run_oof_residual_v47_20261109.py`
**Tests:** `test_oof_residual_v47.py` (3/3 PASS), `test_oof_results_integrity_v47.py` (all saved scores independently recomputed, PASS).
**JSON outputs:** `interaction_study_20261109/oof_residual_4inner_deterministic_20261109/{PROTOCOL,PROGRESS,RESULTS}.json`. Pair-level output caches remain private to the local PC.

## 1. What was changed

Two otherwise matched branches were trained from the **same** random seed initialization, with the same graph/protein backbone, model capacity, optimizer, 5 epochs, minibatch order and data:

- **Control — in-sample residual targets:** `(y - prior_full_fit[train_compounds]) / sigma_train`.
- **Treatment — 4-way scaffold-neighbor-group-excluded OOF residual targets:** for every training chemical cluster, fit a separate chemical prior on the other 3/4 of chemical clusters (using the same TRAIN proteins) and predict each held-out training compound. Train the network on `(y - prior_scaffold_OOF[train_compounds]) / sigma_train`. **At inference, both models use the identical original full-fit chemical prior.** This means the experiment specifically changes the residual target semantics, not inference prior quality.

In both cases the scoring prediction is `full_fit_chemical_prior + neural_residual` before optional blending. No geometry, routing, or BindingDB data was used.

A scaffold-neighbor group is a connected component of identical Murcko scaffold OR 2048-bit chiral Morgan radius-2 Tanimoto >= 0.5. **Each group is entirely omitted from its OOF prediction fit**; chemical-prior ridge selection is independently rerun within the other available training chemicals. No held-out group's affinity labels enter its own prior fit. This was verified by label-mutation unit tests.

Four nested inner protein/compound splits were used per outer fold (inner fit **12 proteins × 27 compounds**, rather than previous 8 × 18). On each inner training rectangle the chemical prior and OOF chemical priors were fit independently and both neural arms retrained with three seeds (2931/2932/2933). The global blend weight `w` was selected from `{0, .25, .5, .75, 1}` by **mean target-macro double-cold RMSE across all four inner validation splits**. `w=0` exactly recovers chemical-only. There are **no similarity gates**.

The outer score evaluation uses the exact previously inspected three historical folds from `run_profile_study.py`, where the fit uses 16 proteins × 36 compounds and the double-cold evaluation uses eight different protein identities and eighteen different chemicals, grouped by scaffold/near-neighbor component. Protein families/homology components are **not independently split**.

## 2. Numeric outcomes (target-macro RMSE; lower is better)

### Double-cold: unseen protein identities and unseen chemical groups

| Arm | Fold 0 | Fold 1 | Fold 2 | Three-fold mean |
|---|---:|---:|---:|---:|
| Chemical-only prior | 0.562363 | 0.752673 | 0.509107 | **0.608048** |
| In-sample-residual neural, inner-selected blend | **0.558181** | 0.751562 | 0.511522 | **0.607088** |
| Scaffold-excluded OOF-residual neural, inner-selected blend | 0.562363 | **0.749644** | **0.535013** | **0.615673** |

Inner-selected global blend weights: in-sample = **0.50, 0.25, 0.50**; OOF = **0.00, 0.25, 0.50** for folds 0, 1, 2 respectively. In fold 0 the OOF neural arm's selected weight was zero: it fell back exactly to the chemical prior.

Important: the old v4.6 inner-selected stack was 0.607092 and this fresh v4.7 in-sample matched control was 0.607088. The ~0.000004 numerical difference is NOT an independent gain and cannot establish an exact bitwise replay of v4.6. This new comparison is matched **within v4.7** with deterministic settings.

### Cold-compound: new chemicals, protein identities seen during training

| Arm | Fold 0 | Fold 1 | Fold 2 | Three-fold mean |
|---|---:|---:|---:|---:|
| Chemical-only prior | 0.711121 | 0.835655 | 0.579208 | **0.708662** |
| In-sample-residual neural, same inner-selected weights | **0.696477** | 0.836550 | 0.567206 | **0.700078** |
| OOF-residual neural, same inner-selected weights | 0.711121 | **0.833702** | **0.545596** | **0.696806** |

Here OOF performs **0.011856 lower RMSE (1.67% relative)** than chemical-only, and **0.003272 lower** than the in-sample control. This is a positive **secondary** result under a double-cold-selected blending policy; the primary double-cold result still got worse. Most of the observed cold-compound improvement comes from fold 2, so it is fragile.

### Additional fixed-weight and unblended controls

| Model | Double-cold mean | Cold-compound mean |
|---|---:|---:|
| Chemical-only | 0.608048 | 0.708662 |
| In-sample full neural ensemble, `w=1` | 0.611084 | 0.697213 |
| In-sample fixed 50/50 | 0.607057 | 0.700602 |
| **In-sample inner-selected** | **0.607088** | 0.700078 |
| OOF full neural ensemble, `w=1` | 0.635813 | **0.688613** |
| OOF fixed 50/50 | 0.613779 | 0.690147 |
| **OOF inner-selected** | **0.615673** | 0.696806 |

These fixed controls were predeclared, but using their outer scores retrospectively to choose a winner would introduce selection bias. The OOF `w=1` variant's 0.688613 cold-compound score is NOT a valid reason to select it after looking at this table.

### Direct training-prior optimism diagnostic

| Outer fold | Training compound in-sample chemical RMSE | Same compounds, scaffold-excluded OOF chemical RMSE |
|---|---:|---:|
| 0 | 0.666948 | **0.728905** |
| 1 | 0.532246 | **0.616939** |
| 2 | 0.767741 | **0.881467** |

In all three folds the OOF chemical prior has more error on training compounds. This is **consistent with** the proposed in-sample optimism. But it also reflects fitting the OOF prior on **fewer chemical groups**, possibly different selected ridge strengths, and a change in representation coverage. These factors are confounded; this table does not isolate kernel self-similarity as the unique cause.

### Paired uncertainty on PRIMARY double-cold outcome

Paired per-protein RMSE differences relative to the chemical prior, with 10,000 descriptive bootstrap resamples across 24 previously inspected protein identities:

- **In-sample selected:** mean difference −0.000959; 95% descriptive interval **[−0.008391, +0.007010]**.
- **OOF selected:** mean difference **+0.007626**; interval **[−0.007979, +0.025631]**.

Both contain zero. These intervals are not confirmatory confidence statements about independent protein families, and the split history invalidates a publication claim. We intentionally do **not** report a misleading three-outer-fold bootstrap interval.

## 3. Reproducibility and leakage audit

- `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set **before importing torch**, and seed reset separately for both arms. The run finished without a nondeterministic kernel exception. **A second full clean-room rerun was not performed**, so bitwise parity over separate runs is not independently established.
- A module provenance assertion requires `profile_model.py`, `repaired_model.py` and `run_profile_study.py` to load from the current backend repo, never silently from the old checkout.
- New v4.7 code makes **no outer-score replay acceptance checks or 0.003 tolerance**. It does not condition training, hyperparameters or continuation on earlier outer results.
- Prior fitting and OOF ridge choices strictly use training-only y submatrices. Held-out scaffold group exclusion is explicitly checked, including full group integrity.
- Exact pair predictions are stored as local `.npz` with labels and remain **Git ignored**, while public reports contain only aggregate metrics and non-sensitive protocol/seed metadata.
- `test_oof_residual_v47.py`: **3/3 PASS** (held-out group labels cannot change OOF predictions for that group, fallback-to-chemical at w=0, nontrivial OOF effect).
- `test_oof_results_integrity_v47.py`: **PASS** (all three saved pair caches reproduce selected and unblended macro RMSE values; selected blend weights minimize the saved inner-only grids; source SHA256 and deterministic policy validated).
- During initial implementation, the first process stopped with a **Python positional argument order bug**, before producing any outer scores. Its incomplete protocol was preserved locally in `oof_residual_4inner_aborted_argorder_20261109`; the corrected run used a fresh directory and completed. The aborted directory is excluded from Git.
- No original DAVIS 8-protein DEV labels, unapproved BindingDB assay outcomes or protected prospective test scores were opened or used for model selection.

## 4. Interpretation / next action

The chemical-prior optimism hypothesis is well motivated and is supported by the TRAIN-only prior-diagnostic comparison. The OOF variant shows an **interesting secondary gain for novel chemicals on familiar proteins**, but **hurts the primary double-cold setting**. Thus, under these fixed settings, simply replacing in-sample residual targets with scaffold-excluded OOF targets is **not a general model improvement** and should not replace v4.3/v4.6 in the live demo.

Do not rerun more hyperparameter sweeps on these outer folds. To test the OOF branch scientifically, first freeze an endpoint and protocol, source-verify an independent protein-family/chemical-scaffold split, and run equal-budget original PerceiverCPI and chemical-only controls on that identical approved dataset. Target-centered residual supervision and stronger chemical priors remain possibilities, but require fresh train-only selection and a genuinely new evaluation.

## Local execution

```powershell
$py = 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
$root = 'C:\Users\J0124\DynaCPI-Backend'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONHASHSEED = '0'
$env:OPENBLAS_NUM_THREADS = '1'
& $py "$root\test_oof_residual_v47.py"
& $py "$root\test_oof_results_integrity_v47.py"
# Completed experiment refuses to overwrite its output.
& $py "$root\run_oof_residual_v47_20261109.py"
```

**Status:** completion verified. This is a research engineering negative result with a narrow secondary benefit, not a published PerceiverCPI benchmark win.
