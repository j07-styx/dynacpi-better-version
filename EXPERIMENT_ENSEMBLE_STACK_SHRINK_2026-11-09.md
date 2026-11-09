# DynaCPI v4.6 exploratory seed ensemble, nested stacking and residual shrinkage

**Executed:** 2026-11-09 on the connected Windows RTX 4060 GPU
**Status:** Completed and numerically verified; historical engineering TRAIN-only double-cold folds (NOT an independent study or published PerceiverCPI comparison)
**Run code:** `run_ensemble_stack_shrink_20261109.py`
**Tests:** `test_ensemble_stack_shrink.py` and `test_ensemble_result_integrity.py`
**Detailed artifacts:** `interaction_study_20261109/ensemble_stack_shrink_20261109/PROTOCOL.json`, `PROGRESS.json`, `RESULTS.json`
**Local-only, uncommitted exact pair prediction caches:** `fold_{0,1,2}_predictions.npz`
**Old model checkpoints:** unchanged, original v4.3 unchanged; NO model trained on blocked BindingDB cohorts.

## Experiment

The earlier user-provided critique proposed combining three anchored-sequence seeds at **prediction level** rather than averaging three independent RMSE scores, then adding a chemical-prior-safe stacking/shrink mechanism with inner-data-only tuning.

We implemented the first three cheap hypotheses, not the proposed larger target-centered-loss retraining or a fundamentally improved Morgan prior:

1. **Unweighted seed ensemble:** replay the v4.3 anchored sequence architecture with exactly the old seeds 2931, 2932, 2933 and five-epoch training protocol and average per-pair pKd predictions before measuring RMSE.
2. **Stacking with fallback to chemical:** candidate `prediction = chemical_prior + w*(neural_ensemble - chemical_prior)`, `w ∈ {0, .25, .5, .75, 1}`. `w=0` recovers the chemical-only baseline exactly, and `w=1` recovers the seed ensemble. Optimal *inner* weights are selected without looking at that outer fold's targets.
3. **Similarity-gated shrink:** the above residual optionally multiplied by per-query max training-compound Morgan/Tanimoto similarity, or that value times max cosine to a TRAIN protein pooled ESM-2 embedding. Similarities are **not calibrated confidence**. Candidate gate choices: `global`, `chemical_tanimoto`, `chemical_times_protein_cosine`. The gate and weight were selected together on inner CV; the outer evaluation remained untouched during selection.

Data/selection:

- Original DAVIS engineering **TRAIN only**, 24 protein identities × 54 compounds = **1,296 pairs**.
- Reused three existing paired outer splits. Per fold outer model fit: **16 TRAIN proteins × 36 TRAIN compounds**, evaluate 8 previously held-out protein identities × 18 scaffold/near-neighbor-excluded chemical identities.
- Compound clusters preserve same Murcko scaffold or Tanimoto >=0.5 in the same grouping; all cross-train/heldout chemical similarities were <0.5 under the earlier protocol.
- Proteins disjoint by **identity**, not independently reviewed biological families.
- The chemical-only `AffinityProfile` prior is separately fitted on each outer training rectangle with regularization selected by training-only nested choices, as in prior `run_profile_study.py`.
- Within each outer training rectangle, create **two more train-only paired double-cold splits**. For each inner split refit a chemical-only prior with regularization selected on its own inner-training data, then train all 3 neural seeds for 5 epochs on only inner-training rows. Evaluate candidate weights/gates solely on these double-cold inner evaluation rectangles. Choose lowest mean of two inner target-macro RMSEs, tiebreak smaller weight.
- Refit all three neural seeds on the original outer training rectangle with the selected outer chemical prior and 5 epochs; evaluate the resulting ensemble and inner-selected stack on the unseen-to-that-outer-fit cells.
- `AdamW` learning rate 0.0003, weight decay 0.0001; frozen precomputed ESM-2 inputs, trainable sequence/compound neural head; output label scale from fit data only. All scoring uses **target-macro RMSE** (mean of per-protein RMSE).
- No original 8-target DAVIS DEV labels, unapproved BindingDB source/assay data, or protected test measurements were accessed for training or model selection.
- **Important prior-inspection limitation:** this family of three outer folds, and its chemical-only score, had already been inspected before this experiment was proposed, so this cannot be counted as a new independent/confirmatory test. Even properly nested stack selection does not erase the project-level reuse of the outer data.

## Exact executed results — double-cold target-macro RMSE (lower is better)

| Model / policy | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| Chemical-only profile | 0.562363 | 0.752673 | 0.509107 | **0.608048** |
| Mean of three separate anchored-seed RMSEs (replayed) | 0.558419 | 0.754895 | 0.517522 | 0.610279 |
| **Unweighted per-pair seed ensemble** | 0.556772 | 0.754218 | 0.512101 | **0.607697** |
| **Nested, inner-selected blend/shrink** | 0.556866 | 0.752309 | 0.512101 | **0.607092** |
| Fixed 50/50 chemical/ensemble (predeclared control, NOT inner-selected) | 0.557839 | 0.751435 | 0.507647 | 0.605640 |
| Fixed 50% chemical-similarity gate (predeclared control, NOT inner-selected) | 0.560187 | 0.751801 | 0.508402 | 0.606797 |

The previous v4.3 anchored neural three-seed mean was 0.610480. The freshly replayed 0.610279 is close but not bit-exact (see reproducibility warning below). All rows except original historical comparator use the same **new** replayed model predictions and identical outer splits.

The three *inner-selected* stacking configurations (without outer score tuning) were:
- Fold 0: **global gate**, `w=0.75`; inner mean 0.706787, outer 0.556866.
- Fold 1: **global gate**, `w=0.75`; inner mean 0.572600, outer 0.752309.
- Fold 2: **global gate**, `w=1.00`; inner mean 0.703245, outer 0.512101.

**None of the similarity gates won inner model selection.** The fixed 50/50 blend looks better on these old outer folds, but selecting it *because it looks better now* would be retrospective outer-score model selection. Report it only as a predefined ablation; do not promote it as the scientifically validated winner.

## Effect size and uncertainty

Comparing the **inner-selected** policy with the same-fold chemical prior:

- Paired target-macro change: **−0.000955483 RMSE** (0.607092140 vs 0.608047623), a **0.157% relative reduction**.
- 10,000-resample paired **protein-level descriptive bootstrap** of 24 target error deltas: **95% interval [−0.013246, +0.011519]**.
- 10,000-resample **outer-fold descriptive bootstrap** of three paired fold effects: **95% interval [−0.005497, +0.002994]**.
- Both intervals include zero, by considerable margins. With only three historically inspected paired folds and non-family-grouped proteins, neither interval can establish a novel generalized improvement.
- The unweighted ensemble changed RMSE by only **−0.000351025** vs chemical-only, with protein bootstrap **[−0.013542, +0.012739]** (includes zero).
- The fixed 50/50 control attained 0.605640, but cannot be called the new "winner" from this existing reused data: it was not the inner-selected policy and is not an independently validated scientific result.
- **PerceiverCPI remains unmatched** on identical approved train/test/label/support configurations. This experiment cannot prove a PerceiverCPI victory.

## Reproducibility checks and warnings

- Three per-fold local NPZ files contain exact heldout labels, prior predictions, all three neural predictions, ensemble and stacked predictions; they are ignored by Git and kept on the original local PC.
- `test_ensemble_result_integrity.py` independently loaded the three caches and verified tensor shapes, exact ensemble arithmetic, macro-RMSE recomputation and correct aggregate averaging across all folds: **PASS**.
- `test_ensemble_stack_shrink.py` passed **4/4 synthetic tests** (zero-weight baseline parity, gate bounds, inner selection's ability to fall back to chemical-only, bootstrap-zero-delta behavior).
- The first attempt at neural replay used a slightly different label casting order. That run was **aborted** and preserved as `ensemble_stack_shrink_20261109_aborted_label_precision`; after matching the prior `run_profile_study.py` training-label float64 scaling and one float32 cast, replay differences decreased.
- Despite correction, outer fold 2 seed2933 still differed from its old saved RMSE by about **+0.00148** (old 0.561561, replay 0.563038), plausibly due to GPU/train numerical sensitivity. We recorded this and completed the last fold using a bounded **0.003 replay-consistency tolerance**; this was a reproducibility threshold, not an outer model-selection choice. The first two folds' earlier successful replay and cached predictions were preserved and verified separately. Therefore the new results are a **fresh, approximately matched replay**, not bitwise identical old predictions.
- The `PROTOCOL.json` and `PROGRESS.json` files preserve the original run/continuation record. `RESULTS.json` and the source reflect the final finished protocol. The local-only aborted attempt is not reported as scientific evidence.
- This experiment does not produce a new deployable trained production ensemble checkpoint, and the previously serving backend v4.3 remains unchanged.

## Decision

We obtained a small numerical reduction below the project-internal 0.6080 chemical-only reference, but the uncertainty spans substantial improvement and harm. The most useful result is that **prediction-level ensembling appears more stable than averaging RMSEs**; in this tiny dataset, even properly nested stacking has little margin over chemical-only.

**Next scientific step:** Freeze the tested algorithm *before* acquiring/reviewing new independent held-out chemistry and protein families; perform matched original PerceiverCPI versus DynaCPI with identical support, data transformations, and scoring. Do not keep tuning `w` on these same 3 outer folds. Target-centered auxiliary-loss training and alternative Morgan prior can be tested on new **inner training** partitions, but never claim generalization from this reused double-cold panel.

## Commands (local Windows environment)

```powershell
$py = 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
$root = 'C:\Users\J0124\DynaCPI-Backend'
& $py "$root\test_ensemble_stack_shrink.py"
& $py "$root\test_ensemble_result_integrity.py"

# DO NOT rerun the training script into the completed result directory.
# To replicate, copy and rename the experiment's output folder constant,
# preserving all existing trained-result artifacts and source hashes.
& $py "$root\run_ensemble_stack_shrink_20261109.py"
```

Public GitHub includes the source script, four synthetic tests, integrity verifier, and non-sensitive aggregate/provenance JSONs. Local .npz caches are intentionally excluded.
