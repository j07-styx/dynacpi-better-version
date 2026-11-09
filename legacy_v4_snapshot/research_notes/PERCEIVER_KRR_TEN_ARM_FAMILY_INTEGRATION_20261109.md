# PerceiverCPI KRR-first few-shot: ten-arm source-family trial integration

**DynaCPI v4 | 9 November 2026 | Software-only integration | No scientifically approved assay evaluation**

## Decision this change addresses

The old DynaCPI geometry-versus-Perceiver prototype can beat a calibration-only
Perceiver in a deliberately constructed example and still lose to a
**same-support chemical kernel-ridge** Perceiver comparator. A subsequent
P5-KRR-first candidate introduced a support leave-one-out (LOO) geometry
check; a support-disagreement permutation guard reduced false geometry
activations in a *fabricated* noise stress experiment. Neither is evidence of
real drug–target performance.

Previously the formal source-family evaluator could compare only the earlier
seven arms. **We have now integrated all three stronger KRR challengers** on
exactly the same source-task, support-draw, query, affinity-label and frozen
Perceiver checkpoint representations.

## Ten distinct experimental arms

| Arm | Question |
|---|---|
| `p0` | What does original frozen PerceiverCPI predict without new support? |
| `p5_global` | How much does calibration with K support examples help? |
| `p5_local` | How much does similarity-weighted calibration help? |
| **`p5_kernel_ridge`** | **Primary fixed strong PerceiverCPI KRR comparator with the same support observations** |
| `d5_no_geometry` | Earlier prototype: global-calibration fallback |
| `d5_shuffled` | Earlier prototype: chemically misaligned geometry residual |
| `d5_geometry` | Earlier prototype: global calibration + real geometry residual |
| `krr_loo_geometry` | New prototype: chemical KRR + LOO-gated geometry residual |
| **`krr_permutation_geometry`** | **New proposed challenger: chemical KRR + LOO + support-permutation-gated geometry** |
| `krr_shuffled_permutation_geometry` | Negative control: same KRR/permutation algorithm with deliberately misaligned frozen geometry signal |

`dynacpi/perceiver_family_trial_evaluator.py` now adds the three KRR arms
only when explicitly invoked with `include_krr_challengers=True`; its
original seven-arm behavior remains compatible. Both modes require
`reference_arm` to be chosen in advance from an equal-support original
Perceiver control, never by picking the winner on evaluation outcomes.

The ten-arm comparison reports individual source-family macro RMSE,
pooled-query RMSE and MAE, and *paired source-family* bootstrap intervals
for the new guarded KRR model against (1) fixed original Perceiver KRR,
(2) unguarded LOO KRR+geometry, and (3) shuffled-geometry guarded KRR.

## Label separation and provenance requirements

1. `dynacpi/perceiver_krr_family_prediction_bundle.py` computes all three
   KRR challenger predictions from only an in-memory `PredictionEpisode`
   (whose support observations contain only K known affinities), frozen
   Perceiver predictions and a separate frozen geometry predictor. The
   prediction layer has **no query-affinity-label argument**. Missing
   geometry yields identical baseline KRR values for all three challengers.
2. `dynacpi/perceiver_krr_family_eval_adapter.py` joins a separately
   constructed `EvaluationLabels` object **after** all models have predicted.
   It reuses the existing seven-arm label adapter, hashes the same support
   IDs + values into a fixed panel digest, and validates every prediction
   belongs to the same ordered query panel and support budget.
3. It rejects tampering with the original Perceiver-KRR baseline, LOO
   agreement, the guard's candidate/acceptance decision, ±0.5 pKd cap,
   frozen model alignment, or geometry-absence fallback.
4. `perceiver_family_trial_evaluator.py` confirms the exact same tasks,
   constructs, protein components, source-family digests, query panels,
   support digests and seed/draw grid across all ten arms. It rejects
   invalid family mapping, duplicate rows, changed labels between draws,
   missing KRR arms, nonfinite values or unapproved source identity formats.
5. The evaluator **does not verify a digest against original bibliographic
   evidence or approve a protein construct**. Its source-family and
   construct IDs must originate from a separate independent primary-assay
   review and fully frozen scientific cohort. Simply constructing a hash
   is not scientific approval.

## Synthetic regression, not measured benchmark performance

The new integration test constructs six entirely fictional source families,
24 same-support episodes and 48 fabricated paired query labels. Within this
mock system, the family-macro RMSE is approximately 0.5034 for original
P5-KRR and 0.4677 for the new guarded KRR challenger. The same model is
0.5034 under the shuffled-geometry negative control. It is possible to
obtain a positive source-family bootstrap interval in this fixture because
the author deliberately planted the *same correlated geometry signal* in
several fake families.

**These are not six independent papers, real protein constructs, original
assay outcomes, real frozen Perceiver model predictions or evidence of
generalization.** Treating this fixture as a scientific win would be
circular and incorrect. Its role is verifying paired aggregation, exact
alignment, stable bootstrap, baseline fallback, label isolation and
adversarial rejection. The original prior fabricated stress tests contain
counterexamples and remain preserved.

Test suite:
`tests_v4/perceiver_krr_family_trial_integration_test.py` (16 checks),
plus the unchanged old family evaluator tests and existing KRR/no-geometry
regressions. No source-role-sensitive private artifacts are committed.

## Preregistered next real experiment, not yet executable

The primary comparison would be `krr_permutation_geometry` versus
`p5_kernel_ridge` on a **source-family- and protein-homology-disjoint**,
independently original-assay-reviewed cohort with identical K-shot panels.
Test K=5 primarily and K=0/10 secondarily, with preselected settings,
multiple seeds, original geometry versus support-disagreement-shuffled
geometry, and a separately trained sequence-only capacity control.
A proposed numerical screening target is at least 5% source-family-macro
RMSE reduction, positive paired family-bootstrap lower interval bound,
and no pooled-RMSE deterioration. Real source-family independence and
assay endpoint/construct authenticity must be established *before* numbers
are given scientific meaning.

**The real gate remains `BLOCKED_UNAPPROVED_NOT_TRAINING_READY`. No
BindingDB experimental query labels, protected prospective TEST results,
new model training or scientific baseline-win claims were used.**

## Verification record

The `perceiver_krr_family_trial_integration_test.py` suite passed all
**16 synthetic tests** after correcting the initial pooled-metric bookkeeping
issue; the original seven-arm evaluator tests also passed unchanged.
The wider run completed **161 selected automated tests** successfully,
plus a separate synthetic scientific-cohort-gate regression. The actual
local scientific gate continues to report
`BLOCKED_UNAPPROVED_NOT_TRAINING_READY`.

KRR-challenger mode requires
`reference_arm="p5_kernel_ridge"`, refuses weaker substitute primary
comparators, and reports pooled RMSE/MAE for all ten arms. Neither the
unapproved source manifest nor the reserved prospective TEST labels were
opened, scored, re-split or used for tuning.
