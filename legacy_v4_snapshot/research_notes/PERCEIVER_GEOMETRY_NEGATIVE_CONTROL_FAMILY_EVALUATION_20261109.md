# DynaCPI fair-baseline negative controls and source-family evaluation

**9 November 2026 | Synthetic and software-only checkpoint | No approved biological benchmark**

## Objective

DynaCPI should not be called better merely because it changes predictions or
outperforms an unadapted PerceiverCPI. Its added geometry signal must produce
a reliable gain over a **preselected strong, equal-support PerceiverCPI control**
and over geometry-free and chemically misaligned geometry controls, assessed
across independent publication-linked **source families**, not only proteins.

This work implements the missing comparison and evaluator infrastructure.
It does not load BindingDB affinity outcomes, train any model, or open the
archived prospective test.

## New negative control: randomize the compound alignment, not the labels

`dynacpi/perceiver_geometry_negative_control.py` creates a reproducible
within-task/assay *cyclic derangement* of the **frozen geometry–Perceiver
prediction disagreement** across compound IDs. The same exact support labels,
support molecules, query molecules and Perceiver predictions are retained.
The residual *multiset* is unchanged; each residual is assigned to another
compound, using a deterministic task/assay/ID hash and predeclared shuffle seed.

The comparator emits the original D5 geometry result, deranged D5 result and
no-geometry P5-global result for identical queries. The negative control
tests whether *compound-specific alignment* matters; it is **not** a true
protein-structure shuffle, a separate geometry-network capacity ablation,
or a source-family generalization result. Future real evaluation should fix
multiple independent shuffle seeds before seeing query outcomes.

## New paired, source-family-aware evaluator

`dynacpi/perceiver_family_trial_evaluator.py` accepts evaluation-only
records containing the same query, label, support-panel SHA256, seed/draw
and original/negative-control predictions across seven arms:

`p0`, `p5_global`, `p5_local`, `p5_kernel_ridge`,
`d5_no_geometry`, `d5_shuffled`, and `d5_geometry`.

The required `reference_arm` **must** be one explicit prespecified
equal-support Perceiver control; the evaluator refuses `p0` or
a free-text `best_on_test` selection.

It calculates query RMSE per episode, averages per task, construct, protein
component, then *connected source family*, and finally averages families.
It separately reports pooled query RMSE and MAE. Paired 95% bootstrap intervals
resample complete **source families**, never treating correlated protein
components in one publication-connected family as independent studies.
Five source families are a minimal software sanity gate, **not** a claim
that five families establish scientific confidence. Future study design
needs substantially more genuinely independent reviewed sources.

The evaluator rejects duplicate queries, inconsistent support-panel hashes
within one draw, task/construct/component/source role conflicts, nonfixed
query or seed/draw grids, missing inputs, changes to query truth across draws,
nonfinite predictions, unsupported information budgets and broken
no-geometry equality. It does not certify the source-family mapping itself:
that requires the separate approved-source gate and primary-method review.

## Explicit query-label boundary

`dynacpi/perceiver_trial_record_adapter.py` converts the **already computed**
inference-only predictions into evaluator records. Only this adapter accepts
the separately held `EvaluationLabels` object. It enforces ordered query
identity and hashes exact support IDs and support-label payloads into one
support-panel identifier for paired consistency. It accepts a separate
predeclared task-level geometry availability flag, not a guessed activation
state. The source family and original construct ID must be supplied by the
future independently approved cohort; this adapter does not confer approval.

## Synthetic adversarial finding: a stronger fair baseline can still win

A fabricated eight-source-family / nine-protein-component evaluation was
intentionally built with a known geometry effect present in only some
families. Its family-macro *fictional* RMSEs were:

| Arm | Synthetic family-macro RMSE |
|---|---:|
| P5-global | 0.200 |
| P5-kernel ridge | **0.110** |
| D5-geometry | 0.125 |
| D5-shuffled | 0.225 |

Thus D5 improves against the calibration-only and the shuffled control, but
**fails the strongest prespecified Perceiver kernel-ridge comparison**.
This unfavorable result is preserved as a regression assertion to prevent
synthetic demonstration cherry-picking. It has **zero bearing on real CPI
accuracy** and is *not* a paper-relevant effect size.

An additional end-to-end pipeline fixture builds fictional predictions
and support panels in six nominal source families, then joins fabricated
query labels **after** prediction and verifies the family evaluator accepts
24 episodes / 48 paired query rows without leaking query truth upstream.

## Reproducibility

- `dynacpi/perceiver_geometry_negative_control.py`
- `dynacpi/perceiver_family_trial_evaluator.py`
- `dynacpi/perceiver_trial_record_adapter.py`
- `tests_v4/perceiver_geometry_negative_control_test.py`
- `tests_v4/perceiver_family_trial_evaluator_test.py`
- `tests_v4/perceiver_fake_fewshot_pipeline_test.py`

These modules are designed for a future scientifically approved and
pre-registered K-shot evaluation. They do **not** implement a new training
entrypoint or automatically promote any architecture. The prior provisional
split remains archived; original source/construct review remains incomplete.

**Scientific gate: `BLOCKED_UNAPPROVED_NOT_TRAINING_READY`.**
No new experiments used actual binding labels, and no protected prospective
TEST outcomes were inspected. Real superiority over PerceiverCPI is **not
demonstrated**.
