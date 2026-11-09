# DynaCPI: baseline-beating experiment charter (prospective design)

**8 November 2026 | Architecture pivot, code-only prototype | No new affinity training or final evaluation**

## Objective: beat strong original PerceiverCPI, not an artificially weakened baseline

The matched short-schedule provisional inner-DEV pilot favored original PerceiverCPI at **1.325136 component-macro RMSE** versus **1.508015 for TransferCPI** (seed 2601, five epochs, different architectures/compute). PerceiverCPI pooled RMSE was 2.019847 and a train-only mean reference was 1.803902; model selection by macro alone did not deliver pooled robustness. Neither result is scientifically confirmatory because source-family split independence failed on retrospective audit.

**New principle: keep the frozen original Perceiver predictor as an anchor**; learn only changes that demonstrate added value under a scientifically valid, source- and homology-disjoint evaluation. Do not automatically grant new DynaCPI variants the adjective "better" because they have geometry, large ESM, ChemBERTa or additional attention blocks.

## Priority 1 — Target-specific adaptation, K=5 (main research bet)

Scientific question: can a modest, bounded geometric/sequence residual improve performance on an unseen target when given five assay-compatible known affinity observations, *over a strong PerceiverCPI comparator receiving the same five observations*?

Prespecified four nested **frozen PerceiverCPI + equal-support** control predictions are implemented in `dynacpi/perceiver_equal_support_controls.py`:

| Arm | Available assay labels | Computational method |
|---|---|---|
| **P0** | K=0 (or intentionally ignores support) | Frozen original PerceiverCPI |
| **P5-global** | Exact same five support labels | Shrunken target-level residual offset |
| **P5-local** | Exact same five support labels | Query-weighted, stereo-aware Morgan Tanimoto^4 residual offset |
| **P5-KR** | Exact same five support labels | Regularized kernel ridge on residuals, with bounded prediction update |
| **D5** (future experiment) | **Exact same support and queries** | Frozen Perceiver baseline plus preregistered DynaCPI residual/adaptation; compare against **best preselected** fair P5 comparator |

Additional information budgets K=0 and K=10 test a dose-response, with K=5 the primary target. Same task, assay, construct, exact query compounds, support compounds, random seeds, chemical fingerprints and same predeclared regularization/tuning policy. Supports are held outside the query panel; all methods must receive the same support measurement information. D5 must not exploit any unreviewed assay or prospective-test data.

The existing `dynacpi/frozen_adapter.py` contains A/B/C/D geometry and local adaptation contrasts, but it must not be reported as a PerceiverCPI-relative improvement until its frozen base is truly the original PerceiverCPI and it beats **these same-support controls**. No guarantee of success comes from "support learning" in isolation.

## Priority 2 — Preserve PerceiverCPI at initialization; learn bounded residuals

`dynacpi/perceiver_anchored_residual.py` is an **opt-in trainable head** on cached, detached baseline predictions:

- `p_D = p_Per + availability * 0.5 * tanh(f_theta(features))` with a zero-initialized last layer, so `p_D = p_Per` exactly before training.
- `availability=0` enforces exact fallback, including after parameters change, when a required experimental feature (e.g. usable 3D geometry) is unavailable.
- Frozen baseline gradients are disconnected, and absolute learned corrections are bounded; the feature head alone receives gradients.
- Geometry/sequence residual features are not proof of structural information being useful. Run a **sequence-only residual capacity-matched control**, a **shuffled/ablated geometry control**, and an **availability-only calibration control**.

Why this could help: the newer model will not start from a worse backbone. A weak geometry correction can be suppressed instead of replacing the known functioning predictor.

It can still lose after fitting or under distribution shift. The current code is NOT trained; zero initial correction and unit tests are engineering properties, **not measured predictive improvements**.

## Priority 3 — Robustness and efficiency as independent potential wins

- **Cold proteins and cold scaffolds:** prespecify source-family and certified protein-homology disjointness; keep scaffold-disjoint and cross-assay evaluations separately reported. Never use cross-source study groups to tune the model.
- **Calibration / catastrophic error:** report component-macro RMSE (primary) **and pooled RMSE, MAE, large-error frequency and calibration**, because the prior pilot's macro-versus-pooled discrepancy matters.
- **Compute efficiency:** if DynaCPI does not lower RMSE, it can still improve on parameter count, inference throughput or memory **only if measured against original PerceiverCPI on identical hardware and matched inputs**. No current efficiency win is claimed.

## Fairness, success criteria and stop rules (hypotheses, not results)

1. Before fitting anything, obtain an independently **approved** BindingDB experimental cohort and preregister a new source-family- and homology-disjoint study split. The previous provisional split is permanently unsuitable for scientific training/model selection. Current gate: `BLOCKED_UNAPPROVED_NOT_TRAINING_READY`.
2. Freeze original PerceiverCPI checkpoint, its preprocessing and the support-label budgets. Only train a controlled residual head when the scientific cohort has formally passed the source-aware gate. Separate training seeds and support draws must be fixed in advance; match actual GPU-hours or FLOPs as well as epochs when comparing from-scratch alternatives.
3. For **primary K=5 unseen-protein-family evaluation**, a *proposed* promotion threshold is at least **5% reduction in component-macro RMSE** versus the strongest **preselected** PerceiverCPI K=5 control, with positive paired component-bootstrap lower bound at 95% confidence and non-degrading pooled RMSE. Use the same predeclared task, seed and support-draw grid. The bootstrap is descriptive until enough independent reviewed source families exist.
4. Track K=0 and K=10 as secondary; require either a reproducible error win, a properly quantified calibration win or a hardware-matched efficiency win. Never reinterpret a support-only win versus P0 as a novel architectural win.
5. **Stop or simplify** after the predefined experiment budget if gains vanish against P5-KR, if the 95% uncertainty interval crosses zero, if improvements occur only on pooled vs macro objectives without a prespecified tradeoff, or if geometry ablations match the real geometry branch. Do not tune on the reserved test or selectively revise the planned metric after seeing outcomes.

## What was implemented now

- `dynacpi/perceiver_anchored_residual.py`: baseline identity at initialization, detached baseline, strict tensor/availability validation, bounded optional improvement path.
- `dynacpi/perceiver_equal_support_controls.py`: P0, P5-global, P5-local and P5-KR; reads only in-memory support labels and cached frozen Perceiver predictions; cannot read query labels via its API.
- `tests_v4/perceiver_anchored_residual_test.py` and `tests_v4/perceiver_equal_support_controls_test.py`: synthetic-only mathematical, boundedness, privacy and same-support invariants.

**No PerceiverCPI baseline checkpoint was changed; no new model was fit; no BindingDB assay was approved; no reserved-test labels, metrics or predictions were accessed.** The new modules are a ready-to-evaluate hypothesis package, not demonstrated superiority.
