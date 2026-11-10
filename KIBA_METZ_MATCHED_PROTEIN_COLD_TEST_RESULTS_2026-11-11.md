# DynaCPI versus original PerceiverCPI — first matched KIBA/Metz held-out tests

**Experiment date:** 11 November 2026. **Outcome:** DynaCPI achieved lower held-out RMSE point estimates on both KIBA and Metz protein-cold experiments. The paired cluster-bootstrap uncertainty favors DynaCPI on both KIBA metrics but includes zero on both Metz metrics. Neither result proves universal superiority or dominance on other splits.

## Frozen matched held-out results

| Source dataset and split | Held-out observed pairs | DynaCPI two-seed target-macro RMSE ↓ | Original PerceiverCPI target-macro RMSE ↓ | DynaCPI two-seed pooled RMSE ↓ | Original PerceiverCPI pooled RMSE ↓ | Point-estimate result |
|---|---:|---:|---:|---:|---:|---|
| KIBA / unseen proteins | 20,755 | **0.647370** | 0.753914 | **0.608841** | 0.694440 | DynaCPI lower on both |
| Metz / unseen proteins | 5,981 | **0.751161** | 0.777072 | **0.735788** | 0.772058 | DynaCPI lower on both |

Relative to original PerceiverCPI on the identical held-out rows, KIBA macro RMSE decreased approximately **14.1%** and pooled RMSE **12.3%**. Metz macro RMSE decreased approximately **3.3%** and pooled RMSE **4.7%**.

**Do not compare absolute KIBA and Metz RMSE numbers.** KIBA uses a composite KIBA score; Metz retains its own source assay label units.

## Paired bootstrap uncertainty (5,000 predeclared cluster resamples)

Deltas are **DynaCPI minus original PerceiverCPI**; negative favors DynaCPI. Intervals are descriptive percentile intervals, not an independent confirmation across splits or laboratories.

| Scenario | Macro RMSE delta | 95% paired cluster interval | Pooled RMSE delta | 95% paired cluster interval |
|---|---:|---|---:|---|
| KIBA protein-cold | **−0.106545** | **[−0.164013, −0.051187]** | **−0.085599** | **[−0.153684, −0.026865]** |
| Metz protein-cold | −0.025911 | [−0.085858, +0.033047] | −0.036269 | [−0.085000, +0.016172] |

**Interpretation:** Both KIBA intervals lie below zero, supporting its within-split advantage. Both Metz intervals cross zero; report the Metz point-estimate win with this uncertainty rather than claiming a robust or statistically confirmed advantage.

## Controls, provenance, and limits

- **Original PerceiverCPI:** unmodified dmis-lab/PerceiverCPI original source pinned at git SHA `755f076ef51099a90332fd1ec8437f94e3750c0b`, 30 epochs, batch size 128, seeds **2931 and 2932**, original learning and validation-checkpoint-selection procedure.
- **DynaCPI:** frozen LowRankDynaCPI v5 two-seed ESM-2 protein representation / Morgan fingerprint two-tower model, selected on validation only, not adjusted after held-out exposure.
- **Pair matching:** both ensembles used exactly the same predeclared held-out observed protein-compound pairs, with the same continuous assay target and per-protein macro plus pooled RMSE definitions. Original training's printed `test mse` used a validation-only proxy, not this held-out test.
- **Split provenance:** KIBA DeepDTA source (`a546a8433a6822e958f36171c4356ad6f414d623`) with 2,111 compounds, 229 proteins and 118,254 observed scored pairs; Metz PADME source (`e01c592cc06c4de04b3ed6db35da5af5ff7f863f`) filtered to 1,423 compounds, 170 proteins and 35,259 scored pairs. The original published paper's dataset filters/folds may differ.
- **Sequence grouping:** original frozen protocol uses 3-mer TF-IDF similarity connected groups with threshold 0.35; not guaranteed curated protein-family disjointness. These are independently frozen splits, not claims of reproducing a paper-specific benchmark.
- **Pretraining asymmetry:** DynaCPI has frozen pretrained ESM-2 embeddings, original PerceiverCPI does not. These results establish comparative performance of the **complete systems**, not that every DynaCPI architectural component or 3D geometry is responsible. This v5 does not implement the proposed 3D equivariant or expert-routing extension.
- **Statistical scope:** One test partition per dataset, two trained seeds per model, descriptive cluster bootstrap. No external double-cold or compound-cold conclusion is supported by these results.
- **Prospective scheduling amendment:** Unlike the originally stricter all-six-at-once evaluation gate, these two *completed* frozen scenarios were unblinded once before the other four original model pairs finished training. All six DynaCPI checkpoints and splits were frozen before any external test, and both original checkpoints for each scored scenario completed validation selection before opening its test. The change and its limitations are recorded in `EXTERNAL_INCREMENTAL_ONCE_TEST_PROTOCOL_2026-11-11.md`.
- **Private audit artifacts:** The exact once-only `TEST_RESULTS.json`, original and DynaCPI checkpoint SHA256s, paired test predictions, prediction logs, and `PAIRED_CLUSTER_BOOTSTRAP.json` reside in the gitignored `external_benchmarks_20261110/locked_external_protocol_v1/INDIVIDUAL_EXTERNAL_SAME_SPLIT_TESTS_ONCE_20261111/{kiba,metz}_protein_cold/` directories. Never put raw test labels or private predictions into the public git repository.

## Remaining work

The external original-model training queue continues for **KIBA compound-cold, KIBA double-cold, Metz compound-cold and Metz double-cold**. Those test labels remain sealed. The amended six-scenario gate will import the existing two completed individual tests unchanged (including verified paired bootstrap) and evaluate the four remaining scenarios once their original model pairs become ready. Never use these two held-out scores to alter remaining training/evaluation settings or select a newer DynaCPI checkpoint.

**Permissible claim:** DynaCPI v5 achieved lower same-split held-out protein-cold RMSE than original PerceiverCPI on KIBA and Metz; the within-split paired uncertainty is stronger for KIBA. This does **not** establish superiority under every cold-start condition, original paper folds, equal pretraining budgets or 3D model architectures.
