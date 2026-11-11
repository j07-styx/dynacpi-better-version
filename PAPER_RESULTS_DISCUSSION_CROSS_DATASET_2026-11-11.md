# Results and discussion: frozen DynaCPI v5 versus unmodified PerceiverCPI

**Research manuscript section, 11 November 2026.** This document consolidates the actual matched, frozen, two-seed *held-out* comparisons available at the time of writing. It intentionally omits raw test labels, pair-level predictions, and personal file paths. It does not present an untested aspirational architecture as implemented.

## 1. Compared systems and methodology

The reference system is the original unmodified PerceiverCPI from dmis-lab/PerceiverCPI, pinned source commit 755f076ef51099a90332fd1ec8437f94e3750c0b, run with its original optimizer/training procedure, 30 epochs and fixed model-training seeds 2931 and 2932 for each external split. The original training script was supplied byte-identical validation data for its internal proxy-test input; that output is not an independent held-out test result.

The experimental system is **LowRankDynaCPI v5**, which combines frozen pretrained ESM-2 650M protein representations (1,280 dimensions), 2,048-dimensional Morgan chemical fingerprints, a train-derived top-16 Tanimoto chemical-affinity prior and a learned two-tower residual interaction. Its selected two-seed checkpoints and split assignments were frozen before external test outcomes were opened. **It does not contain a 3D equivariant encoder, support-conditioned experts or learned expert routing.**

Each two-seed ensemble was evaluated against the same held-out compound–protein pair rows for its particular split. Lower **pooled RMSE** indicates smaller error over all observed pairs. Lower **target-macro RMSE** indicates a lower average of protein-wise RMSE values, weighting each scored protein equally. The models have different upstream pretraining budgets (ESM-2 on DynaCPI), so performance differences establish *complete-system* superiority on selected comparisons, not proof of architecture-only novelty or equal-feature-budget efficiency.

## 2. DAVIS: four new frozen held-out split experiments

| Split | Held-out observed pairs | Dyna macro RMSE | Original macro RMSE | Dyna pooled RMSE | Original pooled RMSE | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Strict protein A | 4,488 | **0.579709** | 0.715804 | **0.609198** | 0.743908 | Dyna both metrics |
| Strict protein B | 5,848 | **0.720277** | 0.767670 | **0.773445** | 0.799888 | Dyna both metrics |
| Compound scaffold cold | 5,746 | **0.619061** | 0.670314 | **0.745825** | 0.797344 | Dyna both metrics |
| Double cold | 1,339 | 0.609833 | **0.595813** | **0.680169** | 0.682757 | Mixed: original wins macro |

All four models used frozen validation-selected two-seed checkpoint pairs, with no test-driven checkpoint selection. **DAVIS conclusion:** DynaCPI wins both point-estimate RMSE metrics on **three of four** tested split conditions. The double-cold result does *not* support general dominance.

**Descriptive paired-bootstrap 95% difference intervals (DynaCPI minus original PerceiverCPI, so negative favors DynaCPI):**

| DAVIS split | Macro RMSE difference interval | Pooled RMSE difference interval |
|---|---|---|
| Strict protein A | **[−0.204767, −0.064070]** | **[−0.204730, −0.060108]** |
| Strict protein B | [−0.122655, +0.029656] | [−0.114932, +0.049987] |
| Compound scaffold cold | **[−0.087977, −0.010174]** | [−0.139136, +0.011670] |
| Double cold | [−0.070571, +0.112139] | [−0.080777, +0.109976] |

All intervals reflect the frozen split and resampling protocol, not an independently sampled biological population. Multiple partitions of DAVIS are not four independently acquired datasets.

## 3. KIBA and Metz: completed independent-source held-out comparisons

**KIBA source:** DeepDTA snapshot, 229 proteins, 2,111 source compounds, 118,254 observed interaction scores. The KIBA score is a composite measure; it is not DAVIS pKd. **Metz source:** filtered PADME snapshot with 170 proteins, 1,423 compounds and 35,259 observed numeric-affinity pairs. Its own affinity label convention was preserved. These differ in source filtering and/or splits from some published PerceiverCPI tables; **do not compare RMSE between the two datasets or against nonmatching paper folds.**

| Held-out split | Test pairs | Dyna macro | Original macro | Dyna pooled | Original pooled | Point estimate |
|---|---:|---:|---:|---:|---:|---|
| KIBA protein cold | 20,755 | **0.647370** | 0.753914 | **0.608841** | 0.694440 | Dyna both |
| Metz protein cold | 5,981 | **0.751161** | 0.777072 | **0.735788** | 0.772058 | Dyna both |
| Metz compound cold | 6,117 | **0.748509** | 0.764061 | **0.781367** | 0.804360 | Dyna both |

**KIBA protein cold:** macro RMSE reduction ~14.1%, pooled RMSE reduction ~12.3%. **Metz protein cold:** macro ~3.3%, pooled ~4.7%. **Metz compound cold:** macro ~2.0%, pooled ~2.9%. All percentages are relative to the original on the *same* held-out split, not relative to a different benchmark.

**Descriptive predeclared paired-cluster bootstrap (5,000 resamples; 95% difference intervals):**

| Held-out source and split | Macro RMSE difference interval | Pooled RMSE difference interval | Interpretation |
|---|---|---|---|
| KIBA protein cold | **[−0.164013, −0.051187]** | **[−0.153684, −0.026865]** | Both intervals favor DynaCPI |
| Metz protein cold | [−0.085858, +0.033047] | [−0.085000, +0.016172] | Both intervals include zero |
| Metz compound cold | [−0.051546, +0.024340] | [−0.065268, +0.021794] | Both intervals include zero |

**External conclusion at this snapshot:** DynaCPI v5 lowers both RMSE point estimates on all **three of the six planned** external split conditions completed so far. **Only the KIBA protein-cold split has both descriptive uncertainty intervals wholly favoring DynaCPI.** Results for KIBA compound cold, KIBA double cold and Metz double cold are still sealed/pending; do not infer they will be positive.

## 4. Diagnostic explanation for double-cold fragility

The experimental model's pooled RMSE versus its train-derived chemical-only prior was **0.758663 vs 0.755425 on KIBA double-cold validation**, where the prior is slightly better, but **0.778389 vs 0.803484 on Metz double-cold validation**, where DynaCPI is better. Thus the learned protein/chemical residual model is not uniformly helpful under compound- and protein-cold conditions.

A post-hoc *training/validation-only* stratification by nearest-training-neighbor protein ESM-2 cosine and compound Morgan Tanimoto indicates that the KIBA double-cold pooled deficit is concentrated among less chemically similar validation compounds, while DynaCPI performs slightly better among the most chemically similar validation compounds. These subgroup analyses are exploratory, may be affected by strongly correlated pair counts, and **must not be used to change the checkpoints or selection protocol while the sealed double-cold tests remain unopened**. The granular analysis is documented in KIBA_METZ_DOUBLE_COLD_VAL_NOVELTY_DIAGNOSTIC_2026-11-11.md.

## 5. Reproducibility and limitations

1. Every published comparison must disclose the source snapshot, exact split manifest, same observed pair rows, both seeds, checkpoint hashes, held-out metric definition, interval method and any negative results. Private raw predictions and test labels are stored locally, gitignored; this report contains only aggregates.
2. Protein similarity group splits based on connected 3-mer TF-IDF neighbors (threshold 0.35) are **not certified protein-family partitions**. Murcko scaffold disjointness likewise does not guarantee complete chemical dissimilarity.
3. Results compare complete systems **with different feature pretraining budgets**. Future attribution requires equalized protein representations, controlled protein/compound ablations and fresh, not-yet-used validation/test data.
4. The original six-external-tests-at-once evaluation schedule was prospectively amended to permit one-time evaluation of ready scenarios. All six DynaCPI checkpoints and splits were frozen before the first external test, and each original two-seed pair finished validation selection before its specific test was opened. This differs from the stricter all-at-once blinding promise and must be disclosed, as done in EXTERNAL_INCREMENTAL_ONCE_TEST_PROTOCOL_2026-11-11.md.
5. A single held-out partition and two training seeds per system do not demonstrate robust universal performance; bootstrap uncertainty intervals are descriptive, not a guarantee across different biological families, labs or assay types.
6. Do not describe the proposed 3D geometry/rotation-equivariant architecture or dynamic expert routing as tested features: these were **not present** in the v5 experiments.

## 6. Research claims defensible at this stage

**Supported:** Frozen DynaCPI v5 demonstrates better same-split held-out macro and pooled RMSE point estimates than original PerceiverCPI across three of four new DAVIS split settings and all three completed external-source settings; the evidence is especially encouraging for protein-cold KIBA, where the paired intervals favor DynaCPI for both metrics. Its Metz advantages are smaller and uncertain under cluster bootstrap.

**Not supported:** Universal improvement, same-fold reproduction of original publication tables, architecture-only causal benefit, gains attributable to 3D/protein geometry, all-cold-condition dominance, equal-pretraining comparisons, or a guarantee of superiority on the three untested external splits.

### Verifiable public-method documentation
- DAVIS complete four-split gate: evaluate_all_four_preregistered_davis_once_20261110.py; paired_bootstrap_preregistered_davis_20261110.py.
- KIBA/Metz source/split builder: prepare_external_kiba_metz_frozen_benchmarks_20261110.py.
- Frozen DynaCPI external trainer: train_external_sparse_dynacpi_v5_20261110.py.
- Original PerceiverCPI two-seed training: run_external_original_perceiver_frozen_20261110.ps1.
- Individual once-only external scorer: evaluate_external_ready_scenario_once_20261111.py.
- Predeclared external paired uncertainty: paired_bootstrap_external_kiba_metz_20261110.py.
- Double-cold validation diagnostic: audit_external_double_cold_novelty_validation_20261111.py.
- External single-use aggregate: evaluate_external_kiba_metz_all_six_once_20261110.py.

**Data provenance:** Local protected summary JSON and checkpoint/model/data SHA256 manifests are retained in the gitignored generalization_suite_20261110 and external_benchmarks_20261110/locked_external_protocol_v1 directories. The original source commit is publicly declared; source/cohort-specific hashes are recorded in their frozen audit records.

**Future update:** Once the three remaining external frozen split comparisons finish, append the complete six-scenario result, including losses, mixed metrics, and paired uncertainty. Do not rewrite the current dated findings or adjust remaining frozen experiments based on already viewed test results.
