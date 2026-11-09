# DynaCPI — COMPLETE TECHNICAL HANDOFF TO THE NEXT AI
**Date:** 9 November 2026  
**Project / requested GitHub repo:** dynacpi better version (`j07-styx/dynacpi-better-version`)  
**Status:** ACTIVE RESEARCH ENGINEERING; NO DEMONSTRATED SUPERIORITY TO PUBLISHED PERCEIVERCPI  
**Immediate goal:** Make a credible compound–protein affinity predictor with true protein-specific, chemistry-specific generalization; for the upcoming demo, show real, labeled, reproducible model behavior, not unsupported performance claims.

> **START HERE:** This document records what was implemented, trained, falsified, diagnosed and preserved. Do not declare a "win over PerceiverCPI" based on any numbers below: the published paper uses a different protocol. Do not reuse the repeatedly inspected DEV data to obtain or assert a new test win. Preserve all negative experiments. The earlier BindingDB transfer experiment's provisional scientific cohort has **zero human-approved assays** and has a blocked source-family integrity gate. Do not unfreeze its protected final test or bypass the gate.

## 0. Latest v4.8 read-only diagnostics — loss baseline and protein-selectivity similarity

**9 November 2026. No new training.** See [DIAGNOSTIC_ZERO_RESIDUAL_PROTEIN_LOO_2026-11-09.md](DIAGNOSTIC_ZERO_RESIDUAL_PROTEIN_LOO_2026-11-09.md) and the source script `diagnose_zero_residual_protein_loo_20261109.py`. Original per-fold chemical-only double-cold RMSE reconstructed from saved data/hyperparameters matched exactly all three old study results.

- Anchored neural original **zero-residual** standardized TRAIN loss: **0.79001**, compared to actual anchored mean **0.81132 at epoch 1** and **0.74320 at epoch 5**. Only **5.9% improvement** over zero residual in TRAIN, with one fold not beating zero residual after 5 epochs. This is evidence that the network learns *some* residual structure, not proof of undertraining or an independent generalization gain.
- Original 24-protein **leave-one-protein-out** selectivity, all 54 chemical identities warm: target-centered MSE **0.40770** chemical mean, **0.32134** oracle best SINGLE donor chosen with withheld protein labels (not deployable), **0.63087** raw mean-pooled ESM cosine nearest donor. ESM matched oracle donor **1 of 24**, nearest ESM cos mean **0.96672**. This indicates poor correspondence between raw mean-pooled ESM similarity and affinity selectivity, not proof that any feature fix works.
- The protein-neighbor diagnostic is *not* double-cold, target family-disjoint, independent, or a matched PerceiverCPI benchmark. Predeclare and evaluate better embeddings only on a fresh approved split; do not optimize based on these reused small samples.

## 0.1. Latest v4.7 follow-up — 4-way scaffold-excluded out-of-fold residual training

**Executed 9 November 2026 under deterministic CUDA.** Full report: [EXPERIMENT_OOF_RESIDUAL_V47_2026-11-09.md](EXPERIMENT_OOF_RESIDUAL_V47_2026-11-09.md); code: `run_oof_residual_v47_20261109.py`; tests: `test_oof_residual_v47.py`, `test_oof_results_integrity_v47.py` (**3/3 synthetic tests + aggregate integrity PASS**). The 4-way scaffold-excluded prior was used to generate neural training residual targets, with a matched in-sample prior control, four nested inner protein/chemical folds, same three seeds, fixed 5 epochs and conservative inner-only stacking weights. The old model checkpoints and v4.3 API were not modified.

| Metric (target-macro RMSE ↓) | Chemical-only | Matched in-sample residual | New OOF residual |
|---|---:|---:|---:|
| **Double-cold** (new proteins + chemicals) | **0.608048** | **0.607088** | **0.615673** |
| **Cold-compound** (new chemicals, known proteins) | 0.708662 | 0.700078 | **0.696806** |

The three TRAIN-only prior optimism checks reported in-sample vs scaffold-OOF residual RMSE **0.66695/0.72891, 0.53225/0.61694, 0.76774/0.88147**. This supports an OOF residual-target distribution shift, but fewer OOF fit compounds may contribute too. The primary double-cold outcome worsened; OOF improvement was restricted to cold compounds on previously seen proteins, dominated by one fold. The OOF double-cold paired protein-bootstrap interval for change vs chemical baseline **[−0.007979, +0.025631]** includes zero. The repeated old outer folds, non-family-disjoint protein splits and lack of same-protocol PerceiverCPI preclude scientific claims. Do not tune these folds further. One first implementation failed with an argument-order error before scoring; it is preserved separately local-only. The successful run required deterministic algorithms and did not use prior outer scores as an acceptance gate.

## 0.1. Previous v4.6 continuation (after original handoff snapshot)

**Experiment performed 9 November 2026 and verified on the GPU.** A new nested, training-only test replayed the old three anchored sequence-model seeds on the same 24-protein/54-compound DAVIS TRAIN-only, three-fold scaffold/target identity double-cold splits, formed a per-pair prediction ensemble, and selected chemical-prior stacking weights and similarity gates using independent inner splits WITHIN each outer fit rectangle. Full details, tests, limitations and exact data are in [EXPERIMENT_ENSEMBLE_STACK_SHRINK_2026-11-09.md](EXPERIMENT_ENSEMBLE_STACK_SHRINK_2026-11-09.md).

- Chemical-only baseline **0.608048** double-cold target-macro RMSE.
- Original anchored mean of three seed scores **0.610480**; fresh approximate replay **0.610279**.
- Three-seed prediction-level ensemble **0.607697**.
- Inner-selected stacking (weights 0.75/0.75/1.00; all selected GLOBAL gate) **0.607092**, only **0.000955 (0.16%)** better than chemical-only.
- Predefined fixed 50% chemical/ensemble control **0.605640**, but not selected by inner CV and therefore should NOT be chosen retroactively using the old outer scores.
- Protein-level descriptive paired bootstrap for selected stacking effect **[−0.013246, +0.011519]**; includes zero. Existing outer folds already inspected. Not independent scientific evidence; still **NO demonstrated published PerceiverCPI win**.
- Four synthetic tests and exact local per-pair cache integrity checks passed; one historical CUDA seed replay drifted by about +0.00148 RMSE. Checkpoint files and running v4.3 backend unchanged; the new scripts only generate experimental results.

**Stop optimizing against these same splits.** Require an untouched source-reviewed chemistry-and-protein-family-disjoint cohort and matched published PerceiverCPI comparator before any performance claim.

## 1. Executive snapshot

- Working engineering model: **DynaCPI v4.3**, a chemistry-aware, anchored sequence predictor. Its prediction is a frozen Morgan/Tanimoto-derived chemical-affinity prior **plus** a learnable protein-conditioned neural correction. It is distinct from the earlier unhelpful GVP geometry variant.
- On **8 previously inspected DAVIS engineering DEV protein targets (432 pairs)**: v4.3 anchored neural mean target-macro RMSE **0.71903**, matched repaired controls **0.76113**, and the simple per-compound TRAIN mean **0.70166**. Lower is better; the simple baseline wins.
- Historical training-data 3-fold double-cold (new target identities and new compound clusters): anchored neural **0.61048**, repaired neural **0.64262**, chemical-only profile **0.60805**. Tiny gap **+0.00243** versus chemical-only means there is still **no demonstrated protein-specific superiority**.
- Latest follow-up (v4.4) tested a train-inner-selected centered protein+compound kernel correction. Double-cold RMSE **0.61616** versus matched chemical-only **0.60994**; baseline **0.60805**. This was a **negative experiment**, committed locally as `072d555`. Reversing protein identities raised RMSE to 0.62618, suggesting protein sensitivity without net performance gain.
- Earlier v4.2 repaired sequence encoder/head: **0.75141** against matched original v4 sequence architecture **0.80942** DEV RMSE (7.17% relative better), three seeds, same 5-epoch TRAIN setup. These are engineering gains, **not** better than the compound mean and not a fair paper benchmark.
- Older v4 frozen GVP geometrical branch on the initial 4-protein DAVIS development panel **worsened RMSE** (mean about **0.93791** across 3 seeds) compared with frozen sequence 0.92569. Correct residue alignment and misalignment behaved similarly. Its downstream model almost ignored compound differences.
- Published PerceiverCPI and any future matched candidate should be evaluated on **identical data, label transformation, splits, model selection, metric aggregation, and support labels**. Prior provisional BindingDB matched single-seed 5-epoch comparison had **PerceiverCPI 1.325136 versus TransferCPI 1.508015** target-component macro RMSE, so that older TransferCPI **lost**. These are different data, units/aggregations and training regimes from the DAVIS 0.6–0.9 scores; **never juxtapose them as if comparable**.
- The current presentation/API is on Windows at **http://127.0.0.1:8781**, running Flask v4.3 with a `/api/v1/predict` endpoint, model registry, targets, K-shot calibration and profile/CV metadata. A legacy v4 API also exists at port 8780. The frontend is built elsewhere; do not break it.
- Source code, local checkpoints, training histories and result JSON files exist. No user-facing ML performance guarantee is warranted.

## 2. Locations, accounts and execution environment

| Purpose | Current machine location |
|---|---|
| Main inherited project (upstream PerceiverCPI + DynaCPI v4) | `C:\Users\J0124\dynacpi_v4` |
| Isolated current research + trained candidates + API | `C:\Users\J0124\DynaCPI-Backend` |
| Virtual environment Python | `C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe` |
| Historical DAVIS pilot labels | `dynacpi_v4\data\engineering_pilot\rows.json` |
| All compound SMILES and Morgan bit features | `dynacpi_v4\data\staging\compounds.json` and `morgan_radius2_2048_chiral.npz` |
| ESM-2 residue embeddings | `dynacpi_v4\data\esm2_t33_650M_layer33_w1000_o128_fp16` |
| Limited 3D AlphaFold-derived graph maps | `dynacpi_v4\data\engineering_pilot\structure_graphs` |
| Legacy frozen sequence model | `dynacpi_v4\research\pilot_sequence\checkpoint.pt` |
| New v4.2 trained and matched models | `DynaCPI-Backend\checkpoints` |
| New v4.3 anchored and kernel-profile checkpoints | `DynaCPI-Backend\interaction_study_20261109` |
| New v4.4 negative control | `DynaCPI-Backend\interaction_study_20261109\conditional_kernel_20261109` |
| Backend code | `DynaCPI-Backend\app.py` (v4.3, port 8781) |
| Former source/research scripts | `dynacpi_v4\dynacpi`, `dynacpi_v4\scripts`, `dynacpi_v4\tests_v4` |
| New repo source snapshot | `legacy_v4_snapshot` inside this repo |
| GitHub username | `j07-styx` |
| Original legacy repo (DO NOT PUSH HERE FOR THIS TASK) | `https://github.com/j07-styx/Demo-smth`, branch `dynacpi-v4` |
| New requested repository | `https://github.com/j07-styx/dynacpi-better-version` |

Hardware: Windows 10/11 with NVIDIA RTX 4060 Laptop GPU ~8 GiB, CUDA; a more powerful GPU might be available later, but results below were produced on this PC. Git credentials use Windows Git Credential Manager. The original Chemprop MPN has CUDA-specific behavior. Scripts may use the shared `dynacpi_v4\.gpu.lock` to avoid overlapping GPU training and inference. The current code hardcodes the local original root; moving the repo to a new machine **requires refactoring paths, supplying compatible ESM caches and installing the upstream dependencies**. Do not misrepresent this GitHub repo as fully standalone.

Git branch on the new local repo: `main`; the original old repo branch is `dynacpi-v4`. Legacy upstream source `https://github.com/dmis-lab/PerceiverCPI` is MIT licensed; the original `LICENSE.txt` is retained. This GitHub repo is **PUBLIC** as requested. Only source code, documentation, tests, and non-sensitive aggregate experiment results are published. Raw datasets, protected source-review records, checkpoints, and learned kernel coefficient binaries remain local.

## 3. Scientific target and critical data definitions

Target: real-valued compound–protein binding affinity prediction. Input is **a compound SMILES string and a protein**; output is predicted binding affinity expressed as pKd where appropriate. For exact dissociation constants: `pKd = 9 - log10(Kd in nM)`. A DAVIS experimental screening floor around pKd 5 is frequent. These screen-floor values are **not automatically exact physical Kd measurements**; avoid assuming that all floor entries are identical true affinities.

Architecture originally contemplated: learn compound embeddings from 2D molecular graphs and chirality-aware Morgan fingerprints, protein embeddings from ESM-2-650M sequence residues, optionally 3D protein graphs via GVP; combine with compound-conditioned interaction and support-conditioned calibration. Geometry is a protein **3D feature**; "sequence-only" means protein sequence/ESM (1D input) **plus the compound's 2D graph/fingerprint**, not that the compound is one-dimensional.

Data regimes MUST be kept separate:

**A. DAVIS engineering pilot used by new backend experiments**

- Full engineering matrix: **24 TRAIN proteins × 54 known compounds = 1,296 TRAIN pairs**, and **8 previously inspected DEV proteins × 54 identical compounds = 432 DEV pairs**. Total 32 proteins / 1,728 pairs. Protein identities differ between TRAIN and DEV; **all 54 compounds are shared**, so an unseen-protein score is **not** an unseen-compound score.
- Early limited 3D graph pilot: 9 graph-ready training proteins and four graph-ready development proteins, 54 compounds per protein, i.e. 486 train vs 216 dev pair subsets for geometry studies. Three of the older 3D pilot seed runs used more restrictive 7/2 graph fitting/inner training-only selection. The frozen sequence model was earlier fit on all 24 TRAIN proteins; prior geometry inner splits are not independent tests of that foundation.
- Full precomputed frozen ESM2: 33 layers, 650M-parameter model, window 1000 overlap 128, 1280-dimensional final residue features. Cached files have sequence hashes.
- Chiral Morgan 2048-bit fingerprints radius 2; 2D graph via Chemprop MPN / RDKit.
- Some DEV proteins are mapped to exact AlphaFold v6 structures; the serving API supports **ADCK4, FRK, MEK5, OSR1** as four known pilot targets, not all eight DEV targets.
- The eight-target DEV is repeatedly consulted by the project historically; treat all analyses there as engineering diagnostics, **not scientific independent validation**.

**B. DAVIS training-data double-cold cross-validation**

- The latest `run_profile_study.py` excludes all original DAVIS DEV labels; it runs **three paired outer folds** over only the 24×54 TRAIN rectangle.
- Per outer fold: 16 protein identities and 36 compounds in fit; other proteins or compounds appear in cold-protein, cold-compound or double-cold evaluation cells.
- Chemical grouping: connected components of same Murcko scaffold OR Morgan/Tanimoto >=0.5; train–heldout compound max similarity per fold **0.4831, 0.4831, 0.4444** (all <0.5).
- Protein **identities** are disjoint but not guaranteed protein family/homology disjoint. Model-selection/hyperparameter choices use nested training-only inner partitions, but there are only 24 proteins, 54 compounds, and **these outer folds have now been inspected repeatedly**. Not an independent prospective test.
- Current benchmark target to beat **within this internal experimental setting**: chemical-only profile **0.608048 target-macro RMSE on double-cold**. This is NOT the published PerceiverCPI score.

**C. Separate provisional BindingDB / TransferCPI research**

- Much larger provisional technically split BindingDB experiment: 82 training protein components / 27,118 pairs; 20 inner-dev components / 7,387 pairs; 33 protected prospective-test components / 2,936 pairs. Older dev quarantined.
- This split is **provisional and not approved**. A publication/source-family audit found **seven related source families crossing roles**. Existing training scripts were intentionally locked before further execution. Do not unlock by editing a status flag or cherry-picking a working directory.
- 30 source-review tasks / 921 source records had technically consistent joins and Kd→pKd conversions but remained **0/30 scientifically approved, 921/921 unreviewed**. A whole-cohort technical screen referenced 1,146 candidate assay tasks and 142 source families; those are NOT source-validated experimental cohorts.
- Formal gate status: `BLOCKED_UNAPPROVED_NOT_TRAINING_READY`. Human reviewers must verify original publications, physical protein constructs, assay methods, qualifiers, endpoint identity and source record correspondence. Never auto-approve, infer source evidence from metadata alone, or score the protected test.

## 4. Models and architecture chronology

### v4.0: Frozen sequence foundation

Implementation: `legacy_v4_snapshot/dynacpi/sequence_model.py`. Internals:
1. compound SMILES → Chemprop MPN raw graph vector (256), with an early per-molecule min–max transform.
2. chirality-aware Morgan fingerprint (2048 bits) → linear fingerprint projection (256).
3. custom channel-attention treats fingerprint projection as QUERY and graph vector as keys/values; `compound = ChannelAttention(graph+ChannelAttention(fingerprint,graph,graph), same, same)`. Neither original channel-attention path directly preserves the fingerprint features into the final compound embedding.
4. protein ESM2 [L,1280] → sequence projection to 256-dimensional residue tokens.
5. compound-query multi-head attention across all residues plus whole-protein mean. A later output attention/head predicts standardized scalar affinity.
6. Standardized outputs rescaled using **TRAIN-only** label mean/std.

Critical diagnosis: original checkpoint had only **63 optimizer updates** before supervised freezing, far too little evidence of reliable chemical discrimination. Custom scalar-channel logits were divided by 16; entropy gave **~99.999% effective channel coverage**, nearly uniform attention. Changing scale alone with same trained weights **did not fix** discrimination.

Input Morgan diversity was real: mean pairwise Tanimoto **0.1367** among 54 example compounds; 91% below 0.2. Yet final compound features averaged cosine **0.999885**, with effective variation rank approximately **one**. Feature-direction cosine along pathway: learned fingerprint projection **0.450239**, raw MPN **0.994760**, first channel attention **0.999872**, final attention **0.999885**. Frozen baseline predicted nearly constant pKd for different compounds of the same target. ADCK4 predicted SD was ~0.00131 versus measured SD ~0.568; FRK predicted SD ~1.35×10^-5 versus recorded SD ~1.128. Singleton/batch inference consistency was ~10^-7 pKd, not the cause of collapse.

### v4.0 geometry and few-shot hypotheses: negative/ambiguous

GVP encoder in `legacy_v4_snapshot/dynacpi/geometry_branch.py`, real AlphaFold graph maps with per-residue vector/scalar data. The geometry branch augmented protein sequence tokens before frozen downstream attention. The original design also included the K-shot same-target support adaptation (K=0,1,2,5,10,20) comparing:
- A zero-shot mixture;
- B support-calibrated mixture;
- C differential-only;
- D full adapted = B plus geometry differential.

Actual D-vs-B geometry differential corrections were **essentially zero** (K=5 changes ~10^-11 to 10^-9 in RMSE); support ridge and tiny sequence/geometry disagreement variance made the differential term unidentifiable. Small K=5 geometry-mixture effects varied by seed and target, not a win. A non-geometric residual sometimes did as well or better.

Older geometry studies:
- centered GVP (3 seeds): mean DEV 0.925685342 versus frozen 0.925685078, effectively unchanged.
- bounded fusion 2026-11-07: sequence residual **0.920612**, correctly mapped geometry **0.921045**, shuffled geometry **0.921663**, frozen foundation **0.925685**; cannot isolate a clear benefit of correct 3D residue alignment.
- real GVP training-only selection on the 7/2 graph-ready training subset (seeds 1901/1902/1903) scored **0.936573 / 0.939935 / 0.937222**, all worse than frozen 0.925685. Corresponding misaligned controls **0.934605 / 0.937851 / 0.936788**, also worse and often slightly better than real alignment.
- One seed's 4-target detailed result: FRK 1.2854→1.2836, MEK5 1.3789→1.3774, ADCK4 0.6177→0.6644 and OSR1 0.4208→0.4209.
- ADCK4 geometry gave an almost constant +0.1019 pKd shift across 54 ligands, and a same-trained-weights residue-map shuffle affected predictions only ~0.000619 pKd; mean-pool and compound-conditioned attention each contributed, but both were almost ligand independent. Readout was estimated **41–77× more sensitive** to uniform protein token shifts on ADCK4 than other proteins. ADCK4 foundation already mean-overpredicted by +0.243 pKd.
- Conclusion: geometry behaved like a protein-wide offset added to a weak and chemically collapsed predictor; stronger geometry alone was not the missing ingredient.

### v4.1 causal repair and microfit diagnosis

Source: `causal_diagnosis_20261109/causal_diagnosis.py`, `repair_holdout.py`, report, and JSONs.

Counterfactual 16 TRAIN-pair, 200-step fitting tests (TRAIN RMSE only, **memorization**, not generalization):
| Architecture | Training RMSE |
|---|---:|
| Original architecture fully unfrozen | 0.2602 |
| Compound encoder bypass only | 0.0979 |
| Interaction readout bypass only | 0.1008 |
| BOTH bypasses | **0.0013** |

Architecture **can** learn, but the inherited information path/short early training made it difficult. Original architecture was not mathematically incapable of chemistry.

Inner fixed 18 TRAIN target fit / 6 other TRAIN target validation, same 54 compounds, three seeds, 20 epochs, checkpoints selected internally:
| Model | 3-seed mean inner-validation RMSE |
|---|---:|
| Original | 0.8644 |
| Compound-average over fit proteins | 0.8166 |
| BOTH bypasses repaired | **0.8076** |

Two-path repair won against original in three seeds, but compound average was closely competitive; original 24 TRAIN targets had been explored historically, so not independent science. Compound-identity shuffle raised repaired RMSE from 0.8076 to 1.0411, supporting genuine chemical input use.

### v4.2 trained repaired SEQUENCE model

Implementation: `repaired_model.py`.

```python
compound = LayerNorm(fingerprint_projection + raw_mpn_graph_vector)
protein = LayerNorm(mean(residue_tokens) + compound_conditioned_residue_attention)
prediction = MLP(concat(compound, protein, compound * protein, abs(compound - protein)))
```

No GVP geometry in this trained candidate. Two direct feature bypasses; removes early per-molecule MPN min–max scaling and the original two scalar-channel gates. Unfreezes trainable molecular/sequence/readout pathways together. The previously cached ESM2 raw residue representations remain frozen. Repaired model adds ~263k parameters (1.912M vs original 1.649M), so **not capacity matched**.

Full TRAIN protocol: all 24 proteins/1,296 examples, 5 epochs, seeds 2831,2832,2833, AdamW lr=3e-4/wd=1e-4; 8 old DEV scored only at the end, never to select epochs. Fixed epoch 5 chosen from prior inner TRAIN splits.

| Seed | Matched original DEV RMSE | Repaired DEV RMSE | Reduction |
|---|---:|---:|---:|
| 2831 | 0.808063 | 0.763457 | 5.52% |
| 2832 | 0.810654 | 0.750148 | 7.46% |
| 2833 | 0.809548 | 0.740614 | 8.52% |
| **Mean** | **0.809422** | **0.751406** | **7.17%** |

Mean pairwise learned molecule cosine: original 0.999990/0.999643/0.997779; repaired **0.36794/0.42720/0.40454**. Chemical representations became distinguishable. But **TRAIN compound-mean prior 0.701664 RMSE** was still better than 0.751406 on DEV. Generalization did not prove strong protein specificity.

Artifacts: `checkpoints/SUMMARY.json`, `MATCHED_COMPARISON.json`, `MATCHED_SIMPLE_BASELINES.json`, per-seed `COMPLETE.json`, `history.json`, and binary `checkpoint.pt`; tests `test_model_parity.py` and `test_api.py` passed. Prior 18/6 scripts and reports remain in `causal_diagnosis_20261109`.

### v4.3 chemical affinity profiles and anchored neural predictor

Source: `profile_model.py`, `run_profile_study.py`, `train_anchored_full.py`, `profile_serving.py`, `diagnose_interactions.py`, and `interaction_study_20261109` files.

`AffinityProfile` fits regularized Tanimoto chemical affinity prior, optional ESM protein offset, optional centered ESM×compound kernel ridge interaction. Its three explicit inference modes are:
- `profile_chemical` = **protein-independent** chemical-only control;
- `profile_additive` = chemical prior plus protein-only offset;
- `profile_interaction` = chemical prior plus protein×compound kernel residual.

Neural anchored model is `profile_chemical(Morgan) + sigma_train * repaired_neural_residual(SMILES, ESM2)`, where the chemical prior is frozen after TRAIN-only fitting; output residual head initialized to 0 to start exactly at chemical baseline. The matched repaired control uses same repaired neural architecture and training budget but global training mean rather than a compound-conditioned prior. This initial floor helps stabilize training and allows the model to focus on target-dependent chemistry residuals. It is **sequence-only**, not a 3D geometry result.

Nested train-only cross-validation on 24×54 matrix, 3 outer paired folds; chemical ridge choices and neural prior selected internally; 5 epochs, 3 seeds 2931–2933. Model scores are target-macro RMSE:

| Model family | Cold protein | Cold compound | DOUBLE COLD |
|---|---:|---:|---:|
| Global fit mean | 0.735693 | 0.721294 | 0.620948 |
| Chemical-only profile | 0.675726 | 0.708662 | **0.608048** |
| Chemical+protein additive | 0.674123 | 0.685417 | 0.615880 |
| Chemical+protein×compound kernel interaction | 0.692399 | **0.671631** | 0.628385 |
| Repaired sequence neural (3 seeds) | **0.658610** | 0.694822 | 0.642617 |
| Anchored sequence neural (3 seeds) | **0.657944** | 0.698930 | **0.610480** |

CAUTION: This table contains multiple model families, selection procedures, and 3-seed averages; it is exploratory, not independent, not family-level disjoint, and the three folds are not independent datasets. Compared to chemical-only, anchored is marginally worse double cold by +0.002432. It beats repaired neural by ~5% but does not establish a useful protein-specific advantage.

Full 24-protein TRAIN fixed-5-epoch v4.3 checkpoint/DEV diagnostic with NEW seeds 2931–2933:

| Seed | Matched repaired DEV RMSE | Anchored DEV RMSE |
|---|---:|---:|
| 2931 | 0.799359 | 0.713819 |
| 2932 | 0.739259 | 0.699621 |
| 2933 | 0.744769 | 0.743552 |
| **Mean** | **0.761129** | **0.719031** |

Anchored wins in all three, averaging ~5.53% reduction against fresh repaired control. **Simple per-compound TRAIN mean is 0.701664; chemical-only kernel on same DEV ~0.704729.** Neither neural model is a confirmed independent improvement.

IMPORTANT nuance: best single anchored checkpoint seed2932 gives 0.699621 on old DEV, numerically below 0.701664, but choosing a winning seed after inspecting results is **selection bias**; do not claim this seed beat the chemical baseline scientifically. Fixed-5-epoch, 3-seed mean 0.719031 loses.

API v4.3 was tested live and passed: all chemical-profile candidates, all anchored checkpoints, output component decomposition (chemical prior + residual), K=5 calibration, canonical SMILES/Morgan parity for all 54 training compounds, invalid SMILES rejection, old v4 API compatibility, CORS, inner-CV metadata. A saved report is `interaction_study_20261109/API_VALIDATION.json`. Source files and weights are saved, not merged into the old Demo-smth Git repo.

### v4.4 follow-up: protein-conditioned kernel shrinkage FAILED

Source: `test_conditional_residual_20261109.py`; artifacts `interaction_study_20261109/conditional_kernel_20261109/`. Local source/results commit `072d555`.

Hypothesis: a protein/compound kernel residual could improve over the chemical-only prior with limited data. Preselected candidate settings included protein feature choices raw ESM / train-centered ESM / amino-acid 3-mers, chemical Tanimoto or Tanimoto-squared, residual ridge 0.1/1/10/100 and correction strength alpha 0/0.25/0.5/1.0. Chemical-prior ridge was selected separately on **inner double-cold only**, and the winning combo per outer fold was chosen before evaluating that fold.

| Condition | 3-fold mean DOUBLE-COLD RMSE |
|---|---:|
| Previously established chemical-only profile | **0.608048** |
| New chemical-only control (independent inner ridge) | 0.609942 |
| New protein-conditioned correction | **0.616156** |
| Same model with protein identities swapped | 0.626179 |

Protein residual helped fold 0 (0.565707→0.553487), essentially tied fold1 (0.756796→0.757858), hurt fold2 (0.507324→0.537125). Mean absolute protein-specific correction about **0.05174 pKd**. Genuine use of protein input is observed, but no overall positive predictive value versus chemistry-only. **Do NOT promote v4.4 to live inference**, and do not tune again on these outer folds.

### Separate older PerceiverCPI / TransferCPI matched provisional study

A separate provisional BindingDB full-data project built `TransferCPI`, ESM-aware cross-attention and ChemBERTa variants inspired by GenSPARC and others, plus a reproduced PerceiverCPI. This is not the v4.3 DAVIS model and has its own locked source-family split. In five-epoch **single-seed** exploratory inner DEV:
- Original PerceiverCPI: **1.325136** component-macro RMSE.
- TransferCPI: **1.508015**.
- TransferCPI chemical-only: **1.560025**.
- TransferCPI + ChemBERTa: **1.600903**.
- Dual chemical-only: **1.620020**.
- Train-mean predictor had lower pooled RMSE (1.803902) than any neural arm, but do not conflate pooled and macro aggregates.
- Result: **NO gain over PerceiverCPI**. Experimental transfer training later LOCKED because publication-source-family overlaps across provisional train/dev/protected-test. No prospective-test metric exists.

See copied `legacy_v4_snapshot/research_notes/HANDOFF_TRANSFER_CPI_20261108.md` and `TRANSFER_CPI_MATCHED_DEV_DECISION_20261108.md`.

## 5. Benchmark interpretation (NEVER mix scopes)

- **0.608048**: best chemical-only comparator on a small previously inspected **historical DAVIS TRAIN-only double-cold** 3-fold protocol. It is an **engineering target**, NOT a PerceiverCPI paper metric.
- **0.610480**: v4.3 anchored neural on exact above double-cold protocol (near tie/slightly worse).
- **0.616156**: latest v4.4 conditional-kernel ablation (worse).
- **0.719031**: v4.3 anchored on **8 old DAVIS DEV** targets, same 54 known compounds present in TRAIN.
- **0.701664**: simple per-compound mean on that 8-target DEV panel (stronger).
- **0.751406**: v4.2 repaired DEV mean; compare only same old DEV protocol.
- **0.925685**: earlier frozen foundation on **4 geometry-ready DAVIS DEV** targets, not eight; direct 0.925 vs 0.719 comparisons are **NOT a strictly matched split/model-budget evaluation**.
- **1.325136 vs 1.508015**: original PerceiverCPI vs TransferCPI on **different provisional BindingDB component DEV**, not comparable to DAVIS results; prior Perceiver wins there.
- The old full published PerceiverCPI paper (Bioinformatics 2023, paper `btac731`) uses historical DAVIS/KIBA/METZ evaluations with specific transforms/splits; **no matched v4.3 versus actual PerceiverCPI run on the above double-cold protocol has been completed**.
- Macro RMSE across proteins and pooled RMSE across all pairs are different. Report number of proteins, compounds, assays, seeds, folds, label scale and whether support labels were available. Do not silently "improve" by comparing different endpoints, scales, source families, K-shot budgets or tensor preprocessing.

## 6. What is currently functional and how to run it

Original environment is Windows + RDKit + PyTorch + Chemprop legacy MPN + frozen ESM feature cache + Flask.

Run the currently working v4.3 backend from PowerShell:

```powershell
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' 'C:\Users\J0124\DynaCPI-Backend\app.py'
```

By default it binds localhost:8781, not public internet. HTTP endpoints:
- `GET /api/v1/health` — backend version, currently v4.3.
- `GET /api/v1/models` — 14 model registrations (legacy frozen seq/geometry, 3 repaired, 3 anchored, 3 chemical/additive/interaction profiles; exact count subject to saved checkpoint availability); `sequence_baseline` remains default.
- `GET /api/v1/targets` — 4 local DAVIS DEV targets: FRK, MEK5, ADCK4, OSR1.
- `GET /api/v1/training`, `/api/v1/training/repaired`, `/api/v1/training/profile` — metadata and internal-CV comparisons.
- `POST /api/v1/predict` — canonicalize SMILES, select target/model/K, make frozen model estimate and optional K-shot same-target support calibration. Support observations are from the known DEV demo pool; query compound identity is excluded from support. The response does not disclose support labels. Performance is a model estimate, not actual experimentally verified affinity.

Example:
```json
{
  "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
  "target": "FRK",
  "support_k": 5,
  "model_id": "anchored_sequence_2932"
}
```

Model registry IDs:
- `sequence_baseline` (safe historical default, weak model);
- `repaired_sequence_2831`, `repaired_sequence_2832`, `repaired_sequence_2833`;
- `profile_chemical`, `profile_additive`, `profile_interaction`;
- `anchored_sequence_2931`, `anchored_sequence_2932`, `anchored_sequence_2933`;
- historical `legacy_geometry_1107` and `trainonly_geometry_1901/1902/1903` (negative/legacy experiments).

New model serves two explicit output components: `chemical_prior_pkd`, `residual_correction_pkd`; `nearest_training_compound_tanimoto` is presented as *similarity, not confidence*. Anchor is **not** 3D, and `geometry_available` should be false.

The v4.3 API was last verified alive via `http://127.0.0.1:8781/api/v1/health`; previously checked K=5 `anchored_sequence_2932` returned finite D_full_adapted_pkd. It uses a lock to serialize GPU model loading and share `dynacpi_v4\.gpu.lock`. More than one Flask development backend may be running on the PC (legacy 8780, current 8781, and a separate old 8000/demo), so confirm ports/PIDs before terminating anything. Never kill unrelated training, browser or frontend processes.

Verification commands:
```powershell
$py = 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
$dir = 'C:\Users\J0124\DynaCPI-Backend'
& $py "$dir\test_model_parity.py"
& $py "$dir\test_profile_model.py"
& $py "$dir\test_profile_api.py"
& $py "$dir\test_api.py"
```

Results: original model parity and chemical identity passed, 4 AffinityProfile unit tests passed, profile/anchored API integration passed, older v4.2 HTTP suite passed. Do not assume these tests validate biological prediction generalization.

The training scripts `train_repaired_full.py`, `train_original_matched.py`, `run_profile_study.py`, `train_anchored_full.py` and `test_conditional_residual_20261109.py` save/check fingerprints, split identities, checkpoint SHA256 and selection rules. They **refuse to overwrite** completed result files: rerun only in a NEW experiment directory while preserving previously saved results.

## 7. Artifact tree and provenance

Top level of this repo:
- `HANDOFF_COMPLETE_2026-11-09.md` (this document).
- `README.md` and `EXPERIMENT_REPORT_2026-11-09.md` — first experiment-facing summary. The present handoff supersedes any earlier "best" narrative.
- `app.py`, `repaired_model.py`, `profile_model.py`, `profile_serving.py` — serving/model code.
- `train_repaired_full.py`, `train_original_matched.py`, `run_profile_study.py`, `train_anchored_full.py`, `test_conditional_residual_20261109.py` — fitting/paired evaluation.
- `test_model_parity.py`, `test_api.py`, `test_profile_model.py`, `test_profile_api.py` — parity/unit/HTTP tests.
- `causal_diagnosis_20261109/` — original 16-pair microfit / 18×6 holdout causal evidence. It contains original source and full JSON results.
- `checkpoints/` — v4.2 original and repaired 3×3 checkpoints/metadata plus PROTOCOL/SUMMARY and matched controls.
- `interaction_study_20261109/` — kernel profiles, anchored versus matched v4.3 checkpoints, nested cross-validation outputs and HTTP test record.
- `interaction_study_20261109/conditional_kernel_20261109/` — v4.4 negative model, nested CV results.
- `legacy_v4_snapshot/dynacpi/` — original v4 research model modules from the old checkout.
- `legacy_v4_snapshot/scripts/` — old research/evaluation scripts, copied as source only.
- `legacy_v4_snapshot/tests_v4/` — old testing scripts, copied as source only.
- `legacy_v4_snapshot/project.md`, `research_notes/`, `LICENSE.txt` — additional historical handoffs, source review constraints, upstream MIT license.
- `docs/legacy_checkpoint_manifest.md` and optional exported baseline/geometry checkpoint snapshots — consult if bundled.

All data-derived checkpoint binaries and fitted kernel coefficient `.npz` files are **excluded from the public GitHub repository** and remain local. It does **not** include the massive ESM-2 caches, raw BindingDB tables, protected prospective test outcomes, private review worksheets, or original DAVIS raw data. The old local checkout remains required for exact reproducibility. Binary checkpoint files can contain learned information about historical data and should be treated as research artifacts; the GitHub repository should remain private.

Local first new-repo commit: `5caed82` (2026-11-09 v4.2 two-path repair and API). Second: `072d555` (2026-11-09 v4.4 negative double-cold kernel experiment). The new v4.3 source/results were initially uncommitted in the local new repo; the latest handoff commit should include them. Refer to `git log` and the GitHub remote for actual final SHA and whether a successful push occurred; do not assume from this document alone.

Legacy original repository `Demo-smth` is **not** the target of the current push. Its last seen old-root workbench commit `57b2a9d` has its own dependencies and is not overwritten.

## 8. Research literature and proposed extensions (not present experimental claims)

- Original PerceiverCPI (2023): nested graph/sequence attention; `https://academic.oup.com/bioinformatics/article/39/1/btac731/6842322`. A new result should compare directly to its implementation under a matched dataset and well-defined metric.
- GVP-GNN (2021): geometric protein modeling; `https://mlanthology.org/iclr/2021/jing2021iclr-learning/`. A GVP layer does not by itself encode ligand binding poses.
- GenSPARC: compound multimodal encoders with structure-aware protein representations/attention; `https://www.nature.com/articles/s42004-025-01844-0`. The older TransferCPI was *inspired by*, not a GenSPARC reproduction.
- EquiCPI (2025): protein/ligand 3D equivariant modeling; `https://pubmed.ncbi.nlm.nih.gov/40600339/`. Its ligand-context modeling is fundamentally different from our global protein-graph residual.
- Attention Is All You Need (2017): appropriate query–key scaling follows actual query–key dot-product dimensionality; this helps diagnose the scalar-channel design but scale-only edits failed in our trained checkpoint.
- Screening and split leakage research motivates assay and protein-family/source-family controls. Do not infer that any external paper independently proves DynaCPI's code-level problem.

Promising future (untested) ideas: properly trained fingerprint-preserving 2D encoder with direct compound–protein readout; target-centered auxiliary loss to discourage protein-wide bias; stronger compound-geometry interaction using true ligand–protein structural context; calibrated support adaptation with equal support labels on a frozen, strong PerceiverCPI reference; protein/chemically disjoint source-reviewed generalization.

Do NOT revive "four experts" support-conditioned routing, large GVP sweeps or equivariant complexity without a meaningful ablation over a strong repaired sequence and chemical baseline. Routing was discussed as a possible DynaCPI/DynaCPI Demo extension but **no verified improvement** exists. The working v4.3 is a simple sequence+chemical prior, not an implemented 4-expert router.

## 9. Highest-priority next steps for the next AI

1. **Verify current GitHub state** and rerun integrity/API tests. Do not tamper with the prior `Demo-smth` repository, old running services, or protected dataset.
2. **Validate the handoff + checkpoint hashes** (SHA256 and saved immutable protocols). Check how many checkpoints are present in Git vs only on the original PC.
3. **Stop model shopping against the 3 old outer folds or 8 old DEV proteins.** Those sets have been repeatedly inspected. Avoid any model-selection-by-best-seed on the old DEV.
4. **Create a genuinely new predeclared evaluation** with independent protein-homology family components and chemical scaffold clusters; obtain original measurement-source/construct approvals first. Maintain separate support/calibration data and physically meaningful label transformations. If only DAVIS pilot data exists, label all results exploratory.
5. **Establish strong controls**: training global mean, known-chemical train mean where applicable, 2D fingerprint-only KRR, protein-only, additive protein+compound kernel, plain repaired neural, anchored residual, published PerceiverCPI rerun with same splits, and optionally an equal support-label PerceiverCPI calibrator.
6. **Require evidence that protein features add predictive value** beyond the chemical-only reference: report target-centered metrics, per-target harms, protein feature permutation, strong chemical baselines, different protein families, and data confidence intervals; don't just increase prediction variance.
7. **Only reintroduce geometry when validated on a strong foundation**, with real vs shuffled graph, node/edge corruption, matched capacity, genuine compound-dependent effect, and structure/residue/physical construct provenance. Existing GVP results are predominantly protein-level shifts.
8. **Do not claim baseline superiority until PerceiverCPI has actually been rerun on exactly the same approved split**, with identical labels, metrics and support budget, three seeds, blinded test and component-level uncertainty.
9. **For the upcoming presentation** choose a frozen, functioning model ID such as `anchored_sequence_2932` or the averaged profile comparator, show input → feature extraction → model estimate → optional K-shot calibration, cite its experimental nature, and show a simple baseline beside it. Do not call a score "published" if it is an internal development run.
10. **Keep full provenance in Git**: dates, version, experiments, failures, train/dev/test identities, hashes, checkpoint SHA, model/feature/input contract, fitting epochs and source references. Document all new code/research changes in a new commit; avoid empty version-number-only commits.

A winning model is not a requirement for a useful handoff. The important achievement is a falsifiable progression from a nearly compound-blind foundation to a chemically responsive sequence model, an anchored predictor close to a strong chemical baseline, and a correctly documented negative protein-residual trial.

## 10. Direct prompt for the next AI

> You are continuing the DynaCPI CPI affinity research. Start by reading HANDOFF_COMPLETE_2026-11-09.md, README.md, EXPERIMENT_REPORT_2026-11-09.md, all saved protocols/results in checkpoints and interaction_study_20261109, and the legacy_v4_snapshot source. Check existing code and Git history before changing anything. The original double-cold chemical-only comparator is 0.608048; v4.3 anchored 0.610480; negative v4.4 protein kernel 0.616156; NEW exploratory three-seed prediction ensemble 0.607697 and inner-selected chemical stacking 0.607092 (paired interval includes zero, repeatedly inspected folds). The old eight-target DAVIS DEV is not independent; old provisional BindingDB test is blocked by source-family/assay approval and must remain unopened. Do not claim a victory over PerceiverCPI until both are rerun on a new approved identical split. Focus on meaningful measured protein-specific generalization, matched controls, source/family leakage prevention, new data and reproducibility. Preserve all checkpoints, old running API and old Demo-smth repository, report exact outcomes including failures, make substantive code changes, and commit to the NEW dynacpi-better-version repo only.

---

**This report is a 2026-11-09 engineering handoff, not a paper-ready external validation.** All reported numbers and paths above are linked to local or copied model/train experiment files; consult the exact JSON artifacts before calculating further aggregate statistics. The PUBLIC GitHub repo contains a transferable **source and result snapshot**, not full protected research datasets, model checkpoint binaries, or an end-to-end independent scientific benchmark.
