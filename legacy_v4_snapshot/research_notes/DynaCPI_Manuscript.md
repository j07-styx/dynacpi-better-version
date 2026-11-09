# DynaCPI separates affinity calibration from support guided adaptation of protein geometry

Research manuscript draft version 0.1  
5 November 2026  
Author names and affiliations to be supplied by the research team

## Abstract

Compound–protein affinity prediction must generalize across protein targets, chemical series and experimental contexts. PerceiverCPI combines molecular graphs, fingerprints and protein sequence representations through nested attention, but its published evaluation does not establish whether a small target-specific support panel can identify the incremental value of protein geometry. We propose DynaCPI, a staged architecture comprising a frozen sequence foundation, a geometric vector perceptron branch and an explicit local support adapter. Centered predictor disagreement separates a shared calibration offset from changes in the contribution of geometry. This separation makes the incremental mechanism identifiable only when the support panel probes variation between the two predictors. We derive a constrained ridge solution and introduce a bounded query correction. The proposed evaluation compares geometry adaptation against an identical calibration-only control under protein-group holdout. Preliminary results concern implementation and data integrity: an audit of local legacy fold-zero files found 409 and 127 compound–sequence pairs shared between training and validation in DAVIS and KIBA, respectively, while exact train–test compound and sequence intersections were zero. These are observations about local files, not findings about the authors’ complete original datasets. Two hundred synthetic optimization checks support the adapter derivation. The pinned PerceiverCPI interaction block also passed a synthetic GPU batch-consistency check. No DynaCPI affinity benchmark has been completed; predictive improvement, robustness and novelty remain to be established.

## 1 Introduction

Predicting affinity for an unmeasured compound against a target is a regression problem whose practical difficulty depends on more than architecture. Protein identity, chemical novelty, measurement qualifiers and assay context determine which generalization claim an experiment can support. A low error under random pair separation does not establish generalization to an unfamiliar protein family or a new chemical series. A model that consumes measured support affinities must also be compared with baselines granted equivalent experimental information.

PerceiverCPI [1] combines D-MPNN molecular features, Morgan fingerprints and protein sequence features through sequential attention. Its strongest reported gains concern several novel-pair and novel-compound comparisons; the paper qualifies its novel-protein results and recognizes poor absolute cross-domain performance and limitations relative to docking. Its future-work discussion explicitly identifies protein structure, transfer learning and learning from small datasets. Accordingly, adding ESM or AlphaFold-derived geometry cannot by itself establish a novel research contribution.

Our question is narrower: can a small panel of measured affinities identify when geometry improves prediction for additional compounds, after accounting for a simple calibration shift? This distinction matters because support labels may correct assay-wide bias without demonstrating a preference between predictive representations. We therefore hold the two predictors fixed, fit an offset shared by the comparison variants, and evaluate the additional disagreement-dependent correction.

This manuscript is maintained as an evidence-bearing draft. Published claims, inspected source behavior, local dataset observations, synthetic checks and unperformed experiments are labeled separately. Appendix A tracks material claims in the eight-page main PerceiverCPI article. Appendix B extends the register to the ten-page supporting information. We do not interpret a difference between the article and an implementation detail as evidence of research misconduct or invalidity of all published results.

## 2 Related work and scope of contribution

MetaDTA [2], FS-CAP [3] and AdaMBind [4] establish support-based or few-shot affinity adaptation as an existing research area. MINE [5] studies local dynamic regressor competence, while local prediction pools [6] adapt expert combinations using local evidence. AssayRouter [7] further motivates comparison with frozen-source selection and query-local alternatives. These precedents preclude a broad novelty claim based only on support conditioning, local weights or fallback under sparse evidence.

The proposed contribution is an experimental decomposition of calibration and geometry-dependent adaptation, coupled with a controlled architecture and explicit identifiability limits. The statistical estimator is a regularized local regression construction, not a claimed new general learning algorithm. Whether this combination warrants a research contribution depends on comparison with established methods, reproducible held-out results and informative characterization of failures.

## 3 Audit of the PerceiverCPI reference implementation

We pin the official repository to commit 755f076ef51099a90332fd1ec8437f94e3750c0b. The legacy track preserves upstream behavior; the extension track records intentional changes and is trained anew. Checkpoint parity and paper-metric reproduction are distinct requirements.

### 3.1 Attention axes and feature extraction

The inspected forward path compresses protein features into one vector before the cross-attention block. Within each head, projected query and key vectors are reshaped to form a feature-channel outer product. For head width d, the score matrix has shape d by d. Its logits are an outer product; the softmax attention matrix need not remain rank one. These channels are not residue positions. Consequently, the attention block cannot directly be treated as a residue-level interaction map.

The generic featurize method omits protein and CAB processing and therefore is not a valid adapter for extracting the full pair representation. The appropriate exposure point is the actual forward output immediately before the affinity FFN. DynaCPI adds a separately defined compound-conditioned residue readout and uses it identically in both predictive paths.

### 3.2 Aggregation and numerical behavior

The source uses summed incoming bond messages inside directed updates and a mean over a padded incoming-message array at the atom readout. BatchMolGraph sets that padding width using the largest degree in the current batch. The resulting division can therefore vary with unrelated batch members. For messages [2,4], padded widths two and four produce means three and 1.5, whereas division by valid degree yields three in both cases. This isolated calculation demonstrates the arithmetic issue; its magnitude in trained molecular predictions remains to be measured.

The extension uses a masked valid-degree mean, device-aware tensors and guarded min–max normalization. The legacy path retains the original arithmetic for reproduction. Changes must not be presented as checkpoint-equivalent by default. The source also applies label-range weights during MSE training, which must be included when reproducing its training objective.

### 3.3 What our audit does not establish

The local dataset inventory is not a reconstruction of every published fold. Sequence equality does not establish equality of assay constructs when identifiers or mutation annotations have been lost. Shared input pairs are therefore flagged for provenance review before they are classified as duplicate measurements. No performance claim is refuted merely by the existence of those pairs. Protein-homology and scaffold leakage require additional analyses beyond exact identity intersections.

## 4 Statistical formulation

For task t, let a query compound q and support set S_t={(c_i,y_i)} share a verified construct and assay context. The sequence predictor returns s_i and the structure-enhanced predictor returns g_i. Both are frozen during adaptation. Define d_i=g_i-s_i and a global development-fitted prior coefficient w_0 in [0,1]. The zero-support prediction is m_0(q)=s_q+w_0 d_q.

A simple affine model s_i+b+w d_i cannot separately identify b and w if every d_i equals a constant. Increasing support count does not remove this degeneracy if the disagreements remain constant. This observation concerns statistical predictions, not causal evidence that geometry changes molecular binding.

For nonnegative query-local weights k_i, set W=sum_i k_i and, for W>0,

$$\bar d=\frac{\sum_i k_i d_i}{W},\qquad x_i=d_i-\bar d,\qquad u_i=y_i-s_i-w_0d_i.$$

The coefficients solve

$$\min_{|b|\le B,\,-w_0\le\theta\le1-w_0}\sum_i k_i(u_i-b-\theta x_i)^2+\lambda_b b^2+\lambda_\theta\theta^2.$$

Since sum_i k_i x_i=0, the mixed quadratic term vanishes. With positive regularizers, the minimizer is unique and separates into

$$b=\operatorname{clip}_{[-B,B]}\frac{\sum_i k_i u_i}{W+\lambda_b},\qquad \theta=\operatorname{clip}_{[-w_0,1-w_0]}\frac{\sum_i k_i x_i u_i}{\sum_i k_i x_i^2+\lambda_\theta}.$$

The prediction is

$$\hat y_q=m_0(q)+b+\operatorname{clip}_{[-C,C]}\{\theta(d_q-\bar d)\}.$$

We fix initial engineering caps B=2 and C=1 pKd. The query clipping is an additional extrapolation safeguard, not part of the fitted quadratic objective. The total correction is bounded by B+C, but this does not imply non-degradation. The estimator can predict outside the interval between the two base predictions; it is not a pure convex mixture. At W=0 the output is m_0. At K=1 or constant disagreement, the differential term vanishes. A common shift of support labels changes calibration but leaves the differential coefficient invariant when weights and settings are label-independent.

The variation statistic V=sum_i k_i x_i^2 describes how strongly the panel probes differences between the predictors. It is not calibrated confidence. Activity cliffs, measurement noise, unrepresentative support and query extrapolation can invalidate the assumed local relation even when V is large.

## 5 Architecture and training protocol


A directed message passing network and a 2048-bit chiral Morgan fingerprint (radius two) encode each compound. The graph network performs three steps with 256 hidden features. Its atom readout divides by valid degree. The fingerprint projection and graph vector enter the shared PerceiverCPI attention operations. The resulting compound query attends over projected residue features with four heads. A masked mean residual and layer normalization produce a pooled protein vector for the final channel-attention operation. A two-layer scalar head maps the pair vector to affinity. We invert training-label normalization before estimating support corrections.

Frozen ESM-2 layer 33 supplies 1280 features per residue. We project these to 256 channels. For long constructs we use 1000-residue windows with 128 residues of overlap and average overlapping representations. This trades full-sequence context for complete residue coverage. Sequence hashes, window offsets, tokenizer and weight versions accompany every cache entry.

For the geometry pathway we require a uniquely mapped AlphaFold structure with complete finite N, CA, C and O coordinates at ≥80% of construct positions and exact agreement at mapped standard residues. Each valid residue receives up to 30 spatial neighbors within 20 Å and any verified peptide neighbors. Construct indices remain intact across coordinate gaps. Node inputs comprise 14 scalars and three vectors. Edge inputs comprise 37 scalars and one vector. Directed PAE and confidence fields carry availability flags. Three GVP convolution layers use 128 scalar and 16 vector node channels. An invariant scalar projection supplies a residual in the 256-dimensional sequence space [8]. The projection starts at zero. Missing geometry gives the sequence prediction exactly.

First we train the sequence model against standardized training labels. We freeze its parameters and normalization state before training the geometry branch. Gradients through the frozen interaction block still reach the geometry encoder. We then freeze both predictors. This shares one foundation across two forward passes and isolates the geometry branch during adaptation. A sequence residual branch matched within 5% of the trainable geometric parameter count controls for extra capacity. A metadata-only control tests whether coverage or confidence explains an apparent gain.

Training defaults are AdamW, learning rate 0.0003, weight decay 0.0001, gradient norm limit one and effective batch size 64. The epoch ceiling is 100 with early stopping after 15 validation checks. Separate protein groups determine base checkpoint selection, prior fitting and adapter validation. Support weights equal the fourth power of fingerprint Tanimoto similarity. We retain their absolute scale so distant supports have little influence. We fit a single global prior weight on adaptation-training groups and select offset regularization from 1, 5 and 20 using calibration-only validation error. With that offset fixed, we select differential regularization from 0.01, 0.1, 1 and 10. The query labels remain inaccessible throughout fitting and inference on final-test tasks.

## 6 Experimental design

We distinguish a reproduction of the published PerceiverCPI evaluation from a common-split comparison of original PerceiverCPI and DynaCPI. The first requires the original endpoint, code, preprocessing, splits, tuning and MSE/concordance metrics. The second tests a new hypothesis and requires matched data, preprocessing and access to measured support. Both zero-support and five-support results will be reported. The comparison with a calibrated PerceiverCPI baseline tests whether the extension improves on what the original model could achieve with the same experimental information.

The proposed primary cohort uses exact Kd measurements converted to pKd within a verified construct and assay context. Each task needs five support compounds and ten disjoint queries. Related proteins are grouped by connected components at ≥30% identity over ≥80% of the shorter construct. The five group partitions target 60/10/10/10/10 percent for base training, base validation, adapter training, adapter validation and testing. Eligibility, group counts and effective precision must be established before final testing. The target-only condition permits shared chemistry between distinct targets. Separate scaffold and dual-shift analyses measure generalization to new chemistry.

The four factorial variants are the default mixture (A), mixture plus calibration (B), mixture plus differential adjustment (C) and the full model (D). D versus B isolates adjustment beyond identical calibration. Compare with residual nearest neighbors, kernel regression, support fine-tuning, a query-only router, FS-CAP, AdaMBind and verified MetaDTA. All methods receive matched episodes. Individual predictors, global mixture and the parameter-matched sequence control expose alternative explanations.

We compute RMSE within each task and average tasks within constructs, constructs within protein clusters and clusters equally. The primary decision rule requires a ≥2% relative RMSE reduction of D versus B and a paired 95% cluster-bootstrap interval excluding zero. This is a provisional project criterion. We will report the absolute pKd difference, MAE, rank correlation, per-target harm and screening utility. Three trained seeds and ten predetermined support draws describe variation within targets. The number of independent protein clusters determines the effective generalization evidence.

## 7 Preliminary results

Table 1 reports all nine locally audited fold-zero CSVs. We computed intersections after RDKit stereochemistry-preserving canonicalization and exact protein-sequence matching. Each source file has a SHA-256 digest in the machine-readable inventory. Chemical and sequence identity provide a reproducible screening key. Two records with the same key can still represent different constructs, assays or experimental conditions.

| Dataset | Train rows | Validation rows | Test rows | Shared train–validation pair keys | Shared train–test pair keys |
|---|---:|---:|---:|---:|---:|
| DAVIS | 15,552 | 3,888 | 1,148 | 409 | 0 |
| METZ | 18,625 | 4,656 | 1,290 | 0 | 0 |
| KIBA | 61,720 | 15,430 | 4,432 | 127 | 0 |

Across datasets, staging retained 126,741 observations, 2,642 distinct canonical compounds and 462 distinct sequences. All staging rows passed the script's nonempty-sequence and RDKit molecule checks. The fingerprint cache has shape 2,642 × 2,048. These totals describe the local files before scientific eligibility filtering. Among DAVIS training rows, 10,950 of 15,552 labels equal five. The same value appears in 755 of 1,148 test rows. The longest test sequence was 2,527 residues in DAVIS and METZ and 4,128 in KIBA. This directly motivates auditing the original 500-position input limit against the residues actually retained.

The adapter solution agreed with numerical box-constrained optimization on 200 random problems. The largest absolute objective difference was 1.42 × 10⁻¹⁴. Sparse support, constant disagreement, support reordering, common label shifts, missing geometry and bounded corrections behaved as derived. On the RTX 4060 Laptop GPU, the pinned CAB classes evaluated four synthetic 256-channel vectors in batch and individually. The maximum absolute output difference was 1.49 × 10⁻⁸. This check covers isolated channel attention. End-to-end molecular encoder invariance and trained-checkpoint parity require separate tests.

### 7.1 A direct test of the reported DAVIS prediction floor

Supplementary Figure 2 describes first-fold predictions at or above pKd 5. We matched all 1,148 rows of an archived local PerceiverCPI test prediction file to the fold-zero DAVIS reference by ordered SMILES and sequence. The prediction file SHA-256 was 7072cbc46141847b9ec1ef20d8cb9065e99bbb0546cfe38a82991b4903096420. The resulting MSE was 0.700482. Of these predictions, 115 (10.0%) fell below five, with a minimum of 4.945896. Thus the floor observed in the authors’ illustrated run does not transfer exactly to this local checkpoint. Checkpoint equivalence, ensemble selection and all five folds remain unverified. This is a retrospective comparison of an existing baseline, not a DynaCPI training result.

### 7.2 Independent feature-cache and BindingDB feasibility checks (5 November 2026)

We checked the complete staged ESM-2 650M cache after extraction rather than trusting a progress marker alone. All 462 sequence hashes had corresponding readable, finite FP16 tensors with expected `[length,1280]` shapes, matching extractor metadata and no gaps across 570 windows covering 346,865 residue positions. The verification is recorded in `data/audit/esm_cache_integrity.json`. Two actual staged DAVIS compounds (with 545- and 588-residue proteins) were passed with genuine cached ESM embeddings through the randomly initialized sequence predictor. Batch and individual outputs agreed exactly in this limited check, with nonzero sequence-projection gradients under synthetic loss labels (`research/cached_sequence_pair_smoke.json`). These results establish software interface compatibility, not binding-affinity performance.

An independent geometry smoke test used the first 96 residues of AlphaFold's canonical human BRAF structure (AF-P15056-F1 version 6), constructing 2,094 directed residue edges with observed coordinates and available PAE. Invariant scalar outputs agreed after a rigid rotation and translation to a maximum token difference of 2.38 × 10⁻⁷; GVP input and projection paths both received finite nonzero gradients, and missing-geometry fallbacks passed. The sequence tokens were synthetic and there was no experimental affinity prediction (`research/geometry_smoke.json`). Full construct alignment, structure coverage and training remain outstanding.

The first BindingDB feasibility scan examined only the November 2026 *Articles* archive, not the full BindingDB distribution. It parsed 93,712 records, of which 2,650 had nonempty Kd fields and 2,490 were positive numeric, exact-looking Kd values. After basic single-chain, canonical residue alphabet, SMILES and mapped assay-ID filters, 2,430 candidate measurement rows remained across 555 preliminary construct–assay groups. Only 29 groups had at least 15 distinct raw SMILES and eight had at least 30. RDKit canonical isomeric SMILES parsing rejected 29 candidate rows; 28 eligible protein sequences at the 15-compound threshold formed 24 preliminary sequence-similarity components, while the seven sequences at the 30-compound threshold formed just four. These figures are screening evidence only: they neither establish full assay compatibility nor an eligible science cohort. The four-component high-support subset cannot fill five disjoint protein-group roles. The broader, approximately 594 MB full November BindingDB archive is required for the proposed group-based study. Source and screening outputs are in `data/audit/bindingdb_articles_kd_eligibility.json` and `data/audit/bindingdb_articles_kd_cluster_screen.json`.

All experiments above are software/data-audit observations. No trained DynaCPI accuracy comparison, held-out-target improvement or scientifically validated structural mechanism is reported.

### 7.3 Full November BindingDB exact-Kd screening, protein-group feasibility and source-QC limitations

We resumed the November 2026 BindingDB full archive with HTTP ranges, preserving its original incomplete download. The compressed 594,284,548-byte archive passed a complete ZIP CRC test; its SHA-256 digest is `470a3c2e4aebe3c493de3e9e46e739592eb8e29b3adafb5189c875442268a0b2`. The publisher's standalone MD5 endpoint returned HTTP 404, so this digest is locally computed rather than independently checked against a publisher MD5. Its uncompressed TSV contains 9,004,582,822 bytes. The audit scanned 3,243,660 source records and found 131,531 nonempty Kd fields, of which 82,481 had positive numeric, syntactically exact-looking nM values. After preliminary single-chain, residue-alphabet, SMILES, and reactant-to-assay filters, 80,139 candidate records were linked to assay keys. This is a strict *syntactic* pre-screen, not proof that source publications report precise, assay-compatible Kd values.

RDKit canonical isomeric-SMILES screening retained 80,042 candidate rows, resulting in 1,146 preliminary construct–assay tasks with >=15 distinct compounds and 377 tasks with >=30. They involve 555 and 215 unique construct-sequence strings, respectively. We compared all 4,157 preliminary exact-Kd candidate sequences using MMseqs2 18-8cc5c+ds-1 (all-vs-all `easy-search`, `-s 7.5`, sequence identity >=0.30, aligned coverage >=0.80 of the shorter sequence, connected components on accepted edges). Of 2,225 components across the complete candidate sequence set, 168 contain K=5-eligible tasks and 81 contain K=20-eligible tasks. The largest K=5 component holds 492 tasks, illustrating strong family imbalance. Excluding accepted local alignments shorter than 80 residues changes these eligible component counts to 171 and 82; the variation does not eliminate the giant-group imbalance. This is a sensitivity check, not a retrospectively chosen alternate final split.

In 1,000 *simulated* five-way allocations by independent protein components, a nominal 10% test-group quota yielded 17–687 K=5-eligible test tasks (median 48) because component sizes are highly unequal. These draws used no affinity values and did not select or commit a scientific train/test split. They demonstrate the importance of predeclared grouped evaluation, weighting by independent clusters, and reporting target/task coverage rather than assuming that pair count determines statistical precision.

Assay text was joined through the publisher's `ENTRYID_ASSAYID` metadata mapping. All 15,179 preliminary raw sequence–assay groups had name and description text, but text alone cannot prove comparable experimental measurements. Within 1,146 K=5 candidate tasks, 1,082 lack a reported pH, 1,107 lack a temperature and 366 have no article DOI; 95 have within-task replicated compounds whose reported pKd values differ by more than 0.1. Source provenance is dominated by ChEMBL and patent-derived entries. Two high-support groups contain inconsistent target-name annotations under a shared sequence–assay key, including FLT3 variants, and ten candidate rows have a source-vs-extracted sequence-hash discrepancy. These records require identity and assay review rather than being silently averaged or used as independent support examples. Missing condition fields are not themselves proof of incompatibility. The raw-source metadata and QC inventories are preserved under `data/audit/bindingdb_full_kd_*` for reproducibility.

The dataset is therefore materially more promising for *feasibility* than the 93,712-row Articles subset, but **the final assay/construct-valid eligible set, protein-family split, structure-mapped cohort, and measured model performance are all still unresolved**. No benefit of geometric support adaptation is established by these counts.

### 7.4 Verified software integration with real cached sequence and protein geometry

The complete staged official ESM-2 650M cache passed an independent file-by-file integrity check: 462 of 462 sequence features matched source hashes, metadata and tensor shapes, with 346,865 amino acids covered by 570 extraction windows. With two real staged DAVIS compounds and corresponding ESM embeddings, the untrained sequence predictor's batched and single predictions agreed exactly in an engineering test, and its sequence projection received nonzero gradients. A subsequent combined test passed a 766-residue canonical AlphaFold BRAF structure, its 18,789 directed GVP graph edges, real ESM features and a real RDKit molecule through the shared sequence/geometry readout. The geometry projection's all-zero initialization and missing-geometry pathway reproduced the sequence prediction exactly; a nonzero geometry projection changed the output, and a synthetic loss produced nonzero GVP gradients. The combined test used randomly initialized predictive weights and a synthetic target and does **not** report an experimental affinity score.

### 7.5 Pre-split episode identity and quality-triage verification (5 November 2026)

Using the complete November 2026 BindingDB export (3,243,660 source rows), the initial exact-looking Kd filter identified 80,139 assay-linked candidate observations. A subsequent source-provenance, exact-sequence-hash and RDKit identity verification retained 80,034 technically valid observations and 78,493 within-task canonical-compound median records. These are not scientifically approved exact-affinity observations: assay text, construct identity and source qualifiers still require review. We identified 1,146 candidate tasks with at least 15 distinct compounds and 377 with at least 30.

MMseqs2 18-8cc5c+ds-1 was used at sensitivity 7.5 to search 4,157 protein sequences against themselves. Under an operational threshold of at least 30% aligned sequence identity and at least 80% coverage of the shorter sequence, connected components yielded 2,225 sequence groups overall. The K=5 and K=20 candidate cohorts occupy 168 and 81 components, respectively, but the largest component contains 492 of the 1,146 K=5 tasks. This imbalance makes naive group-count-balanced splitting unsuitable as a guarantee of balanced task counts. An additional sensitivity audit varying minimum alignment length did not remove the dominance of the largest component. Short shared domains, unobserved homology and assay constructs remain biological concerns.

The candidate episode-generation script reserves ten distinct query compounds by deterministic hashes derived only from task and compound identity, separately from prospective support compounds. Independent verification confirmed 1,146 K=5 and 377 K=20 candidate episodes with disjoint query/support compound identities, nested support prefixes, unique task and compound IDs, and valid preliminary component assignments. These are **PRE-SPLIT quarantined manifests**, not final evaluation episodes. For a label-blind metadata triage, 1,041 K=5 and 299 K=20 tasks had no hard automated review flags, spanning 158 and 72 preliminary protein components, respectively. Only 19 of the 1,146 K=5 tasks had no automated flags at all, primarily due to missing reported pH or temperature. Missing reporting does not independently demonstrate assay incompatibility; neither does a shared assay identifier establish compatibility.

Artifact provenance: `data/audit/bindingdb_full_kd_mmseqs_component_screen.json`, `bindingdb_full_kd_candidate_episode_manifest_validation.json`, `bindingdb_full_kd_candidate_review_triage.json`, and the source-linked `bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl` / `bindingdb_full_kd_candidate_episodes_PRE_SPLIT.jsonl` datasets. Full experimental validation, approved eligibility policy, final component assignments, model fitting and affinity results remain outstanding. To support a blinded provenance review, a separate metadata-only queue of 30 tasks across 30 distinct preliminary protein components was selected deterministically without examining measured affinity labels (`data/audit/bindingdb_assay_manual_review_queue_30.csv`). Its blank adjudication fields explicitly require assessment of construct identity, measurement semantics and experimental comparability; it is not an automatically approved or representative scientific test cohort.

### 7.6 Closed-loop provenance controls, episode separation and archived checkpoint parity

A final engineering audit examined reuse of compound identities across multiple affinity assays for the same candidate protein sequence. Among the 555 K=5-eligible candidate sequences, 185 contained multiple eligible assay tasks, with 114 containing at least one compound serving as a query in one assay and a support candidate or query in another. There were 1,933 distinct within-sequence compound identities in the query-to-other-assay-support category. These are *identity overlaps*, not proof of the same biological measurement or actual query-label leakage. To prevent accidental contamination, the prediction-facing `dynacpi/episode_data.py` accepts support observations only from the requested construct–assay task, and excludes all query labels; labels are provided through a separate evaluator-facing interface. Tests over 4,961 preliminary episode requests verified assay isolation, zero query/support identity overlap and label-perturbation independence of prediction inputs. This does not establish that source Kd values are compatible or correctly qualified.

We also created a source-linked, label-blind review sheet of thirty tasks from thirty preliminary protein components, with accessions, assay descriptions and article/patent metadata. A fail-closed validation module refuses approvals unless a human supplies affirmative construct, exact-Kd semantics and observation-compatibility decisions with review evidence. All thirty tasks currently remain unreviewed; none is approved for research training. The group-allocation algorithm was tested only in dry-run mode, assigning 168 preliminary components across the intended 60/10/10/10/10 roles: 100/17/17/17/17 protein groups and 735/110/101/100/100 candidate tasks. The dry run accessed no affinity labels, and no assignments were frozen for scientific evaluation. This procedural success must not be interpreted as independent protein-domain separation or sufficient effective test sample size.

To verify the integrity of the preserved historical PerceiverCPI track, we independently loaded the archived local DAVIS fold-zero checkpoint `FINAL_model0_best.pt` using the unchanged reference implementation (six relevant source files checked byte-for-byte). We reproduced *all 1,148* archived test predictions with the reference 50-compound inference batch size, tokenizer and nonchiral Morgan fingerprints. Mean and maximum absolute discrepancies were zero to recorded floating-point precision. This establishes numerical parity for this exact local checkpoint, dataset and inference convention, but does not prove parity with every trained model or result in the published article. In particular, the new ESM-2/GVP-based DynaCPI has not produced an independent affinity benchmark.

Artifacts: `research/legacy_checkpoint_parity.json`, `data/audit/bindingdb_full_kd_cross_assay_episode_overlap.json`, `bindingdb_full_kd_episode_loader_validation.json`, `bindingdb_assay_manual_review_status.json` and `bindingdb_full_kd_group_split_dryrun_policy.json`. The full eligibility and planned split protocol is maintained in `research/Kd_Curation_and_Split_Protocol.md`. A separate reproducible engineering validation suite reran the adapter, episode, manual-review, split, sequence, geometry and checkpoint-parity checks sequentially, reporting **9/9 passed** in `data/audit/v4_validation_suite.json` with individual execution logs. These tests support implementation integrity only; they do not constitute held-out experimental-affinity accuracy measurements.

### 7.7 Independent reference sequence, publication provenance, and analysis-readiness checks (5 November 2026)

A label-blind verification pass revisited the fixed thirty-task BindingDB assay-review queue without opening or ranking affinity measurements. Official UniProtKB records were retrieved for all 30 accession identifiers. Each candidate sequence in the preliminary binding-data cache matched its linked reference sequence exactly; 30/30 matches establish metadata-to-reference identity only. BindingDB explicitly cautions that its listed reference sequence can differ from the experimental protein used in the original study because of constructs, purification tags, substitutions, or other experimental differences. Thus this observation was not treated as verification of the physical protein tested in any experiment. A separate keyword screen of the original assay descriptions identified 13/30 tasks with indications requiring additional attention to protein domains, engineered tags, complexes, inhibition or competition measurements. Keyword absence does not establish validity.

Europe PMC resolved 20/20 distinct PubMed identifiers cited across the review sample; seven tasks had patents without indexed article references in the supplied queue. Retrieved titles and DOI metadata help identify source literature but do not independently verify experimental Kd semantics. All evidence was summarized in `research/BindingDB_30_Source_Evidence_Dossier_20261105.md`, which excludes affinity labels and provides individual source references and unresolved review questions. **No source publication or patent was independently adjudicated as an exact-Kd-compatible assay in this step**. All 30 tasks remain unapproved.

To prevent premature scientific training, `dynacpi/scientific_cohort_gate.py` now requires separately recorded source-reviewed cohort and frozen protein-group role assignments with matching checksums, review references and disjoint component roles. The real gate correctly rejects the current candidate pool; mock-manifest tests established only structural validation of these controls. In addition, `dynacpi/paired_evaluation.py` implements fixed-query/seed/draw paired regression evaluation, task-then-protein-cluster macro RMSE/MAE and paired cluster-bootstrap confidence intervals. A 3,600-row entirely synthetic regression test passed checks for input consistency, mismatched labels, duplicate queries and non-finite values. No DynaCPI predictions were generated from the preliminary experimental affinity records.

The latest sequential test suite passed 11 of 11 engineering/statistical checks (`data/audit/v4_validation_suite.json`). These are not scientific performance results. The principal unresolved milestone remains source-adjudicated assay/construct eligibility, followed by a truly independent, frozen evaluation split and matched model assessment.

### 7.8 Trained engineering models, exact structural mapping and an initially negative development observation (5 November 2026)

We executed an engineering-only supervised validation of the proposed staged sequence/geometry architecture. The isolated sequence foundation was trained for three epochs on the retained original DAVIS pilot task allocation (24 identifier-disjoint training protein sequences, 1,296 compound–protein observations; eight validation sequences, 432 observations). ESM-2 650M residue embeddings were frozen and read from the audited cache. Target-macro validation RMSE for the sequence branch was 0.811185, 0.813853 and 0.817272 over epochs 1–3. The saved sequence checkpoint corresponds to epoch 3, whereas epoch 1 displayed the lowest validation error; these limited engineering numbers must not be interpreted as a systematic comparison to the original PerceiverCPI benchmark or as evidence for a real cold-family holdout. Raw pKd denormalization was separately verified when reloading the saved model checkpoint.

An independent mapping survey found that 30/32 selected pilot protein sequences matched contiguous subsequences of human UniProtKB reference sequences exactly, including all eight validation targets; EPHA6 and FAK did not match exactly in that strict reference lookup. Importantly, the pilot ROCK2 sequence aligned to residues 686–1388 of the 1,388-residue reference O75116, demonstrating why a full-length 3D coordinate index cannot be directly substituted for the truncated/partial sequence record. A deterministic limited structure cohort of nine training and four validation sequences was assembled using AlphaFold v6 coordinates and PAE. Reference PDB residue indices were transformed into sequence-local coordinates, sequence identity checked before mapping, and complete N/CA/C/O backbone coverage confirmed for every selected region. The preparation recorded source PDB/PAE and graph SHA-256 digests. These criteria establish engineering alignment, not that the historical affinity assay used the corresponding physical construct.

The sequence foundation was then frozen and a three-layer GVP branch optimized for two supervised epochs using 486 training and 216 validation compound–protein measurements from the structure-mapped pilot subset. On the four same validation targets, the frozen sequence predictor achieved target-macro RMSE 0.925685. The geometry branch reached 0.935921 after epoch 1 (best saved) and 0.963520 after epoch 2, **worse overall**. Independent reload of the saved geometry checkpoint reproduced these numbers; geometry reduced RMSE slightly for FRK (−0.002210) and MEK5 (−0.002194) while increasing RMSE for ADCK4 (+0.044869) and OSR1 (+0.000477). This is a negative development observation on a tiny, selected, not-family-disjoint benchmark, not a definitive generalization result about protein geometry.

The A/B/C/D adapter was connected to frozen model outputs and stereochemistry-aware Morgan-Tanimoto fourth-power weights. Twelve actual BindingDB candidate-task identities and 120 queries passed synthetic-affinity interface tests; a complete real-ESM/AlphaFold inference-through-adaptation test passed with fully synthetic support values and random model weights. The two trained engineering checkpoints were subsequently used for a separate *development-only* support exercise on the four geometry-checkpoint-selected DAVIS targets, with 10 identity-determined queries, K=5 and 10 identity-determined support draws per target. Fixed engineering settings were w0=0.5, lambda_b=5 and lambda_theta=0.1, without hyperparameter selection. Mean episode target-macro RMSE for variants A, B, C and D was 0.86101741, 0.86013856, 0.86101741 and 0.86013856, respectively; D minus B differed by +5.2×10^-11 RMSE. Total similarity kernel weights W were approximately 0.003–0.009, while centered disagreement variation V was around 10^-15–10^-11, rendering the regularized differential term effectively inactive. These results reuse checkpoint-selection targets and therefore cannot support a scientific improvement claim.

The unapproved broader BindingDB candidate screen has stronger K=5 chemical neighborhood support (median Tanimoto^4 weight sum 0.215) than this four-target DAVIS development subset, so transferring the pilot null result to the planned main cohort would be unsound. It remains essential to test geometry complementarity and whether support panels have adequate disagreement variation on a genuinely source-verified, independent adaptation cohort. The default v4 estimator and its regularizer grid are preserved pending development evidence; changes are not justified by inspection of an unreleased scientific test set.

The latest `scripts/run_v4_validation_suite.py` checkpoint independently reran 15 engineering and statistical integrity checks, all passing. Artifacts: `data/engineering_pilot/uniprot_exact_subsequence_survey.json`, `limited_alpha_3d_graph_manifest.json`, `research/pilot_sequence/`, `research/pilot_geometry_9train_4val/`, `research/pilot_geometry_9train_4val/support_adapter_development_diagnostics.json`, and `data/audit/v4_validation_suite.json`. The source-adjudicated BindingDB cohort and frozen five-way independent protein-group split remain unavailable and no paper-level performance comparison has been completed.

### 7.9 Support-size sensitivity, artifact monitoring and mechanism diagnosis (6 November 2026)

To determine whether the inactive geometric differential observed at K=5 arose merely from the small size of the support panel, we performed a fixed-protocol **development-only** support-size sweep using the previously saved sequence-foundation checkpoint (epoch 3) and geometry checkpoint (best epoch 1), without further model fitting or hyperparameter selection. The same four DAVIS development targets—FRK, MEK5, ADCK4 and OSR1—were selected because correctly mapped AlphaFold structures and cached ESM embeddings were already available. They were also used in geometry checkpoint selection and are therefore **not an unbiased independent test**. For every target we retained the same ten deterministically selected query compounds, with ten label-independent nested support draws at K=0,1,2,5,10,20. All variants used the original frozen models and fixed engineering defaults: w0=.5, lambda_b=5, lambda_theta=.1, bounded correction, Morgan Tanimoto^4.

| Support size K | B: calibration-only macro RMSE | D: calibration + geometry differential macro RMSE | D–B macro RMSE | Mean support weight sum W |
|---:|---:|---:|---:|---:|
| 0 | 0.86101741 | 0.86101741 | 0 | 0 |
| 1 | 0.86079179 | 0.86079179 | 0 | 0.001240 |
| 2 | 0.86061335 | 0.86061335 | +1.87e−11 | 0.002235 |
| 5 | 0.86013856 | 0.86013856 | +5.21e−11 | 0.006281 |
| 10 | 0.85936112 | 0.85936112 | +1.04e−10 | 0.012388 |
| 20 | 0.8573889581 | 0.8573889583 | +2.46e−10 | 0.028842 |

Although increasing support K produced a modest reduction in the development calibration error, the centered geometry differential remained numerically negligible. At K20 the mean weighted centered predictor-disagreement variance V was 6.50e−11, compared with lambda_theta=.1; the descriptive ratio V/(V+lambda_theta) was approximately 6.50e−10. The corresponding W/(W+lambda_b) was approximately .005735. These ratios illustrate the severe scale mismatch between evidence and fixed regularization; they are not an exact formula for the fitted theta or a guarantee that weaker penalties would help. Indeed the centered support disagreement covariance in the numerator was also tiny. The outcome at K5 independently reproduced the previously archived four-target development report at numerical tolerance.

This diagnostic identifies a real **v4 identifiability failure in the narrow development experiment**, not a disproof of the broader research hypothesis. Protein geometry may require better complementary representations and independently compatible assays, and the unapproved BindingDB candidate pool has a different chemical neighborhood distribution. We did not retune parameters on these observed development labels, change the architecture or make any claims regarding a held-out family test. Source artifacts: `research/support_k_sweep_dev/results.json`, `aggregate.csv`, `SUMMARY.md`, `identifiability.json` and `IDENTIFIABILITY.md`. The K5 consistency/checkpoint hash test is `tests_v4/k_sweep_result_integrity_test.py`. We also implemented `scripts/monitor_benchmarks_v4.py` to track engineering test outcomes, checkpoint hashes, the scientific cohort approval gate and new benchmark artifacts without initiating fitting. The main source-verified exact-Kd BindingDB study remains blocked pending full assay and protein-construct adjudication and a frozen independent protein-group split.

### 7.10 Geometry training seed variability and repeat of the support differential experiment (6 November 2026)

We next tested whether the initial negative geometry result was robust to weight initialization. The first full engineering run used seed 1105. We ran **two additional predeclared initialization seeds (1106 and 1107)** on precisely the same nine training protein sequences (486 DAVIS pair observations), four development validation targets (216 observations), frozen saved sequence foundation and two GVP training epochs. Every run used the same optimizer configuration and isolated model-checkpoint outputs, retaining the original seed-1105 checkpoint unchanged. The minimum observed *development validation* target-macro RMSE from the two epochs was:

| GVP initialization seed | Best development epoch | Frozen sequence macro RMSE | Geometry macro RMSE | Geometry−sequence |
|---:|---:|---:|---:|---:|
| 1105 | 1 | 0.925685 | 0.935921 | +0.010236 |
| 1106 | 2 | 0.925685 | 0.920050 | −0.005635 |
| 1107 | 1 | 0.925685 | 0.915971 | −0.009714 |

The mean of the three best-epoch geometry-minus-sequence differences was −0.001705 RMSE (sample SD 0.010540), illustrating seed sensitivity and the uncertainty of interpreting the initial two-epoch pilot. **These are not independent estimates of generalization or evidence that geometry reliably improves prediction**: the same four targets were used for selecting the best epoch in every run, only three random seeds were examined, and all targets belong to the narrow legacy DAVIS development setting.

For each alternate GVP checkpoint we repeated the **identical frozen support K=0,1,2,5,10,20 episode protocol**, holding query compounds, ten nested identity-based support draws, training-label conversion and adapter regularizers fixed. At K20:

| GVP seed | Calibration-only B macro RMSE | Full D macro RMSE | D−B |
|---:|---:|---:|---:|
| 1105 | 0.8573889581 | 0.8573889583 | +2.46×10^−10 |
| 1106 | 0.8529299312 | 0.8529299314 | +2.00×10^−10 |
| 1107 | 0.85154725 | 0.85154728 | +2.48×10^−8 |

All three runs retained negligible differential benefits even as the geometry prediction errors shifted across seeds. Thus the *local support-differential null effect* is less sensitive to initialization than the raw geometric predictor in this engineering cohort. It is plausible that the centered disagreement covariance and relative regularization suppress the correction, but this does not prove that modifying the model or tuning the penalty would help. No testing on independently held-out, source-approved BindingDB targets has yet been performed. The consolidated engineering suite was expanded to include sweep and seed-result integrity checks and achieved 17/17 passes.

Artifacts: `research/pilot_geometry_seed_sweep_dev/seed_variability.json`, `SEED_VARIABILITY.md`, `adapter_seed_comparison.json`, `ADAPTER_SEED_COMPARISON.md`, `research/support_k_sweep_seed_dev/seed_1106/results.json`, `seed_1107/results.json`, `tests_v4/geometry_seed_variability_integrity_test.py`, and `data/audit/v4_validation_suite.json`. No final inference or drug-discovery improvement claim follows from these observations.

### 7.11 Primary-source discovery and the unresolved exact-Kd curation gate (6 November 2026)

The preceding supervised experiments were restricted to legacy DAVIS development observations, not the independent, source-verified BindingDB cohort required by the central research hypothesis. To improve audit reproducibility without conflating bibliographic metadata with biological verification, we performed an automated **source-access inventory** for the previously fixed thirty candidate assay tasks. Using their source-provided PubMed identifiers, we queried the Europe PMC RESTful service and matched **20/20 distinct PubMed IDs** to publication metadata. For explicitly open-access full-text records, we transiently parsed XML to record only method-related section identifiers, headings, and non-interpretive keyword frequencies; source full-text bodies were not saved or redistributed. Two papers yielded safe open-access method-section indexes; seventeen others were not available as public open-access full text via this route; one nominally accessible XML contained a DTD/entity declaration and was deliberately excluded from automatic parsing pending a safe manual alternative.

The review worklist retains 30 scientifically **UNREVIEWED** tasks: two tasks have an open-access method-section index, twenty need further original-article access/review, seven require primary patent review, and one lacks a usable primary-source identifier. These categories are mutually exclusive task-level access statuses, not findings about experimental endpoints, construct identity or study validity. No construct or exact Kd measurement was approved; all biological evidence fields remain unknown. The canonical human review CSV's SHA-256 matched before and after the source-access stage. The generated reviewer action list records only the original task identifiers, source access category and potential method-section locations; the actual paper/patent and supplementary records still require independent qualified adjudication.

The new source-index integrity test checks the exact thirty-task queue identity, published-identifier matching, absence of human approval, rejection of XML DTD/entity declarations, prohibition on copying source article bodies, and continued blocking of the scientific cohort gate. Added to the prior engineering suite, this yielded **18/18 PASS** on 6 November 2026. This is a software and bibliographic access result, not a new model benchmark or verified affinity dataset. Artifacts are `research/source_access_v4/source_access_index.json`, `SOURCE_ACCESS_SUMMARY.md`, `REVIEW_ACTIONS.md`, `worklist_manifest.json`, `tests_v4/source_access_index_integrity_test.py`, and `data/audit/v4_validation_suite.json`. Neither a scientist-approved exact-Kd cohort nor a frozen independent protein-group split existed at the end of this checkpoint.

### 7.12 Source-paper construct differences versus canonical reference sequence inputs (6 November 2026)

For two previously indexed open-access assay publications, we inspected explicit experimental-method sections and compared their reported protein fragments against the full-length UniProt reference sequence lengths recorded for the corresponding unapproved BindingDB candidate tasks. The NPR-C candidate task uses the exact canonical P17342 input of 541 amino acids, but the source paper (PMID 35333039; PMCID PMC9014859, section `sec4.4`) describes SPR binding measurements performed with a tagged receptor region spanning residues 27–541, nominally 515 amino acids. The BindingDB assay description itself mentions this 27–541 region; the discrepancy is between source assay region and the retained canonical *sequence feature input*, not necessarily between the database description and the source methods.

The Neuropilin-1 candidate task uses the exact canonical O14786 sequence (923 residues), while the source paper (PMID 29648813; PMCID PMC5957473, sections `sec4.2.1`, `sec4.2.3`) describes a tagged NRP1-b1 fragment spanning residues 273–427, nominally 155 amino acids. The narrative also mentions NRP1-b1b2 in connection with binding analyses, reinforcing the need to match particular recorded measurements to the specific experimental construct and protocol rather than assuming a single physical sequence throughout the paper. Neither the source-paper fragment span nor the record's full reference sequence alone establishes the entire biochemical construct with tags, mutations, processing or oligomeric state.

These two cases demonstrate a plausible **sequence/structure-to-measurement alignment error** if future benchmark construction blindly uses full-length UniProt sequence or structure embeddings for measurements obtained on isolated tagged regions. The relevant null hypothesis is NOT disproved by the discrepancy; actual per-compound Kd identity, qualifiers, source table correspondence and same-assay comparability are still UNVERIFIED. We did not train, evaluate, release or approve any source-Kd cohort using these findings. The two cases remain UNREVIEWED within the original thirty-task set. Source locators and derived region-length risks are retained without article text or chemical potency data in `research/source_access_v4/PRELIMINARY_CONSTRUCT_ALIGNMENT_REVIEW.md` and `construct_alignment_risk_screen.json`. The source metadata and approval-gate integrity tests are in `scripts/check_primary_construct_flags_v4.py` and `tests_v4/construct_alignment_risk_integrity_test.py`; the latter tests only deterministic record/locator consistency, not scientific source approval.

### 7.13 Label-independent triage of experimental-assay metadata and protein-construct risks (6 November 2026)

As a further source-curation readiness step, we applied a deterministic and **label-independent metadata review triage** to the fixed thirty-task BindingDB manual-review sample, without accessing per-compound affinity values or changing any human adjudication field. The input comprised the existing assay-description spreadsheet, its previously verified canonical protein reference lengths, the publication/patent source-access index and the two source-paper fragment locators described in Section 7.12. The screen is a set of deliberately interpretable, nonbinding review indicators, rather than an automatic assay classifier or a source-validity judgment.

Four tasks received a P0 metadata-clarification priority, twelve received P1 construct/endpoint review priority and fourteen remained P2 standard primary-source review. Among the flags (which overlap across tasks), six assay descriptions gave numeric amino-acid regions shorter than their stored full-length reference protein sequence, eight mentioned tagged or labeled reagents, four used wording suggestive of a surrogate activity/endpoint, and four included competition or displacement protocols. These counts describe *metadata language*, not confirmed experimental defects. The P0 cases illustrate different uncertainty classes: a candidate CFTR affinity field described as a half-effective concentration; a GLP-1 receptor-targeted record whose description instead discussed a glucose colorimetric assay; a PD-L1 record with no assay description; and a carbonic anhydrase II record without a PMID/patent identifier in the review sample, but with a populated DOI subsequently resolved as documented in Section 7.14. No error in any particular stored numeric Kd value has been proven by these descriptions.

An important multi-protein exception was also identified: the Elongin-C task's metadata refers to VHL residues 54–213, which exceed the 112-residue Elongin-C canonical reference length. **This span belongs to a potential partner chain, not necessarily the named Elongin-C target**; blindly treating every numeric range as belonging to the stored reference sequence would create its own false flags. We therefore explicitly identify this case as requiring *protein-chain association review* and do not infer a physical Elongin-C construct from those residue indices. Conversely, lack of a numeric region or absence of a keyword cannot confirm that the assayed protein was full-length or that its endpoint was exact Kd.

The report is preserved in `research/source_access_v4/assay_metadata_risk_triage_30.json`, `assay_metadata_risk_triage_30.csv` and `ASSAY_METADATA_TRIAGE.md`, with input source hashes. Regression checks `tests_v4/metadata_risk_triage_integrity_test.py` verify the fixed thirty-task identity, immutable reviewer spreadsheet SHA-256, guarded source approval status, numeric region parser controls including a pH false-positive test, and the high-risk metadata examples. **All thirty remain UNREVIEWED; zero were scientifically approved** and no independent Kd training/test split was released. This triage supports reviewer prioritization only, and did not fit, select or evaluate a binding-affinity predictor. On 6 November 2026, the consolidated integrity suite passed **20/20 checks** with zero failed cases, including the new source-metadata triage test; this verifies software and evidence-boundary invariants, not an approved dataset.

### 7.14 P0 external source reconciliation and correction of the DOI-only bibliographic category (6 November 2026)

We revisited the four most urgent metadata-review tasks using source-specific evidence, while deliberately avoiding automatic assay adjudication or access to per-compound numerical affinity labels. The CFTR-associated source publication (PMID 12852954) describes cell-based functional chloride transport and half-maximal stimulation in its abstract; this does not itself substantiate the exact equilibrium protein–compound Kd endpoint implied by the candidate dataset field. The GLP-1 receptor candidate (BindingDB assay `11490_1`) is cross-referenced in an independently catalogued assay registry (`AID 1919769`) whose protocol instead describes a glucose/colorimetric binding experiment. This raises a substantial *reported-target versus measured-assay-participant* question. Both observations are strong prompts for primary experiment and original table/record inspection; neither proves every deposited record is wrong or authorizes an automated rejection.

A separate source-index problem was corrected. The carbonic-anhydrase II candidate was originally marked `MISSING_PRIMARY_SOURCE_IDENTIFIER` because the first access index considered PubMed and patent identifiers but overlooked a populated DOI in the original review CSV. Crossref resolved that DOI to a 1996 binding-site study. The revised category, `PRIMARY_DOI_ONLY_METHODS_PENDING`, accurately reflects an identifiable publication whose primary experimental method and individual dataset measurements remain unreviewed. The fourth candidate, PD-L1, has an identifiable primary patent but its original BindingDB review row lacks an assay description; obtaining the exact patent experiment and record-level endpoint semantics remains necessary. This correction supersedes the earlier claim that the carbonic-anhydrase task had no usable source identifier; its source METHOD remains unverified.

The deterministic source-evidence crosswalk preserves the four original task identifiers, corresponding source evidence tiers (publication abstract, registered assay description, DOI bibliographic metadata or primary patent locator), candidate record identity and outstanding questions. It explicitly stores `human_review_approval=false`, `each_numeric_Kd_measurement_verified=false`, no record exclusion, and no potency measurements. Updated source-index and reviewer-worklist reports distinguish DOI-only access from truly missing source identity, retaining the previous P0=4, P1=12, P2=14 priorities. Artifact paths are `research/source_access_v4/P0_SOURCE_CROSSWALK.md`, `p0_source_crosswalk.json`, `source_access_index.json`, and `REVIEW_ACTIONS.md`. Regression test `tests_v4/p0_source_crosswalk_integrity_test.py` checks original review-CSV SHA, exact source ID match, DOI-only correction, zero approved tasks and blocked scientific training. This is source-governance engineering, not an independent predictive benchmark. Model weights and DAVIS results remain unchanged.

### 7.15 BindingDB original-reactant lineage consistency, beyond source-method metadata (6 November 2026)

After remote access to the original Windows DynaCPI v4 workspace was restored, we performed the first **real, record-level technical lineage audit** of the fixed thirty unapproved BindingDB review tasks. Previously prepared median-level compound observations were independently joined back to the original raw BindingDB reactant rows and preliminary unaggregated candidate records by the source `Reactant_set_id`. The fixed set comprised **919 distinct grouped compound observations referencing 921 distinct original reactant IDs**, with two groups each formed from two original source records (in PDE6D and ENPP1 task candidates). Every referenced source reactant (**921/921**) and preliminary unaggregated candidate row (**921/921**) was located, with no duplicate reactant IDs in the fixed subset.

The audit checked that each raw `Kd (nM)` source string was a strictly positive unqualified numeric value, that the preliminary pKd transformation was numerically consistent with `9 - log10(Kd in nM)`, and that recorded sequence hashes, assay keys, review group counts, and both multi-record compound median/range calculations matched their independently read source inputs. Every one of the 921 raw Kd strings was strictly positive numeric; no source-record loss, mismatch in these conversion operations, protein-reference hash mismatch, grouped median discrepancy, or other predefined critical lineage discrepancy was detected. The reviewer CSV was SHA-256 checked before/after; no source, canonical review decision or original trained checkpoint was altered.

The positive result demonstrates **technical provenance reconstruction and arithmetic consistency**; it does **NOT** validate experimental semantics of the Kd field. For example, numerical transformation can be perfectly consistent even if an original assay describes a functional half-maximal cellular response, a non-equilibrium surrogate endpoint, a mislabeled interacting participant, or a construct that differs from the canonical protein feature. Consequently, this outcome does not clear the previously documented CFTR, GLP-1 receptor, protein-fragment and complex-association questions, and no source-reviewed BindingDB cohort or independent protein-family split was released. All 30 review tasks remain UNREVIEWED.

Artifacts: `scripts/audit_30_record_level_provenance_v4.py`, `research/source_access_v4/record_level_provenance_30.json` and `RECORD_LEVEL_PROVENANCE_30.md` contain source-file hashes and task-level aggregate audit counts, without exporting source numeric affinities, compound identities, or reactant IDs. The isolated `tests_v4/record_lineage_synthetic_test.py` passed five synthetic scenarios (valid replicate median, incorrect pKd conversion, missing raw reactant, qualified source Kd, changed review status); the `tests_v4/record_lineage_real_integrity_test.py` independently reruns the true 30-task audit and checks the science gate. This is a research data engineering result, not a trained-model performance experiment or a source-scientifically-approved task count.

### 7.16 Per-reactant experimental evidence sign-off workflow (6 November 2026)

A key limitation after reconstructing the original 921-record BindingDB technical lineage (Section 7.15) was the absence of an auditable evidence structure for the *scientific validity* of each underlying measurement. We implemented a local-only source-review worksheet generator and validator covering all 921 BindingDB `Reactant_set_id` records in the fixed thirty-task sample. The worksheet preserves immutable task, assay, protein component and sequence hashes and the source's bibliographic/organism/curator identifiers, while explicitly omitting chemical structures and individual numerical affinity labels. It defines review columns for the primary publication/patent method location, precise data table/figure/row mapping, measured binding participants, actual experimental construct sequence and tags, assay method, exact Kd endpoint, units/qualifier, reviewer identity, date and substantive rationale.

For a prospective `APPROVE` declaration, the validator requires affirmative independent record matching, direct equilibrium Kd verification, physical construct verification and compatible assay conditions; it refuses IC50 or other endpoint types and non-EXACT qualifiers. Missing source method/table locators, an unsigned declaration, an incorrect sequence digest, changed original metadata, duplicate/reactant omissions and contradictory claims on an `UNREVIEWED` row all fail validation. It does **not** grant scientific approval merely because an individual supplies structurally complete fields: the summary is explicitly marked as *self-reported review declarations pending independent primary-source adjudication*, and scientific training authorization remains false.

The real local worklist contains **921/921 UNREVIEWED records across 30/30 UNREVIEWED tasks**, zero verified original experiments and zero source-approved tasks. The canonical review CSV and raw provenance inputs were not edited. The source worklist is kept only on the authorized Windows device at `data/audit/source_record_evidence_worklist_30_LOCAL_PRIVATE.csv`, and is explicitly excluded from GitHub to avoid publishing record identifiers or future reviewer notes. A public aggregate readiness summary, `research/source_access_v4/record_source_evidence_readiness_30.json` and `RECORD_SOURCE_EVIDENCE_READINESS.md`, exposes only input hashes and per-task review counts. The new tool is `scripts/build_record_source_evidence_worklist_v4.py`, with ten synthetic fail-closed scenarios in `tests_v4/record_source_evidence_synthetic_test.py` and a separate real-data integration check in `tests_v4/record_source_evidence_real_integrity_test.py`. These tests constitute **research data governance engineering**, not a model experiment or evidence that DynaCPI v4 outperforms calibration-only adaptation.

### 7.17 Frozen geometry main effect versus centered support adaptation (6 November 2026)

Earlier fixed-K model B already mixed frozen sequence and geometry means using w0=0.5, so D-minus-B isolated *additional centered support-conditioned differential adaptation* and did not measure the entire effect of geometry. To explicitly decompose the effects without refitting checkpoints or tuning hyperparameters, we implemented five matching variants: S0 (sequence only), S_B (sequence-only calibration), G_A (sequence+geometry mixture), G_B (geometry mixture+calibration), and G_D (full geometry-mixture+calibration+differential adapter). For each of the existing three geometry initialization seeds 1105/1106/1107, we reused the four DAVIS geometry-checkpoint-selected development proteins, exactly ten fixed query identities per target, ten nested label-blind support draws per target, and K in 0,1,2,5,10,20. This gave 720 correlated target-draw-K evaluations across three seeds, **not** 720 independent targets. G_A/G_B/G_D reproduced their earlier same-seed frozen K-sweep macro values to within 6.16e-10 RMSE.

At the engineering support size **K=5**, S_B target-macro RMSE was **0.857337** on each seed. G_B target-macro RMSE was **0.860139** (seed 1105, +0.002801 worse), **0.855587** (1106, −0.001750 better), and **0.854164** (1107, −0.003173 better), where negative G_B-minus-S_B is favorable. Exactly two of four development targets benefited in each seed, with different target-level signs and effect magnitudes: for ADCK4 alone, geometry-minus-sequence calibrated RMSE changed from +0.012490 to −0.007262 to −0.014007. At K=0, S0 macro RMSE was 0.858202 whereas geometry-mixture G_A varied from 0.861017 to 0.856443 to 0.855010, again showing initialization sensitivity.

By contrast, G_D-minus-G_B at K=5 was 5.20e-11, 4.21e-11, and 5.42e-9 macro RMSE, numerically negligible at the ~0.85 RMSE scale. Even at K=20 the magnitude remained below 2.5e-8. The adapter's weighted centered sequence-versus-geometry disagreement variance V was approximately 1.15e-11, 1.10e-11 and 1.10e-9 at K=5, tiny compared with the fixed differential regularization lambda_theta=0.1. This is strong evidence that the *currently configured differential estimator* is nearly inactive on these development episodes; it is not evidence that a redesigned estimator will improve predictive accuracy.

**Development-stage research decision:** No reproducible benefit from the incremental differential geometry term was observed under the existing fixed model settings; the geometry main effect was seed-sensitive and averaged over only four development-selected targets. This does not establish superiority over PerceiverCPI or sequence-only calibration, and any adjustment selected using these same checkpoint-selected proteins must remain clearly labeled as exploratory model development rather than unbiased validation. The independent exact-Kd scientific study remains blocked pending original-source adjudication, physical construct verification and a frozen family-disjoint benchmark. Frozen weights and original models were unchanged. Reproducibility: scripts/benchmark_geometry_factorial_dev.py, research/geometry_factorial_dev/factorial.json and research/geometry_factorial_dev/SUMMARY.md.

### 7.18 Publication-level evidence checks for six preliminary BindingDB assay tasks (6 November 2026)

Six fixed original review tasks, comprising 147 preliminary records, were compared with independently available publication and assay-registry metadata. Unlike the earlier 921/921 numerical lineage checks, the aim was to check whether the cited publications could support the **claimed experimental binding endpoint**. This is bibliography/assay-metadata research, not a per-compound primary table verification; no chemical identities, measured affinity values or source reactant identifiers were exported, and no human review decision was modified.

ANXA3, local assay 50023141_1 (48 source records), is a promising candidate for detailed review: the original 2025 study (DOI 10.1021/acs.jmedchem.4c02403, PMID 40013713) describes binding experiments and SPR supporting data, while an independent PubChem-derived assay registration (AID 2234142) describes recombinant human ANXA3 binding measured by SPR and likewise lists 48 results. Matching counts and experimental method support a coherent assay-level source lead, **but do not verify the original numerical source rows, constructs or Kd qualifiers**. No records were approved.

Three more tasks, RIO2 kinase domain (20), SRPK3 kinase domain (19) and IRAK3 kinase domain (24), trace to Davis et al. 2011 (DOI 10.1038/nbt.1990, PMID 22037378), whose original Supplementary Table 4 identifies binding Kd values in nM. The 63 preliminary task entries remain unverified against specific table measurements and physically assayed kinase-domain constructs. A phosphoglycerate kinase task has 18 records and is linked to the original 1998 NMR-based binding study (DOI 10.1021/jm970839y, PMID 9804684); the local assay's variable-pH context and exact individual endpoint derivation are still unresolved.

An important source-citation anomaly was found for PA-I galactophilic lectin (18 source records, assay 50018971_9): its assigned PMID 26804741 resolves to a 2016 review titled "Novel Strategies for the Treatment of Pseudomonas aeruginosa Infections", rather than the experimental report that originated the 18 binding measurements. A separate competitive LecA paper (PMID 27488655) concerns IC50-type inhibition data; that second publication cannot be silently substituted as primary source evidence for these original Kd-labelled records. The actual individual source measurements must be located independently.

Deterministic artifact scripts/verify_six_primary_source_leads_v4.py binds six exact task keys, original PMIDs and record counts to the immutable original human-review queue. The results in research/source_access_v4/six_task_publication_evidence.json and SIX_TASK_PUBLICATION_EVIDENCE.md provide bibliographic links and unresolved verification requirements without source numeric measurements. The source-integrity check is part of the engineering suite. **All 147 source records and all six tasks remain unapproved**; scientific training remains blocked until independent original-source measurement/construct adjudication.

### 7.19 Protein-only independent homology challenge of proposed assay review targets (6 November 2026)

As a stress test for future family-disjoint scientific evaluation, we independently realigned **all 435 unordered pairs among the 30 fixed source-review protein sequences**, using local Smith–Waterman alignment with BLOSUM62 (gap open −10, extension −0.5). Only original protein reference sequences were accessed: no compound identity, chemical structure, target-affinity value, support/query episode, model checkpoint or evaluation label entered the analysis. The original MMseqs2 clustering placed the 30 review protein sequences in 30 different connected components, based on an earlier all-4157-protein comparison.

A deliberately permissive local screening policy (at least 30% identity, 80% coverage of the shorter protein, 80 aligned residue pairs) flagged **two cross-component pairs**: CFTR versus ANXA3 (32.3% identity over 269 aligned residue pairs, 83.3% of shorter versus only 18.2% of longer sequence), and phosphodiesterase δ (PDE6D) versus ENPP1 (32.1% identity over 134 pairs, 89.3% of shorter versus only 14.5% of longer). Those values are insufficient to certify shared family membership: a long protein can spuriously align locally to a shorter, otherwise unrelated sequence, and biological domain/homology verification remains necessary.

Under all more stringent controls tested—40% identity with 70% shorter coverage and 80 aligned pairs; 50% identity with 80% shorter coverage and 100 aligned pairs; 80% identity with 90% shorter coverage and 80 aligned pairs; and reciprocal policies requiring at least 50% coverage of the longer sequence—**zero direct pairs** among these 30 sequences passed. The 30 original MMseqs assignments therefore show no strong *direct* contradictory similarity under those policies, although MMseqs2 sensitivity, local sequence/domain diversity and transitive relationships through proteins outside this 30-target sample still require scientific evaluation.

The result is a **protein-leakage policy sensitivity finding**, not an experimental proof of independent folding/family groups or authorization to freeze a split. The two weak cross-component matches demonstrate why blindly joining proteins at one permissive shorter-coverage threshold could mistakenly reduce independent families, whereas a very strict threshold could overlook true shared domains. A defensible predeclared benchmark policy should include domain/profile evidence and source-verified *experimental constructs* before assigning independently held-out roles.

The source-coded audit `scripts/audit_30_protein_similarity_review_v4.py` and `research/source_access_v4/protein_only_review30_pairwise_audit.json` preserve all input hashes, algorithm settings, six prespecified identity/coverage policies, testable direct-edge counts and an accessible report `PROTEIN_ONLY_HOMOLOGY_CHALLENGE.md`. The dedicated test checks that no actual cohort or protein role split was generated. **Every one of the 30 source assays remains UNREVIEWED**, and no real BindingDB training or independent model-performance evaluation was run.

## 8 Discussion

The observed train–validation intersections affect how confidently one can interpret validation-selected checkpoints. They are compatible with zero train–test intersections. The operational question is whether an identical model input carries a repeated observation, a different measurement or a construct distinction missing from the CSV. Resolving that question requires raw identifiers and assay provenance. Simply dropping every repeated pair could erase meaningful experimental variance.

A positive geometry result requires more than an overall decrease in error after introducing ESM and GVP together. The appropriate contrast uses the same sequence foundation, same offset and same episodes. Even that contrast establishes predictive utility only for the tested cohort. Local chemical similarity can fail at activity cliffs, a static structure can miss relevant conformations, and restricting to structures with adequate coverage can change the evaluated target population. We therefore report eligible and all-task results separately. Attention weights and fitted geometry coefficients have statistical interpretations and do not establish binding poses or molecular causality.

## 9 Conclusions

The source and data audits establish concrete implementation differences and validation overlaps worth testing. Biological performance, external validity and the incremental value of structure-conditioned adaptation remain open. The first decisive result will be a paired comparison on independently held-out protein groups with identical support access.

## References

1. Nguyen NQ, Jang G, Kim H, Kang J. Perceiver CPI: a nested cross-attention network for compound–protein interaction prediction. Bioinformatics 39, btac731. https://doi.org/10.1093/bioinformatics/btac731
2. Lee E et al. MetaDTA: Meta-learning-based drug-target binding affinity prediction. MLDD workshop, 2022. https://openreview.net/pdf?id=yzlif16IASM
3. Eckmann P et al. Ligand-Based Compound Activity Prediction via Few-Shot Learning. JCIM, 2024. https://doi.org/10.1021/acs.jcim.4c00485
4. Wan M et al. A meta learning and task adaptive approach for drug target affinity prediction. Nature Communications, 2026. https://doi.org/10.1038/s41467-026-70554-5
5. Moura TJ, Cavalcanti GDC, Oliveira LS. MINE: A framework for dynamic regressor selection. Information Sciences 543, 157–179. https://doi.org/10.1016/j.ins.2020.07.056
6. Oelrich O, Villani M, Ankargren S. Local prediction pools. Journal of Forecasting. https://doi.org/10.1002/for.3030
7. Xu D et al. AssayRouter: Historical Utility Priors for Frozen Molecular Predictor Routing. Preprint. https://arxiv.org/abs/2609.37285
8. GVP-PyTorch implementation, commit 316eea7cd6446dd928e5975634152ec3b28e65f5. https://github.com/drorlab/gvp-pytorch
9. ESM-2 implementation and model documentation. https://github.com/facebookresearch/esm
10. PerceiverCPI implementation, commit 755f076ef51099a90332fd1ec8437f94e3750c0b. https://github.com/dmis-lab/PerceiverCPI

## Appendix A PerceiverCPI claim and observation register

Reported scores below belong to Nguyen et al. Each record identifies the location, present evidence and the comparison needed for independent assessment. Coverage concerns the main article. Assertions attributed to earlier literature are background. Appendix B covers supplementary experiments.

**C01 · Abstract / CPI relevance and docking cost.** Motivation. Measure actual inference and screening costs before drawing practical conclusions.

**C02 · Introduction / development time and cost.** Background cited by the authors. No economic outcome measured.

**C03 · Introduction / continuous affinity.** Regression matches the intended endpoint. Verify measurement type and units.

**C04 · Introduction / fingerprints versus learned graphs.** Proposed complementarity. Compare graph-only, fingerprint-only and fused models.

**C05 · Introduction / pooling and concatenation.** Information loss and inadequate interaction modeling are hypotheses. Match width and parameter count.

**C06 · Contributions / priority of nested attention.** Sequential attention is source-supported. Historical priority requires separate literature evidence.

**C07 · 2.1.1 / chemical representation.** Morgan and directed graph branches appear in code. Pin radius, feature ordering and chirality.

**C08 · 2.1.1 / reverse messages.** Inspected directed update subtracts the reverse bond message. Test index maps.

**C09 · 2.1.1 / averaging and stability.** Code sums in bond updates and averages padded messages at atom readout. Stability benefit remains untested.

**C10 · 2.1.1 / single-head superiority.** Reported comparison requires head-count ablation with equal tuning.

**C11 · 2.1.1 / fusion orientation.** Fingerprints query graph features. Residual compound fusion precedes self-attention.

**C12 · 2.1.2 / sequence representation.** Token embedding, convolution, residual GLU and pooling appear in source. Audit actual axes and truncation.

**C13 · 2.1.2 / residual normalization benefits.** Optimization rationale. Measure stability across seeds.

**C14 · 2.2 / compound–protein interaction.** Compound queries pooled protein keys/values. Channels do not provide direct residue assignments.

**C15 · 2.2 / complexity.** Channel score storage scales as squared per-head width. Complexity depends on the axis being varied.

**C16 · 2.3 / optimization.** MSE, LAMB and Bayesian search are reported. Source additionally weights errors by label range.

**C17 · 2.4 Table 1 / DAVIS dimensions.** Published 442 targets, 68 compounds, 30,056 interactions. Fold subsets and sequence hashes count different units.

**C18 · 2.4 Table 1 / KIBA dimensions.** Published 229 targets, 2,068 compounds and 117,657 interactions. Corpus-level reconciliation pending.

**C19 · 2.4 Table 1 / METZ dimensions.** Published 170 targets, 1,423 compounds and 35,259 interactions. Endpoint provenance pending.

**C20 · 2.4 Table 1 / PDBbind counts.** Prose gives 6,689 unique pairs and table gives 6,989 interactions. Resolve counting definitions from source data.

**C21 · 2.4 / affinity transformations.** DAVIS has a large mass at five, also observed locally. Original qualifiers remain necessary.

**C22 · 2.4 / density weighting.** Code supports label-range weighting. Improvement requires a paired loss ablation.

**C23 · 2.4 Tables 2–3 / GPCR and decoys.** Published GPCR training and DUD-E evaluation use different tasks from exact-Kd regression. Screening reproduction pending.

**C24 · 3.1 / regression baselines.** Paper describes adapting classification outputs/losses. Verify each baseline configuration.

**C25 · 3.1 / novel pairs.** Local fold zero has disjoint exact compounds and sequences across train/test on all three datasets. Homology and scaffolds pending.

**C26 · 3.1 / novel-hard pairs.** Reported threshold below 0.3. Reconstruct the similarity definitions and fold manifests.

**C27 · 3.1 / novel compounds.** Separate compound-disjoint DAVIS setting. Specific folds still required.

**C28 · 3.1 / novel proteins.** Separate protein-disjoint DAVIS setting. Evaluate sequence and family overlap separately.

**C29 · 3.1 / cross-domain split.** Train DAVIS, test PDBbind after overlap removal. Endpoint matching remains to be audited.

**C30 · 3.1 / metrics.** MSE measures error; concordance measures ordering. Reproduce tie rules. New study uses cluster-macro RMSE.

**C31 · 3.1 / five folds and validation.** Paper reports five folds and internal 80/20 split. Local validation overlap is documented in Table 1.

**C32 · Table 4 / novel pair.** Reported DAVIS MSE 0.463 ± 0.013 and concordance 0.638 ± 0.028. Five-fold reproduction pending.

**C33 · Table 4 / novel compound.** Reported MSE 0.378 ± 0.010 and concordance 0.726 ± 0.017. Pending.

**C34 · Table 4 / novel protein.** Reported MSE 0.667 ± 0.018 and concordance 0.758 ± 0.010. DeepDTA concordance 0.759. Paper qualifies significance.

**C35 · Table 5 / novel-hard pair.** Reported MSE 0.701 ± 0.244 and concordance 0.609 ± 0.072. Between-fold variability matters.

**C36 · Table 6 / KIBA.** Reported MSE 0.522 ± 0.010 and concordance 0.638 ± 0.013. Keep KIBA score distinct from pKd.

**C37 · Table 6 / METZ.** Reported MSE 0.658 ± 0.016 and concordance 0.675 ± 0.012. Common-split reproduction pending.

**C38 · Table 7 / cross-domain results.** Reported MSE 4.612 and concordance 0.532. Authors acknowledge poor absolute transfer.

**C39 · 3.1 / stability.** Lower variance depends on setting. Table 6 KIBA MSE SD is 0.010 versus 0.009 for DeepConvDTI.

**C40 · Table 8 / screening.** Reported GPCR and Diverse EF1% of 16.366 and 1.88, BEDROC of 0.236 and 0.031. Target-level reproduction pending.

**C41 · 3.1 / docking caveat.** Authors report stronger docking performance on most targets across metrics. Audit supplementary S7 independently.

**C42 · 3.2.1 / comparison with Perceiver IO.** Protein K/V orientation appears in source. Comparative gain needs matched orientation ablation.

**C43 · Conclusion / benchmark superiority.** Strongest evidence is setting-specific. Novel-protein and docking caveats qualify the broad abstract language.

**C44 · Future work / AlphaFold.** Original authors proposed protein 3D. Our test concerns incremental predictive benefit and its adaptation.

**C45 · Future work / meta-learning.** Prior paper proposed limited-data learning. Compare with established few-shot methods.

**C46 · Future work / ligand 3D.** Original paper suggests geometry from SMILES. Present design uses protein geometry only.

**C47 · Future work / transfer learning.** Frozen ESM uses pretrained sequence features. Separate its contribution from geometry.

**C48 · Limitations / interpretation.** Pooled channels limit atom/residue interpretations. New attention requires independent validation too.

**C49 · Availability / supporting information.** Official code is pinned. Data sources and supplementary comparisons are listed in Appendix B.

## Appendix B Supplementary claims and tests

**S01 · Figure S1.** Directed messages carry atom and bond information and suppress immediate reversal. The pinned bond-update code supports the described directionality.

**S02 · Weighted loss, Eq. S1.** The supplement specifies weights 0.5 inside and 5 outside [0,5] on DAVIS and [11.1,12] on KIBA. Code inspection confirms the conditional weighting operation. Its marginal predictive benefit needs a loss ablation.

**S03 · Figure S2.** The illustrated distributions motivate a claim about predictions remaining above the DAVIS floor. Section 7.1 directly checks this in the archived local model. Marginal distribution agreement alone does not identify which compound–protein interactions a model learned.

**S04 · Figure S3.** The five-fold plots report stronger novel-pair and novel-compound performance and competitive novel-protein performance. Reproduction requires the complete fold series.

**S05 · Tables S1–S3.** On novel GPCR pairs the reported AUC, precision, recall and accuracy are 0.605, 0.607, 0.544 and 0.590. The DUD-E cohorts contain five GPCR and seven Diverse targets. These are classification and screening endpoints. No DynaCPI classification comparison has been run.

**S06 · Table S4.** The first-fold complete model reports DAVIS MSE/concordance of 0.442/0.685, 0.358/0.751 and 0.660/0.755 for novel pairs, compounds and proteins. Replacing both CABs yields 0.844/0.543 on novel pairs. Removal of CAB-2 performed better than removal of CAB-1 in these reported comparisons. These single-fold ablations motivate retaining both blocks. They do not quantify the incremental benefit of our residue readout.

**S07 · Table S5.** Zeroing atom-number features produced a larger loss of accuracy than several other feature removals. Removing formal charge on the novel-protein task left the reported MSE unchanged and raised concordance slightly. The evidence therefore depends on feature, metric and setting.

**S08 · Table S6.** The authors divide aligned amino acids by a specified protein length of 500 and use compound Tanimoto similarity. Their cross-domain maxima reach one for both modalities. Exact pair separation can coexist with component overlap. Our full-construct homology and scaffold audits test distinct notions of separation.

**S09 · Table S7.** Screening performance varies by target. PerceiverCPI reports EF1%/BEDROC of 43.13/0.51 on DRD3 and 0/0 on AMPC. The target-level data qualify the family averages in main Table 8. Prospective ranking utility remains unmeasured in DynaCPI.

**S10 · Data availability.** The supplement identifies DeepPurpose for DAVIS/KIBA, KinasepKipred for METZ, MONN for PDBbind, TransformerCPI for GPCR training and DUD-E for screening. The next provenance check is to trace local split generation to those specific source releases and original construct identifiers. These links alone do not verify the local labels.