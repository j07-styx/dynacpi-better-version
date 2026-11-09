# DynaCPI project handoff

## Independent protein-only split-leakage challenge completed — 6 November 2026

This phase tested **protein-family split methodology on the actual source-review target sequences**, without accessing drugs, compounds, affinity measurements or source-reviewed labels. Local script \`scripts/audit_30_protein_similarity_review_v4.py\` loads the original 4,157-sequence protein-only FASTA and the immutable 30-task review header. It checks all 435 pairwise local protein sequence alignments using Biopython Smith-Waterman, BLOSUM62 and affine gaps. Input sources are SHA-256 verified; zero human-review decisions altered. The 30 reference sequences were originally mapped to 30 independent MMseqs2 components. This tool DOES NOT choose/assign train/validation/test roles or grant scientific approval.

**Important observed sensitivity:** A permissive 30%-identity / 80%-shorter-sequence-coverage / >=80-aligned-residue-pair local alignment policy flagged two across-existing-component pairs: CFTR versus ANXA3 (32.3% of 269 aligned residue pairs, 83.3% coverage of shorter but 18.2% coverage of longer protein) and PDE6D versus ENPP1 (32.1% of 134 aligned pairs, 89.3% shorter but 14.5% longer coverage). These may be artifacts of long-vs-short local alignment and DO NOT establish shared family/function. All four stricter test policies requiring at least 40% identity or reciprocal >=50% coverage of the longer sequence found **zero direct pair conflicts** among the 30 reviewed proteins. Thus the current MMseqs group assignment was not directly contradicted under stricter thresholds, but biological domain/homology review is still mandatory before freezing an independently held-out benchmark split.

New source evidence: \`research/source_access_v4/protein_only_review30_pairwise_audit.json\` and \`PROTEIN_ONLY_HOMOLOGY_CHALLENGE.md\`. Integrity: \`tests_v4/protein_homology_challenge_integrity_test.py\`, which passed its direct run and was added to the consolidated validation suite (full engineering suite completed **28/28 PASS** with zero failed tests on 6 November 2026). Scripts are read-only relative to original proteins and review sheet. No approved source tasks, no frozen scientifically independent split, no new trained model and no change to original DynaCPI/PerceiverCPI checkpoints. Manuscript Section 7.19 explains the limits and possible source of false positive family merges.

**Next scientific step** is independent original-method/construct confirmation for the physically assayed proteins and legitimate source Kd labels. Then apply a predeclared protein-profile/domain/sequence family grouping with robust reciprocal-coverage checks to only source-approved tasks before any model training or final split. Do not retroactively choose the identity/coverage threshold to improve the small DAVIS development metrics.


## First six external paper-source checks across 147 BindingDB candidates — 6 November 2026

The scientific blocker was attacked directly: we crosschecked the exact bibliographic/source-assay identifiers of six fixed preliminary tasks against publicly available original publications and external assay registry metadata. `scripts/verify_six_primary_source_leads_v4.py` checks six immutable original task names, source IDs and record counts; artifacts `research/source_access_v4/six_task_publication_evidence.json` and `SIX_TASK_PUBLICATION_EVIDENCE.md` provide source links, measured-method evidence levels and outstanding original table/construct checks WITHOUT original Reactant_set_ids, chemical structures or individual numerical affinities.

**Real new evidence:** ANXA3 (48 preliminary records) has a 2025 original source paper with supporting SPR data plus independently described public assay metadata for recombinant ANXA3 dissociation constants (48 entries). Three separate kinase-domain tasks, RIO2 (20), SRPK3 (19), IRAK3 (24), have a shared original 2011 publication with supplementary experimental kinase-binding Kd table. However the assayed domain identity may differ from the full-length reference and individual table measurements are NOT yet verified. Phosphoglycerate kinase (18) has an original binding/NMR source with varying pH/measurement-derivation questions. **Serious citation provenance anomaly:** the PA-I galactophilic lectin task (18) cites PMID 26804741, a review article rather than the original experimental source; a different competitive-lectin paper cannot be substituted to justify Kd values. This is an evidence failure requiring original source retrieval, NOT a final scientific reject verdict.

**Total original records touched at source-metadata level:** 147 / 921 across six / thirty tasks. All remain UNREVIEWED; zero approved scientific cohorts, no frozen independent protein-family split and no new source-approved model evaluation. The source-identity integrity check `tests_v4/six_publication_source_evidence_integrity_test.py` passed and is included in the consolidated engineering suite, which completed **27/27 PASS** (zero failures) on 6 November 2026. Source paper work is independently documented in manuscript Section 7.18. Original DynaCPI v4 model/checkpoints and prior negative differential geometry DEV conclusions are unchanged. The source-level and per-reactant evidence private worksheet must not be conflated; the 921-record LOCAL_PRIVATE worksheet is NOT uploaded to GitHub.

**Actionable next scientific step:** compare ANXA3's original primary SPR table and full physical protein construct to the 48 individual source records, and match the original 2011 kinase binding supplementary table to the 63 kinase-domain records. The PA-I paper-citation mismatch must be resolved separately. Only after verified measurement-to-record concordance, experimental constructs and reviewer approval can the research science gate be reconsidered.


## Main architecture contrast dissected using real frozen DAVIS development predictions (6 November 2026)

The project has now run a new **model experiment** rather than another source metadata audit. The script scripts/benchmark_geometry_factorial_dev.py performs a fixed five-variant factorial evaluation of S0 (sequence only), S_B (sequence+support calibration), G_A (geometry mixture uncalibrated), G_B (geometry mixture+support calibration) and G_D (full geometry differential support adapter). It uses existing frozen trained sequence and geometry models from three seeds 1105,1106,1107, four preselected DAVIS development targets, 10 query compounds per target and ten fixed nested label-blind support draws for K=0,1,2,5,10,20. Total 720 correlated target-draw-K evaluations; **only four development targets**, so no claim of 720 independent observations. Entire original v4 architecture and model checkpoints remain UNCHANGED.

The earlier v4 comparator B already included geometry in its base prediction (w0=0.5). Thus the prior D≈B result isolated the incremental *centered differential adapter* but was insufficient to assess whether geometry itself contributed. The new sequence-only-calibrated S_B baseline directly measures the geometry-mixture effect as G_B−S_B.

At K=5 the frozen S_B macro RMSE was **0.857337**. Geometry-mixture calibrated G_B macro RMSE was **0.860139** (seed 1105, +0.002801 worse), **0.855587** (1106, −0.001750 better) and **0.854164** (1107, −0.003173 better). Each seed improves only 2/4 targets; the target-level effect changes sign strongly for ADCK4. At K=5 the incremental full-geometry differential effect G_D−G_B was just **5.20e−11, 4.21e−11, 5.42e−9** RMSE across the seeds. Even at K=20 it stayed below ~2.5e−8. Weighted centered disagreement variances V were tiny compared with lambda_theta=0.1. The original G_A/G_B/G_D K-sweep results were reproduced within **6.2e−10 macro RMSE**, confirming the factorial uses the same checkpoint and episodes.

**Scientific decision:** Under the current fixed v4 architecture/regularizers the support-dependent differential geometry adaptation is effectively inactive even on selected development data; geometry mixture has an initialization-sensitive and target-heterogeneous main effect. **No reliable demonstrated incremental gain** and no defensible claim of superiority over PerceiverCPI. Do not retune this 4-protein validation set and call that generalization. A real scientifically approved, family-disjoint exact-Kd benchmark is still BLOCKED pending independent original-source and physical protein construct review. Distinguish source-data numerical lineage checks (25 engineering passes before this experiment) from the newly computed legacy DAVIS model diagnostics.

Outputs: research/geometry_factorial_dev/factorial.json and SUMMARY.md, test tests_v4/geometry_factorial_dev_integrity_test.py and 26-check consolidated engineering suite. Next actual scientific milestone is sufficiently many source-approved Kd tasks across unrelated protein components with an immutable train/adapter/validation/test split, followed by matched D-vs-B and full-model-vs-sequence calibration paired evaluation. Current architecture not changed; model weights remain the three frozen development checkpoints.


## Per-reactant source-evidence review worksheet — 6 November 2026

- **First actionable source-evidence approval workflow (NO actual scientific approvals):** A new `scripts/build_record_source_evidence_worklist_v4.py` creates/validates `data/audit/source_record_evidence_worklist_30_LOCAL_PRIVATE.csv` on the original PC. It contains **921 source reactant rows from exactly 30 original UNREVIEWED BindingDB tasks**, keyed by immutable `task_id`, `assay_key`, `Reactant_set_id`, protein component and canonical sequence hash; metadata source identifiers include source PMID, DOI, patent, PubChem AID, organism and curator. Crucially the private worksheet contains **NO numeric affinity values and NO SMILES or chemical structures**.
- **No automatic reviewer decisions:** Every one of the 921 rows is `UNREVIEWED`. All human-evidence fields start blank, including method locator, source table/figure and exact record correspondence, physically assayed target, construct description and sequence hash, method, endpoint, qualifier, assay compatibility, reviewer identity/date, and reason. The existing 30-task canonical review spreadsheet and raw source files were SHA-256 protected and not changed. An existing worksheet is **never overwritten** by running the generator again.
- **Strict evidence declaration checks:** The validator refuses duplicated/missing reactant IDs and changed source metadata, any `UNREVIEWED` entry claiming verified YES or a reviewer signoff, missing reviewer identity/date/justification, incomplete `APPROVE` source/table/physical sequence assertions, endpoints other than Kd or non-EXACT qualified values. Structurally complete declarations are reported as **SELF-REPORTED ONLY** and can never themselves authorize scientific training or establish actual source verification. Real scientific approval still requires independent inspection of the primary source.
- **Public/committed output carefully limited:** `research/source_access_v4/record_source_evidence_readiness_30.json` and `RECORD_SOURCE_EVIDENCE_READINESS.md` contain source file hashes, counts, and per-task numerical review progress only. They exclude source `Reactant_set_id`, compound identities, and raw Kd values. `data/audit/source_record_evidence_worklist_30_LOCAL_PRIVATE.csv` is **local-only and explicitly Git-ignored**, never added to the GitHub handoff.
- **Current status:** **921/921 UNREVIEWED, 0/30 source-reviewed, 0 scientific approvals, no approved BindingDB cohort, no frozen independent protein-family split, and zero permitted scientific training.** Previous 921/921 technical lineage check passed with zero processing errors; this latest workflow adds the missing experimental evidence declarations rather than fitting another model.
- **Engineering tests:** `tests_v4/record_source_evidence_synthetic_test.py` completed **10/10 PASS** on isolated fake examples, including fail-closed incomplete/incorrect Kd declarations. `tests_v4/record_source_evidence_real_integrity_test.py` passed on the actual private 921-row worksheet and checks source identity, all 30 task IDs, zero review approvals and the blocked scientific gate. Both are included in `scripts/run_v4_validation_suite.py` (now 25 checks total; verify the consolidated run's finished report before claiming 25/25).
- **VS Code and monitoring:** `.vscode/tasks.json` now exposes **12 runnable tasks**, including `DynaCPI: Validate private 921-record source review worksheet`. `scripts/monitor_benchmarks_v4.py` monitors the aggregate readiness report without reading or publishing the private source worksheet. The existing hourly notification automation remains read-only. Original DynaCPI v4 architecture, checkpoints, previous negative DEV results and source review CSV are unchanged.
- **Next science action:** A qualified independent reviewer uses the private local record worksheet to locate original paper/patent table/figure/method for each raw record, verify the actual experimental protein construct (including mutations, tags and chain participants), exact equilibrium Kd and qualifier, and intra-assay compatibility. Supported decisions may then enter the canonical human review process and a separately approved source evidence manifest. Only after enough unrelated source-approved protein groups exist may a family-disjoint split be frozen and real D-vs-B research begin.


## Real 30-task BindingDB raw-record lineage audit — 6 November 2026 (Windows connection restored)

- **Device and execution:** `built-diff12` was initially intermittently unreachable, then successfully responded to a fresh Windows PowerShell command. The real provenance audit `scripts/audit_30_record_level_provenance_v4.py` was installed in `C:\Users\J0124\dynacpi_v4` and executed successfully on the existing original BindingDB files. This is a REAL local dataset lineage check, not merely a synthetic or GitHub-only code review.
- **Verified input scope:** fixed thirty unreviewed human-review tasks, **919 grouped compound observations referencing exactly 921 distinct original BindingDB Reactant_set_id values**, including two two-source-record groups (targets PDE6D and ENPP1). All **921/921 original source rows** and **921/921 unaggregated candidate rows** were found. No duplicated raw source or preliminary candidate reactant IDs in this fixed subset.
- **Technical numeric and provenance checks:** All 921 source Kd fields were strictly positive numeric and their `pKd = 9 - log10(Kd in nM)` preliminary candidate conversions were within 1e-7. All checked candidate sequence hashes matched the task's original canonical sequence hash; assay keys and review record/group counts agreed. The median and range aggregation of both multi-measurement grouped compounds matched the unaggregated source values. **Zero critical technical flags** were reported. `research/source_access_v4/record_level_provenance_30.json` and `RECORD_LEVEL_PROVENANCE_30.md` store only source-file hashes, per-task aggregate counts and technical warning totals (no compound affinities, individual raw Kd, structures or source reactant IDs).
- **Scientific boundaries remain STRICTLY unchanged:** technical parity of numeric source records DOES NOT establish that a reported source field truly reflects direct equilibrium Kd for the intended target construct, species, complex or source assay. All original 30 task decisions remain `UNREVIEWED`, zero human-approved and zero scientifically approved. No source-approved BindingDB cohort or frozen independent protein-family split exists. Do not release the main BindingDB predictor training or interpret these checks as D-versus-B model performance.
- **Engineering tests:** `tests_v4/record_lineage_synthetic_test.py` has **5/5 isolated synthetic unit cases PASS**: valid multi-record median, deliberately incorrect Kd/pKd conversion, missing original source record, qualified source Kd, and fail-closed changed human review status. `tests_v4/record_lineage_real_integrity_test.py` reruns the exact 30-task real-data audit and checks the blocked science gate. Both have been added to `scripts/run_v4_validation_suite.py` (now 23 checks total). The consolidated engineering run completed **23/23 PASS** with zero failures on 6 November 2026, including all original archived PerceiverCPI baseline parity tests. Evidence: `data/audit/v4_validation_suite.json` and the corresponding `research/validation_suite_logs/` output.
- **VS Code and hourly monitoring:** `.vscode/tasks.json` now includes 11 runnable tasks, including `DynaCPI: Audit 30-task raw BindingDB record lineage`; `scripts/monitor_benchmarks_v4.py` snapshots source reactant and candidate counts and critical technical flags, with hashes in its existing read-only artifact watch. Hourly monitor does NOT start training. The original DynaCPI architecture, model weights, pilot history and experiment results are unchanged.
- **Next step:** Independent qualified review of the original paper/patent record table, endpoint interpretation, exact experimental protein construct and source organism, measurement qualifiers and assay comparability, especially the previously documented CFTR, GLP-1R and source-fragment risks. Only then create a source-approved cohort and immutable independent family-disjoint split for the primary research question.


## P0 external source crosswalk and DOI-only source-index correction — 6 November 2026

- **Precise correction to earlier source-access triage:** The carbonic-anhydrase II task was described as having no usable primary source identifier because the initial index inspected PMID and patent fields but overlooked a populated **DOI** field. The DOI resolves in Crossref to a 1996 research publication, but its full experimental methods/individual affinity values remain unreviewed. This is now classified `PRIMARY_DOI_ONLY_METHODS_PENDING` rather than `MISSING_PRIMARY_SOURCE_IDENTIFIER`. Original source-review CSV is unchanged and prior historical reports should be read with this correction.
- **Four high-priority source-linked review leads:** `scripts/build_p0_source_crosswalk_v4.py` now creates `research/source_access_v4/p0_source_crosswalk.json` and `P0_SOURCE_CROSSWALK.md` from the immutable 30-task review queue and source records, retaining only structured method/registry observation summaries (no source body text or compound affinity labels). Source evidence categories: one CFTR publication abstract describing a functional cellular transport response rather than clearly identifying the intended direct Kd measurement; one GLP-1 receptor metadata/assay-registry record whose stated binding participant is glucose despite a protein target label; one DOI-only CA2 source bibliography; and one PD-L1 primary patent locator with missing assay description. These are **review hypotheses**, not definitive verdicts or scientific rejection decisions. No compound-to-source-table crosswalk was approved.
- **Cross-source guardrails:** The source index `scripts/audit_primary_source_access_v4.py`, review worklist generator, and metadata triage were updated and regenerated, with 20 unique PubMed metadata sources still resolved, two public methods section indexes, and the one explicit DOI-only task. The triage remains P0=4, P1=12, P2=14. No assay has been source-approved; no training cohort/frozen protein-family split exists.
- **New test:** `tests_v4/p0_source_crosswalk_integrity_test.py` asserts the exact four source-reviewed metadata leads match the original immutable source task IDs, provenance hashes, DOI correction, no numeric affinity values or approval decisions, and the still-blocked scientific cohort gate. The source access, reviewer worklist and prior metadata triage integrity tests remain passing. A full extended **21-check engineering suite** has been initiated and must be allowed to finish and its actual results verified before claiming 21/21.
- **VS Code and monitoring:** 10 VS Code runnable tasks, including read-only P0 evidence crosswalk regeneration, and observer `scripts/monitor_benchmarks_v4.py` now snapshots P0 source tiers, number of P0 records and 0 record-level approved/verified Kd tasks. Hourly notification automation only monitors significant changes when the PC is reachable.
- **Unchanged scientific gate:** 30/30 tasks remain UNREVIEWED and 0 scientifically approved. Neither the original DynaCPI v4 architecture/checkpoints nor the support-K geometry development metrics changed. Actual independent target-family exact-Kd study is still BLOCKED until qualified source/construct/record adjudication and immutable family-disjoint role split.


## Label-independent metadata-risk triage for 30 BindingDB candidate tasks — 6 November 2026

- **New 30/30 deterministic source-METADATA review triage, NOT experimental approval:** `scripts/triage_30_assay_metadata_v4.py` consumes the existing unmodified enriched thirty-task reviewer queue, canonical sequence-reference screen, publication/patent access index and two source-fragment locator flags. The script NEVER reads individual compound affinities, changes original review decisions, approves/rejects experiments or creates a training cohort. Output: `research/source_access_v4/assay_metadata_risk_triage_30.json`, `.csv` and `ASSAY_METADATA_TRIAGE.md`, plus original-review SHA256 and source-screen hashes.
- **Metadata-review priorities (not experimental defect verdicts):** P0 (ambiguous endpoint, absent description or absent source identifier) **4 tasks**; P1 (fragment, complex or experimental endpoint interpretation risk) **12**; P2 (standard mandatory primary-source review) **14**. All 30 tasks remain UNREVIEWED and 0 source-approved. P0 examples: CFTR Kd field described as “half effective concentration”; GLP-1 receptor assay metadata discussing a glucose-based colorimetric signal; PD-L1 absent description; carbonic anhydrase II lacking PMID/patent metadata but possessing a DOI; the later DOI-only correction at the top of this file supersedes the initial classification. These warnings are not proof the Kd values are wrong.
- **Overlapping metadata flags:** six candidate descriptions include numeric protein regions shorter than full canonical sequence references; eight mention tagged or labeled reagents; four use potentially surrogate activity-endpoint wording; four mention competition/displacement; four mention protein-complex/cofactor topology; two have published method-section fragment risks from Section 7.12 of the manuscript. Numeric region `54 to 213 residues` in a VHL/Elongin complex is for VHL, not necessarily Elongin-C; the triage explicitly warns `NUMERIC_REGION_END_EXCEEDS_TARGET_REFERENCE_CHECK_CHAIN_ASSOCIATION` and MUST NOT silently map it to the named canonical target. These are non-exclusive indicators; the absence of a flag does not verify an assay.
- **Regression and safeguards:** `tests_v4/metadata_risk_triage_integrity_test.py` checks parser edge cases (e.g., assay residue spans vs pH 3.0–11.0), every source task identity and the frozen reviewer-queue SHA256, no scientific approval or compound-level labels, explicit ambiguity for CFTR/GLP1R/VHL and continuing blocked scientific cohort gate. Added to `scripts/run_v4_validation_suite.py`; the full engineering suite completed **20/20 PASS**, zero failed, on 6 November 2026, with detailed result logs and hashes in `data/audit/v4_validation_suite.json`. The original model checkpoints, prior 3-seed development results and empirical null D-vs-B findings are unchanged.
- **Operational monitoring and VS Code:** `.vscode/tasks.json` now includes a ninth task to reproduce the metadata-only triage. `scripts/monitor_benchmarks_v4.py` tracks its risk flag counts, P0/P1/P2 counts and source-approval status via read-only artifact snapshots. Existing hourly monitor remains enabled; it never initiates training.
- **Next scientific step:** Source reviewers must inspect primary literature/patents and per-compound experimental Kd records, establish actual constructs, assay type/qualifiers and table correspondence, reconcile source region indices to physical sequence/structure, then independently approve enough unrelated protein groups. No main BindingDB training or blinded study until source evidence and frozen five-way independent protein-cluster split satisfy the scientific gate. Strong metadata triage warnings should be investigated, not automatically discarded.


## Primary-source construct-alignment flags — 6 November 2026 (after source access audit)

- **Source-localized experimental region differences identified for two of thirty UNREVIEWED assay tasks:** BindingDB candidate task `69f76b68cfe450d9847cbcf66f2aacfaf9059b740d9266ab6f68052affd644a8` has exact full-length UniProt P17342 reference length **541 residues**, whereas original publication PMID 35333039 / PMCID PMC9014859, Methods `sec4.4`, reports an NPR-C construct covering **residues 27–541** (nominal 515 residues) with a terminal tag for SPR affinity measurements. The assay description mentions this region but the BindingDB candidate record stores the full reference sequence. Physical construct sequence and individual measurement provenance remain unverified.
- Task `83775558aead8993bb3100cbac91d08f9d2879c15bad712388ba49dd380c0a59` has exact full-length UniProt O14786 reference length **923 residues**, while publication PMID 29648813 / PMCID PMC5957473, Methods `sec4.2.1` and `sec4.2.3`, reports the **NRP1-b1 region 273–427** (nominal 155 residues) for a tagged protein and SPR work. The paper narrative also references b1b2, so individual measurement-to-construct compatibility requires verification. Neither the canonical 923-residue string nor a full-length structure should be silently treated as the assayed 155-residue fragment.
- **Scientific interpretation:** These are evidence-backed *source-region versus metadata-reference* discrepancies, not findings that any specific Kd measurement is wrong. Exact amino acid construct, processing, tags, mutations, compound-level source associations and endpoint qualifiers remain **UNVERIFIED**; both task decisions remain **UNREVIEWED**, zero new approvals.
- **Evidence and fail-closed integrity:** `research/source_access_v4/PRELIMINARY_CONSTRUCT_ALIGNMENT_REVIEW.md` and `construct_alignment_risk_screen.json` preserve source PMCID/section locators, canonical reference lengths and fragment spans without copying article bodies or extracting affinity values. `scripts/check_primary_construct_flags_v4.py` checks against the original human review CSV SHA-256, the prior UniProt reference metadata screen and Europe PMC source access index; `tests_v4/construct_alignment_risk_integrity_test.py` checks fixed risk flags and the continuing blocked science gate. The actual content was inspected at the cited paper sections; the automated verifier only checks known locators and metadata, **not** biological approval.
- **Execution boundary:** Do not start source-approved BindingDB model fitting or change architecture based on these two illustrative risk flags. First obtain individually reviewed exact Kd source/construct records from enough unrelated protein groups, finalize physical sequence mapping and independent split, then train matched baselines and validate differential support prediction. Source-provenance remediation is a prerequisite to valid geometry claims.


## Primary-source access and review worklist checkpoint — 6 November 2026

- **Scientific status unchanged:** zero BindingDB source-approved exact-Kd tasks, no verified full assay constructs, and no frozen independent protein-group benchmark roles. The fail-closed `dynacpi/scientific_cohort_gate.py` remains BLOCKED. No main scientific training or final test has been run in this stage.
- **30-task primary-source access audit completed without changing review decisions:** new `scripts/audit_primary_source_access_v4.py` queried the official Europe PMC public metadata API for the 20 distinct PubMed identifiers in `data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv`, and *transiently* inspected openly accessible article XML to index only method-related section IDs/headings and keyword counts (no article body copied). Exactly 20/20 publication IDs matched official metadata. Two open-access articles were successfully indexed; 17 returned metadata without public OA full text through this API and one OA XML was deliberately blocked for containing DTD/entities (requires an alternate safe or manual review route). The task-level access distribution is **2 OA-indexed, 20 requiring separate article access, 7 requiring primary patent review, 1 missing usable source identifier**. These are **source access statuses**, not outcome or assay approvals.
- **Immutable scientific review CSV preserved:** the original review file was hashed before/after the entire source-indexing process and had identical SHA-256. All 30 records remain UNREVIEWED, all verification fields unverified. Source access evidence and generated provenance: `research/source_access_v4/source_access_index.json` and `SOURCE_ACCESS_SUMMARY.md`.
- **Read-only human worklist created:** `scripts/build_primary_source_review_worklist_v4.py` groups the 30 tasks by access category and records exact task IDs and section-location hints, with construct, measurement semantics and assay compatibility explicitly UNKNOWN. `research/source_access_v4/REVIEW_ACTIONS.md` plus `worklist_manifest.json`. It does not output or change raw affinity labels, infer approvals, or bypass the human curation gate.
- **Engineering integrity:** `tests_v4/source_access_index_integrity_test.py` verifies original 30 tasks/20 indexed PMIDs, zero approvals, source CSV hash agreement, raw article body not redistributed, test XML method-section extraction and unsafe DTD/entity rejection, unchanged official review decisions, and scientifically blocked cohort status. Added to `scripts/run_v4_validation_suite.py` as the next regression check. Do not equate software PASS with biological approval.
- **VS Code / monitoring:** `.vscode/tasks.json` now has seven runnable tasks, adding a source-index refresh, read-only review-worklist generation and a source-approval status check. `scripts/monitor_benchmarks_v4.py` now tracks source access counts, review approvals/unreviewed count, and new review artifact hashes alongside existing engine benchmarks, GPU and 3-seed development analysis.
- **Next genuinely scientific step is NOT automated label promotion:** an authorized qualified reviewer must inspect the original articles/patents and supplements to independently verify the physical assayed construct, exact Kd semantics and consistent conditions. Unknown cases remain quarantined. After enough independent target families are source-approved, freeze approved cohort and protein-group roles, then matched training and D-vs-B evaluation. Preserve all negative DAVIS development findings and don't retune on checkpoint-selected validation targets.


## New execution checkpoint — 6 November 2026 (VS Code, benchmark monitoring)

- **Workspace:** `C:\Users\J0124\dynacpi_v4` on online Windows device `built-diff12`. Opened with VS Code and added `.vscode/tasks.json` with commands to run the full engineering suite, support-K sweep, benchmark status snapshot and scientific approval gate; the old upstream PerceiverCPI git checkout is preserved. GPU available: RTX 4060 Laptop 8 GiB.
- **Current architectural specification:** DynaCPI v4, two frozen sequence/geometry predictor heads plus analytical support calibration/differential adapter. Not a newly proposed four-expert MoE. No change to math, model weights or original training history in this continuation.
- **Frozen-checkpoint real-label DEVELOPMENT support-K sweep completed:** new `scripts/benchmark_support_k_sweep_v4.py` evaluated K=0,1,2,5,10,20, ten deterministic nested support draws per K, ten label-independently selected queries, four geometry-checkpoint-selected legacy DAVIS validation targets (ADCK4, FRK, MEK5, OSR1). Same previously saved sequence epoch-3 and GVP epoch-1 checkpoints; no training or hyperparameter tuning. Macro B/D RMSE: K0 0.86101741/0.86101741; K1 0.86079179/0.86079179; K2 0.86061335/0.86061335; K5 0.86013856/0.86013856; K10 0.85936112/0.85936112; K20 0.8573889581/0.8573889583. At K20 D−B = +2.4608e-10, no practically meaningful differential benefit. The K5 numbers reproduce the prior archived development diagnostic. `research/support_k_sweep_dev/results.json`, `aggregate.csv`, `SUMMARY.md`.
- **Mechanistic diagnosis:** `scripts/summarize_support_identifiability_dev.py` reports mean kernel sum W=0.028842 and centered variance V=6.50e-11 at K20; compared with fixed lambda_b=5 and lambda_theta=.1, W/(W+lambda_b)=0.005735 and V/(V+lambda_theta)=6.50e-10. These are descriptive regularizer scale ratios, not exact theta or proof changing hyperparameters will help. All analyzed data are DEVELOPMENT-ONLY on four targets that were also used for geometry checkpoint selection. See `research/support_k_sweep_dev/IDENTIFIABILITY.md`.
- **Automated checks:** added `tests_v4/k_sweep_result_integrity_test.py` to verify K5 consistency, actual saved checkpoint hashes, K0 fallback and support K coverage, now included in `scripts/run_v4_validation_suite.py`. The latest 6 November suite rerun includes the original K-sweep and auxiliary-seed checks: **17/17 PASS**, zero failures, with results in `data/audit/v4_validation_suite.json`.
- **Local monitoring:** `scripts/monitor_benchmarks_v4.py` writes `research/monitoring/latest.json`, `LATEST.md`, and change events; reports checkpoint signatures, updated engineering metrics, GPU snapshot and source-approval gate. An hourly notification automation has been enabled to check for meaningful newly completed benchmarks or failures when the PC is reachable; it never starts training or claims scientific improvement.
- **Outstanding scientifically blocking work:** 0 source-approved exact-Kd BindingDB tasks and no final frozen protein-family split. Do not run the headline BindingDB experiment until primary source/construct/assay evidence is adjudicated and an independent split is locked. Next unblocked work: source-level provenance curation, leakage guards, honest development failure analysis and improved geometry complementarity tests on pre-specified development data. Preserve all negative results and do not tune on held-out test.
- **GitHub:** Snapshot repository `https://github.com/j07-styx/Demo-smth` is separate from this upstream clone. Export new code/docs as a new explicitly dated research continuation commit once checks pass. Git upload milestone labels are not architecture versions.

## Additional fixed-seed geometry and adaptation diagnostics — 6 November 2026

- **Three geometry-training initializations were checked without changing the model design:** original seed 1105, plus additional predeclared seeds 1106 and 1107. The two new runs are isolated in `research/pilot_geometry_seed_sweep_dev/seed_1106` and `seed_1107`; no original checkpoint was overwritten. Same 9 training proteins (486 DAVIS pairs), 4 checkpoint-selection validation proteins (216 pairs), frozen sequence foundation, two epochs, identical optimizer settings. Best validation target-macro geometry RMSE by seed: 1105=0.93592060 (epoch1), 1106=0.92004961 (epoch2), 1107=0.91597119 (epoch1), compared with frozen sequence=0.92568508. Seed effects range from geometry WORSE by +0.0102355 to apparent lower DEV error by −0.0097139. All best epochs are selected on the SAME development proteins, so these are not independent or robust gains. `scripts/train_geometry_seed_sweep_dev.py`, `scripts/aggregate_geometry_seed_variability_dev.py`, `research/pilot_geometry_seed_sweep_dev/seed_variability.json` and `SEED_VARIABILITY.md`.
- **A/B/C/D support-K re-evaluated for each alternate geometry seed without tuning:** `scripts/benchmark_support_k_alternate_geometry_dev.py` uses identical 10 queries per target, 10 nested support draws, K=0,1,2,5,10,20 and fixed adapter settings. At K20 macro RMSE B and D: seed 1105=0.8573889581/0.8573889583 (D−B +2.46e−10), seed 1106=0.8529299312/0.8529299314 (D−B +2.00e−10), seed 1107=0.85154725/0.85154728 (D−B +2.48e−8). **No material incremental geometry differential effect** across all three seed checkpoints, even though the zero-shot geometry-based predictions are initialization-sensitive. See `research/pilot_geometry_seed_sweep_dev/adapter_seed_comparison.json`, `ADAPTER_SEED_COMPARISON.md`, and `research/support_k_sweep_seed_dev/seed_1106`, `seed_1107`.
- **Consolidated validation suite:** `scripts/run_v4_validation_suite.py` completed **17/17 PASS** on 6 November 2026, zero failed, including the new seed and K-sweep integrity checks. `data/audit/v4_validation_suite.json` contains the actual timestamps, scripts' source hashes and individual logs. The source approval gate remains BLOCKED, not scientifically ready.
- **Interpretation / strict boundary:** Stability of the seed-dependent geometry predictor is poor on four development kinases; the D-versus-B null effect is robust across three *development* seeds at fixed regularization. This suggests structural prediction improvements do not by themselves make the support differential identifiable. Do not tune regularizers to create an apparent win on these reused validation targets. No new scientific train/test split, approved Kd cohort, or independent family generalization exists.
- **Monitoring:** `scripts/monitor_benchmarks_v4.py` now also snapshots all seed variability and alternate support-K results in `research/monitoring/latest.json` and `LATEST.md`, with changed-file events; hourly monitoring notifications are enabled only for new meaningful changes or failures when the connected computer can be reached.

## Trained sequence + geometry engineering pilot — 5 November 2026

- **Current authoritative architecture is DynaCPI v4.0, NOT four independently learned experts.** Two complete affinity predictions share one PerceiverCPI-derived sequence foundation. Geometry is a trainable 3-layer GVP branch. Support-only centered calibration and bounded differential correction determine factorial A/B/C/D. Source-approved BindingDB science cohort is still BLOCKED, zero tasks approved.
- **Support adapter fully connected to chemistry:** `dynacpi/frozen_adapter.py` now combines exact pre-split support/query ID checks, stereochemistry-aware RDKit Morgan Tanimoto^4 similarity, immutable frozen sequence/geometry means and A/B/C/D outcomes. `tests_v4/frozen_adapter_real_chemistry_test.py` validated 12 real-smiles candidate tasks/120 queries with entirely SYNTHETIC affinity/prediction values, no measured BindingDB support used. `dynacpi/frozen_pair_predictor.py` runs a frozen eval-only foundation+GVP interface, enforces raw-pKd re-scaling from the saved standardization statistics and deterministic single-compound predictions.
- **Full real-3D inference engineering integration:** `tests_v4/full_inference_real_3d_synthetic_labels_test.py` passed with actual BRAF canonical 766-residue AlphaFold v6 graph (18,789 edges), real 650M cached ESM-2 tokens, eight actual staged compounds, five synthetic support labels and three synthetic query cases. Exact K=0/missing-geometry fallback, batch-vs-single consistency, factorial adaptation, permutation invariance passed. Untrained weights and synthetic labels: NO predictive accuracy claim. `research/full_inference_real_3d_synthetic_labels_test.json`.
- **Actual short supervised sequence pilot:** completed `scripts/train_engineering_pilot.py sequence` on legacy DAVIS 24 identifier-distinct training proteins / 1,296 pairs, eight validation proteins / 432 pairs, three epochs. Validation target-macro RMSE epochs 1/2/3 = 0.811185/0.813853/0.817272; best displayed epoch 1 but `research/pilot_sequence/checkpoint.pt` is the FINAL epoch-3 checkpoint. This was not a family-disjoint or source-approved scientific study. The trained checkpoint loaded strictly and raw-label denormalization was independently validated with zero discrepancy. `research/trained_sequence_checkpoint_inference_test.json`.
- **Pilot protein-to-UniProt mapping:** `scripts/survey_pilot_uniprot_mapping.py` found 30/32 unique pilot sequences exactly matching contiguous subsequences of human UniProt reference accessions (22/24 train; 8/8 validation); EPHA6 and FAK had no exact match in the strict search. ROCK2 pilot sequence is residues 686-1388 of reference O75116 and requires nonzero coordinate offset; canonical full-length numbering must NOT be silently reused as query-local coordinates. `data/engineering_pilot/uniprot_exact_subsequence_survey.json`. Exact reference subsequence mapping is NOT verification of physical experimental construct.
- **Pilot AlphaFold staging:** `scripts/stage_pilot_3d_graphs.py` downloaded and checked AlphaFold v6 PDB+PAE, mapped reference numbering into pilot-local residue indices, and stored graph/provenance hashes for 13 selected proteins (nine train/four validation) with 100% complete mapped backbone coverage. Including offset ROCK2. `data/engineering_pilot/limited_alpha_3d_graph_manifest.json`; `structure_graphs/<seqhash>/`.
- **Supervised geometry branch pilot:** `scripts/train_geometry_engineering_pilot.py` trained ONLY GVP for two epochs on 486 legacy DAVIS train pairs from nine geometry-mapped proteins, with the saved sequence model frozen, and 216 validation pairs from four mapped proteins. Target-macro sequence-only RMSE=0.925685; geometry=0.935921 after epoch 1 (best saved), 0.963520 after epoch 2. Geometry was WORSE on this small development subset; do not claim geometry improvement. Independent `tests_v4/trained_geometry_checkpoint_parity.py` matched saved history and found slight RMSE improvement for FRK(-0.002210)/MEK5(-0.002194), deterioration ADCK4(+0.044869)/OSR1(+0.000477). `research/pilot_geometry_9train_4val/`.
- **Real-label development adapter diagnostic, NOT final evaluation:** `scripts/evaluate_engineering_support_v4.py` used the four checkpoint-selected DAVIS development targets, 10 fixed query compounds, K=5, ten label-independent support draws per target, fixed unoptimized w0=.5, lambda_b=5, lambda_theta=.1. Target-macro mean episode RMSE A=0.86101741, B=0.86013856, C=0.86101741, D=0.86013856. **D-minus-B was ~+5.2e-11, effectively zero.** Mean Tanimoto^4 support weight sums ranged only 0.003-0.009 and geometry disagreement evidence V was tiny (<= ~5e-11). Thus in THIS development cohort, the differential correction was numerically inactive. This neither confirms nor refutes the planned BindingDB scientific hypothesis, where unapproved K5 candidate median W=0.215. Report: `research/pilot_geometry_9train_4val/support_adapter_development_diagnostics.json`.
- **Statistical evaluation correction:** `dynacpi/paired_evaluation.py` now aggregates query RMSE -> episode -> assay task -> **protein construct** -> independent protein component, not simply task -> component, preventing overweighting repeatedly assayed constructs. Synthetic evaluator test passed.
- **Validation checkpoint:** the expanded `scripts/run_v4_validation_suite.py` has now passed **15/15 checks**, including exact original checkpoint parity, the supervised geometry checkpoint reproduction, real-ESM/real-3D model-to-adapter wiring and construct-balanced evaluator controls. Zero failed checks. `data/audit/v4_validation_suite.json` and `research/validation_suite_logs/`.
- **Next hard scientific blocker:** zero source-verified BindingDB exact-Kd assay tasks approved; final independent protein-cluster roles unfrozen. The development curves reuse geometry checkpoint selection targets and must NOT be presented as independent test or paper gains. Do source-level Kd/construct review, freeze split after sufficient verified tasks, fit stage1/stage2 models on legitimate training roles, choose w0/lambdas on disjoint adaptation roles, and run paired D-vs-B test. Check geometry complementarity/variation and similarity-kernel evidence *on development data* before interpreting results. Preserve all failed or negative results.

## Source-evidence and statistical readiness checkpoint — 5 November 2026

- **UniProt metadata check:** `scripts/audit_uniprot_constructs.py` retrieved and cached all 30 official UniProtKB reference records for the label-blind assay-review queue; all 30 existing BindingDB candidate sequences were *exactly equal to their recorded UniProt reference sequences*. Report: `data/audit/bindingdb_assay_review_30_uniprot_construct_screen.json`. This does **not** establish the sequences of the actual assayed protein preparations; BindingDB itself warns that constructs can differ from reference sequences.
- **Publication metadata and potential construct/endpoint clues:** `scripts/audit_publication_sources.py` queried Europe PMC and resolved 20/20 distinct PubMed identifiers supplied by the 30 tasks; seven tasks had patents only. `scripts/screen_construct_clues.py` flagged 13/30 assay descriptions containing relevant domain/tag/complex/inhibition/competition wording. Reports: `data/audit/bindingdb_assay_review_30_publication_screen.json` and `bindingdb_assay_review_30_construct_clues.json`. Neither publication indexing nor regex detection independently verifies an experimental exact Kd.
- **Evidence dossier:** `scripts/build_source_evidence_dossier.py` created `research/BindingDB_30_Source_Evidence_Dossier_20261105.md` with all 30 task identities, accession links, article metadata, patents and review questions, excluding affinity values. Dossier and its review CSV are protected against overwriting. All 30 tasks remain UNREVIEWED; ZERO are experimentally approved.
- **Fail-closed scientific cohort gate:** `dynacpi/scientific_cohort_gate.py` requires separate source-approved cohort and frozen independent protein-group split artifacts, checks their identity hashes, source-review identifiers, disjoint components and nonempty roles, and rejects missing/unapproved manifests. Current real status: **BLOCKED_UNAPPROVED_NOT_TRAINING_READY**. `tests_v4/scientific_cohort_gate_test.py` verified rejection using synthetic data only; it does not certify any human review.
- **Paired statistical evaluation engine:** `dynacpi/paired_evaluation.py` calculates query RMSE per episode, averages across support draws/seeds, then equally across tasks within a protein component and across components, and computes deterministic paired 95% protein-cluster bootstrap CIs. Rejects duplicate rows, altered query labels, mismatched query identities/grids, nonfinite results and task-to-component mismatches. `tests_v4/paired_evaluation_test.py` passed with 3,600 synthetic query records only. No affinity model comparison has been run.
- **Consolidated suite:** `scripts/run_v4_validation_suite.py` now reports **11/11 engineering and statistical tests PASS**, zero failures, in `data/audit/v4_validation_suite.json`. These results establish software checks, not model quality or scientific novelty.
- **Next scientific blocker:** source/patent-level adjudication of experimental constructs and exact Kd semantics; release an independently verified, sufficiently diverse eligible cohort and frozen group assignments before any scientific model fitting. Do not treat UniProt equality, title/DOI matches or preliminary assay metadata as sufficient approval.

## Final verification checkpoint — 5 November 2026

- **Reproducible engineering suite passed 9/9 checks:** `scripts/run_v4_validation_suite.py` ran the 200-case adapter numerical comparisons, task-specific label-isolation tests, curation-review fail-closed tests, label-blind group split dry run, ESM sequence and real AlphaFold GVP engineering checks, and **full 1,148-row** archived DAVIS checkpoint prediction parity. Overall status `PASS_ENGINEERING_ONLY`, 0 failures. Structured audit: `data/audit/v4_validation_suite.json`; per-case logs: `research/validation_suite_logs/`.
- **No publishable DynaCPI benchmark yet:** The approval gate reports 30/30 manual review tasks still UNREVIEWED; no source-verified exact-Kd cohort, final protein-family split or matched DynaCPI affinity predictions are established. Preserve the quarantined data policy and execute scientific training only after approved data/splits are frozen.

## Continuation checkpoint — 5 November 2026, implementation validation

- **Reference checkpoint parity FULLY verified for preserved DAVIS fold 0:** `tests_v4/legacy_checkpoint_parity.py` loaded the unchanged original PerceiverCPI `FINAL_model0_best.pt` with strict state-dict equality and checked SIX unmodified source files by SHA-256. All **1,148** saved DAVIS test predictions were independently recomputed using the original batch size 50, original nonchiral Morgan fingerprints, and original tokenizer; max and mean absolute differences **0.0**. Evidence: `research/legacy_checkpoint_parity.json`. This verifies ONE LOCAL archived checkpoint, not original publication-wide model equality nor v4 improvement. Original legacy project was only read.
- **Cross-assay compound identity overlap:** `scripts/audit_cross_assay_episode_overlap.py` found 185 protein sequences with multiple K5 assay tasks, 114 with query/support compound identity overlap across assays, and 1,933 distinct compound IDs that act as query in one assay and support in another. This is a POTENTIAL contamination channel, not proof of duplicate biological labels. Report: `data/audit/bindingdb_full_kd_cross_assay_episode_overlap.json`.
- **Model-input privacy / label boundary:** added `dynacpi/episode_data.py` and `tests_v4/episode_data_real_manifest_test.py`. 4,961 episode checks over 1,146 candidate tasks passed; 377 support-K20 eligible; zero cross-assay support, zero within-episode query/support overlap, zero query-label exposure to predictor objects. Negative tests reject forged cross-assay support, reuse of query compounds, and query-label perturbations from affecting model inputs. Artifact: `data/audit/bindingdb_full_kd_episode_loader_validation.json`. These are PRE-SPLIT data interface controls.
- **Human Kd assay review gate:** `scripts/enrich_assay_review_queue.py` created `data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv` with source identifiers, accession and article/patent metadata, but no affinity values; generator refuses to overwrite the human review file. `scripts/validate_assay_reviews.py` requires explicit, documented construct/measurement/assay decisions; tampering and fabricated approvals were rejected in `tests_v4/assay_review_gate_test.py`. CURRENT STATUS: 30 UNREVIEWED, 0 APPROVED, scientifically NOT TRAINING-READY. `data/audit/bindingdb_assay_manual_review_status.json`.
- **Dry-run protein split algorithm:** `dynacpi/group_split.py` uses label-blind balancing with exact independent-group role quotas. On 168 preliminary K5 components, `tests_v4/group_split_dryrun_test.py` obtained 100/17/17/17/17 component roles and 735/110/101/100/100 task counts (base-train/base-val/adapter-train/adapter-val/test). All 1,146 tasks were allocated without accepting any affinity values. ONLY summary statistics and assignment SHA were saved in `data/audit/bindingdb_full_kd_group_split_dryrun_policy.json`; **NO FINAL ROLE ASSIGNMENT WAS FROZEN**.
- **Research protocol:** `research/Kd_Curation_and_Split_Protocol.md` documents source eligibility, exact-Kd verification, support/query separation, domain/construct alignment, independent group roles, primary controls and metrics.
- **Next scientific gate:** Have actual source/paper-level evidence reviewed for a sufficiently diverse Kd cohort; approve traceable construct/assay tasks; then freeze independent family splits prior to scientifically valid affinity training. Engineering tests and parity cannot substitute for biological evaluation.

## Live checkpoint — 5 November 2026, audited continuation

- **Project direction:** v4.0 (two complete predictors: sequence and geometry; centered support calibration and differential adaptation). Four learned experts and the v3 reliability controller remain deferred. Do not mistake those older discussions for the authorized primary implementation.
- **Connected machine and path:** built-diff12, `C:\\Users\\J0124\\dynacpi_v4`; preserve isolated legacy PerceiverCPI. Root was inspected; no AGENTS.md exists in the project or its checked parents.
- **ESM-2 extraction:** `data/esm2_t33_650M_layer33_w1000_o128_fp16/progress.json` reports `{"stage":"complete","files":462,"total":462,"peak_MiB":1431.77587890625,"seconds":120.3}`. Real cache files exist. This verifies extraction status only, not biological training or a cached-feature integrity audit.
- **Real-structure geometry smoke:** `research/geometry_smoke.json` reports 96 nodes, 2,094 directed edges on AlphaFold P15056, nonzero GVP/projection gradients, maximum rotation/translation difference 2.38e-7, and missing-node / no-geometry checks passing. Its sequence tokens and regression targets were synthetic. The full production geometry pipeline and affinity training remain pending.
- **New BindingDB subset audit:** `scripts/audit_bindingdb_article_kd.py` scanned versioned `BindingDB_BindingDB_Articles_202611_tsv.zip` (SHA-256 `423609987c3589ca204d59380da7087fe7c67fe0e73b411347e0698ed5fbbdc8`) plus the reactant-to-assay map. It found 93,712 article records, 2,650 populated Kd values, 2,490 positive numerically exact-looking Kd records, 2,430 assay-linked single-chain/sequence/SMILES candidates, 555 preliminary construct-assay groups, 29 groups with >=15 distinct raw SMILES, and 8 with >=30. Output: `data/audit/bindingdb_articles_kd_eligibility.json`.
- **Data limitation update:** The Articles subset alone is inadequate for the planned five-group higher-support protocol, but the full verified November 2026 BindingDB archive is now downloaded and independently screened. The full cohort is still provisional: exact-looking Kd, construct and assay metadata require scientific verification. No final scientific test split has been established.
- **Independent ESM cache integrity:** `scripts/audit_esm_cache.py` checked all 462 files, 346,865 amino-acid positions, 570 windows, with zero missing/bad files. Output: `data/audit/esm_cache_integrity.json`.
- **Actual cached ESM pair smoke:** `tests_v4/cached_sequence_pair_smoke.py` processed two real staged DAVIS compounds with actual frozen ESM residue embeddings (lengths 545, 588). Batch vs individual prediction difference 0; nonzero sequence-projection gradient. The head and loss labels were synthetic, so this remains engineering-only. Output: `research/cached_sequence_pair_smoke.json`.
- **Preliminary RDKit and protein clustering screen:** `scripts/screen_bindingdb_article_clusters.py` rejected 29 invalid candidate SMILES; preliminary 15-compound cohort has 28 distinct sequences in 24 connected components, whereas the 30-compound cohort has seven sequences in FOUR connected components. These counts exclude construct/assay QC and use a preliminary local-alignment similarity protocol. Output: `data/audit/bindingdb_articles_kd_cluster_screen.json`. The Articles subset cannot fill five disjoint protein roles for K<=20.
- **Full BindingDB retrieval completed and verified:** `scripts/resume_bindingdb_full.py` recovered the entire 594,284,548-byte November 2026 archive from the saved partial. ZIP CRC passed and SHA-256 is `470a3c2e4aebe3c493de3e9e46e739592eb8e29b3adafb5189c875442268a0b2`. Upstream `.md5` endpoint was HTTP 404; record is ZIP-CRC verified with local SHA-256, not publisher-MD5 verified. Archive includes 9,004,582,822-byte uncompressed TSV. Manifest: `data/bindingdb_202611/BindingDB_All_202611_tsv.json`.
- **Manuscript update:** Added a traceable Section 7.2 to `research/DynaCPI_Manuscript.md` covering these engineering and preliminary data results with explicit limitations.
- **Full Kd preliminary screen:** `scripts/audit_bindingdb_article_kd.py --full` scanned 3,243,660 source records. Of 131,531 nonempty Kd fields, 82,481 were positive numeric exact-looking values and 80,139 candidate rows linked to assay identifiers (4,157 unique candidate sequences). RDKit task-level screen (`scripts/screen_bindingdb_cohort.py --full`) retained 80,042 valid candidates, 1,146 tasks with >=15 canonical isomeric SMILES, 377 tasks with >=30, corresponding to 555 and 215 sequences. Results: `data/audit/bindingdb_full_kd_eligibility.json`, `bindingdb_full_kd_rdkit_screen.json`.
- **MMseqs2 all-candidate protein-group screen:** using WSL Ubuntu `mmseqs` 18-8cc5c+ds-1, self-search 4,157 unique sequences at sensitivity 7.5, >=30% identity and >=80% coverage of shorter sequence, followed by connected components: 2,225 components overall; K=5 candidate tasks occupy 168 components and K=20 candidate tasks occupy 81. The largest K5 group has 492 tasks; cluster imbalance is critical. Requiring >=80 aligned amino acids instead of no minimum yields 171 and 82 components respectively; do not silently select a threshold after seeing test scores. Reports: `bindingdb_full_kd_mmseqs_component_screen.json` and `bindingdb_full_kd_mmseqs_threshold_sensitivity.json`. These group counts are preliminary, not final independent biological families.
- **Experimental assay/provenance audit:** The `BindingDB_Assays_202611_tsv.zip` join provided assay names and descriptions for all 15,179 preliminary raw construct/assay groups. For 1,146 K=5 tasks, 1,082 had no pH, 1,107 had no temperature, 366 lacked article DOI; 95 had >0.1-pKd conflicting replicate groups. Two groups contain distinct target names under a shared sequence and assay, including FLT3 variant nomenclature. Ten original-source-vs-candidate protein sequence SHA differences were explicitly flagged. Source records are in `bindingdb_full_kd_source_provenance.jsonl`, reports in `bindingdb_full_kd_task_provenance_audit.json`, `bindingdb_full_kd_provenance_mismatches.json` and `bindingdb_full_kd_assay_context_audit.json`. Missing conditions do not automatically invalidate a task.
- **Split feasibility only, NOT final split:** 1,000 hypothetical group-allocation simulations with 60/10/10/10/10 proportions indicate highly unstable task counts due to giant connected components; median K5 test tasks 48, observed range 17–687, with 16–17 test components. No split was selected based on results or task labels. See `bindingdb_full_kd_split_feasibility_simulation.json`.
- **End-to-end ESM+GVP real-input smoke:** `tests_v4/cached_esm_gvp_pair_smoke.py` processed all 766 residues of canonical AF-P15056-F1 v6 BRAF with 18,789 directed edges, actual ESM-2 cache and real RDKit molecule. Zero-projection and missing-structure prediction fallbacks were exactly equal to sequence predictions, GVP gradient nonzero, projected geometry changed the untrained output. Random model parameters and synthetic loss labels: NOT a trained affinity benchmark. See `research/cached_esm_gvp_pair_smoke.json`.
- **PRE-SPLIT candidate episodes generated and verified:** `scripts/build_kd_candidate_episodes.py` screened 80,139 source-linked measurements and retained 80,034 technically valid candidate observations across 15,145 candidate tasks. Generated `data/audit/bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl`, `bindingdb_full_kd_candidate_episodes_PRE_SPLIT.jsonl` and `bindingdb_full_kd_compound_median_candidates.jsonl`. `scripts/verify_candidate_episode_manifest.py` independently confirmed 78,493 deduplicated compound observations, 1,146 K=5 candidate episodes, 377 K=20 candidate episodes, unique task IDs, disjoint support/query identities, nested support prefixes, and component membership; report `bindingdb_full_kd_candidate_episode_manifest_validation.json`.
- **Assay quality triage (NOT final approval):** `scripts/triage_bindingdb_candidate_tasks.py` reports 1,041/1,146 K5 tasks with no hard automated review flags spanning 158 preliminary protein components, 299/377 K20 tasks spanning 72 components. Only 19 K5 tasks have no automated flag whatsoever; missing pH and temperature dominate the rest. The 1,041 tasks are *not scientifically verified compatible assays*. Report `data/audit/bindingdb_full_kd_candidate_review_triage.json`.
- **Label-blind assay-review queue:** `scripts/prepare_assay_review_queue.py` created `data/audit/bindingdb_assay_manual_review_queue_30.csv` with 30 metadata-only tasks representing 30 distinct preliminary protein components, SHA-selected without affinity observations. Each row includes assay name/description, source flags and blank human verification fields. Source: `bindingdb_assay_manual_review_queue_30_manifest.json`. This is an actionable review sample, NOT a vetted or approved cohort.
- **Next unblocked scientific milestone:** source-verify exact Kd semantics, construct identity and assay compatibility beginning with the 30-component review queue; establish a documented inclusion policy and family/domain homology checks. Freeze protein-group splits only after those checks, without consulting held-out query labels. Independently complete original-backbone parity and training. No DynaCPI benchmark exists yet.
- **No claim:** No matched held-out DynaCPI v4 performance improvement, publication novelty, full checkpoint parity, complete residue-aligned 3D training, or final scientific benchmark is established by these checks.

**5 November 2026. Execution authorized and resumed.**

The legacy inventory and fingerprint staging completed on built-diff12. Across nine fold-zero files, 126,741 records yielded 2,642 compounds and 462 unique sequences. DAVIS and KIBA have 409 and 127 shared train–validation pair keys respectively. Construct, assay and endpoint provenance still require resolution. The v4 support adapter passed 200 constrained optimizer comparisons on the PC. The pinned CAB passed an isolated synthetic GPU consistency test. Manuscript source is in research/DynaCPI_Manuscript.md. The next scientific gate is an eligible construct–assay cohort and verified group splits. Full checkpoint parity, ESM extraction, geometry integration and predictive evaluation remain pending.

## Resume checkpoint — DAVIS provenance, 5 November 2026

Use this file as the model handoff. Execution remains authorized. The current manuscript is [DynaCPI_Manuscript.docx](https://chatgpt.com/library?file_id=libfile_b606baf89af4819197d6a5b0d3d63721&account_id=personal). Its source and the recovered ten-page supplement are in research/ on the PC. The manuscript contains 49 main-paper and 10 supplementary claim entries.

The audited legacy DAVIS preparation script uses the first fifth of sorted compounds/sequences for test, then a seeded row shuffle for validation. The input CSV has 30,056 rows, 379 sequences and 25,772 unique SMILES–sequence keys. Of 1,224 repeated keys, 546 have distinct labels. All 379 sequences match the DeepDTA reference dictionary, where 442 named targets collapse to 379 sequences. Eighteen sequence keys map to multiple identifiers, including BRAF and BRAF(V600E). A sequence-only join cannot recover the per-row construct identity. Keep legacy reproduction separate. The script, reference dictionary, hashes and detailed observations are in scripts/audit_provenance.py and data/audit/davis_provenance.json.

The archived PerceiverCPI DAVIS fold-zero run yielded MSE 0.700482 and 115 of 1,148 predictions below five. This was a local baseline result. Main DynaCPI comparison remains pending.

The isolated project .venv now imports PyTorch 2.5.1+cu124 and RDKit 2025.09.2 through an explicitly shared read-only legacy site-packages path. fair-esm 2.0.0 was installed in the extension environment. This environment remains dependent on the legacy environment's location. The corrected MPN port and sequence predictor exist in dynacpi/mpn.py and dynacpi/sequence_model.py. tests_v4/sequence_smoke.py exercised real molecules and synthetic residue features on the RTX 4060: 1,649,153 parameters, batch prediction difference 3.73e-9, compound-feature difference 7.45e-9, finite gradients through the sequence projection. This is an implementation check, not an affinity result. An official ESM download/extraction job was started; check data/esm2_t33_650M_layer33_w1000_o128_fp16/progress.json, research/esm_fetch.log and the actual cache before assuming completion. Its model files use validated HTTP ranges and the resulting checkpoint hash is recorded.

The goal is to improve compound–protein affinity prediction over PerceiverCPI while retaining protein 3D. A matched empirical comparison is required. Do not say the extension has already improved accuracy, is guaranteed to work, is novel, or is publication-ready. A successful software smoke test is not biological validation.

**Confidence statement:** The project has a concrete implementation path, a testable scientific question, explicit controls and defined fallbacks. It is credible to build and evaluate. Outperforming PerceiverCPI and producing a publishable contribution are hypotheses, not established outcomes. The planned controls are designed to distinguish a useful contribution from extra capacity, calibration or data leakage.

## Historical inventory before execution resumed

| Item | Verified status |
|---|---|
| User computer | built-diff12, Windows; Desktop Commander device 6792ae32-bdec-4aed-9b55-4c63a638d68c |
| GPU | NVIDIA RTX 4060 Laptop GPU, 8188 MiB reported memory |
| Legacy project | C:\\Users\\J0124\\OneDrive\\Desktop\\perceiver_cpi; preserve all code, environment, datasets and checkpoints |
| New isolated clone | C:\\Users\\J0124\\dynacpi_v4 |
| New branch | dynacpi-v4, created at upstream commit 755f076ef51099a90332fd1ec8437f94e3750c0b |
| Upstream repository | https://github.com/dmis-lab/PerceiverCPI |
| Existing legacy environment import check | PyTorch 2.5.1+cu124; CUDA available; RDKit 2025.09.2 |
| New extension environment | Not created or dependency-validated |
| Architecture specification | Completed; complete contract reproduced below |
| Neural implementation | Not established. Interrupted adapter-write attempt did not create the intended dynacpi package directory at the read-only check. Inspect any partial files before resuming; do not assume they are valid. |
| Datasets for v4 | No v4 dataset acquisition, provenance audit, split construction or preprocessing completed |
| ESM/structure cache | Not built |
| Training | No DynaCPI training launched in this session |
| Numerical checks | 200 small NumPy/SciPy constrained-optimizer comparisons and adapter edge cases passed; no neural benchmark or full-graph test |
| A100 | User previously said one can be obtained; no connection or actual configuration verified |

The earlier clone/setup and environment-import process finished. This handoff does not assert the state of unrelated user processes or legacy training. Do not stop unrelated processes. A VS Code open command was issued, but successful window display was not verified.

## Execution after go: dataset preparation first

1. Read any AGENTS.md instructions in the new project and its applicable parents. Inspect git status and partial files. Record upstream and extension revisions. Keep user modifications and legacy assets intact.
2. Inventory the legacy reproduction directory for actual DAVIS/METZ/KIBA files, split manifests, configs and checkpoints. Validate provenance before reuse. Existence alone does not establish correctness. Never reuse a checkpoint trained on final-test targets in a held-out-target claim.
3. Resolve the official PerceiverCPI README's dataset acquisition links. Download DAVIS first if a verified copy is unavailable. Record source URL, access date, license, size and SHA-256. Check expected columns, count unique chemicals/constructs, identify missing values and duplicates, and reconcile original targets and units with the source.
4. Keep an immutable raw copy. Make a separate legacy-reproduction dataset and a v4 exact-Kd candidate dataset. Audit DAVIS boundary values from provenance. Do not silently treat all floor values as either exact observations or censoring.
5. Acquire a versioned BindingDB export with exact Kd, construct sequence and sufficient assay/provenance fields. Stream/chunk processing rather than loading the entire export blindly. Preserve original measurement rows and source identifiers. Produce an exclusion ledger for unresolved units, qualifiers, constructs, chemical identities and assay compatibility.
6. Canonicalize chemicals using a pinned RDKit policy retaining stereochemistry and original salt/parent information. Derive radius-2, 2048-bit chirality-aware Morgan fingerprints and graph features. Preserve atom-feature ordering from the pinned upstream code. Do not merge distinct stereoisomers merely because they share an achiral fingerprint.
7. Convert valid exact Kd to pKd; group and aggregate exact replicates only within the same chemical/construct/assay context. Keep count/spread and constituent record IDs. Do not merge Ki, IC50 and KIBA scores into the Kd endpoint.
8. Create protein-similarity connected components, then the five role partitions specified below. Audit component size and domain/family overlap. Save immutable manifests before model fitting. Chemical standardization and deduplication precede splits; learned scalers use base training only.
9. Count eligible tasks: minimum 15 distinct compounds for K=5 plus ten fixed queries; minimum 30 for the full K<=20 curve. Report tasks, constructs, protein clusters, structures, exclusions and counts per partition. If the main dataset cannot support the split roles, stop the scientific training campaign and report the viability problem; keep engineering checks separate.
10. Resolve one pinned, sequence-matched AlphaFold structure per eligible construct under the specification's policy. Verify mappings and completeness before graph features. Retain sequence-only cases. Download only the needed records, with resumable manifests and caching; avoid bulk structure downloads without an eligible sequence list.
11. Build gap-aware GVP graph features on CPU and cache frozen ESM embeddings on GPU. Cache keys include every model, sequence, structure and preprocessing version. Run mapping, graph, masking and numerical checks before processing the full cohort.
12. Prepare METZ and KIBA as separate later tasks after DAVIS integration and data viability. Verify their endpoint definitions. Do not delay the first working DAVIS path for all secondary datasets.

Deliver preprocessing artifacts in the project: observations.parquet, tasks.parquet, compounds.parquet, constructs.parquet, structures.parquet, split_manifest.json, episode_manifest.json, feature_manifest.json and data_audit.md. Paths are proposed outputs, not files claimed to exist. Every derived row must trace to raw records; every exclusion has a reason.

## RTX 4060 plan and parallel work

Use the GPU actively for ESM inference and neural computation, and CPU workers for RDKit, parsing and graph assembly. High useful throughput is the objective; 100% utilization is not a correctness criterion.

- Create an isolated extension environment outside the legacy .venv. Record a tested dependency lock. Do not upgrade the existing working environment in place.
- First profile a small real batch with the exact 650M ESM checkpoint. Start with one sequence/window per inference batch, inference mode, and supported reduced precision. Record peak allocated/reserved memory and elapsed time. No claim is made that every 1000-residue batch fits 8 GB.
- Keep the specified 1000-residue/128-overlap extraction policy consistent. On memory pressure, lower the batch size and use verified memory-efficient attention/offloading if supported. Any change to window length or checkpoint requires a new extraction version and consistent regeneration; never silently swap a smaller ESM model for benchmark results.
- Complete frozen ESM extraction as a separate GPU stage, then release its model memory before predictor training. Do not co-reside the 650M encoder and trainable pair model unnecessarily.
- Use small pair microbatches, initially 1–4, with gradient accumulation to effective batch size 64. Bucket by sequence length and graph size. Begin GVP geometry in FP32; enable mixed precision only after finite-gradient and invariance checks.
- Start with two CPU preprocessing workers and a bounded queue; measure RAM and throughput before increasing workers. Reuse protein features across compounds. Avoid repeated graph/sequence encoding when weights and inputs are unchanged.
- Overlap independent CPU parsing/chemistry with the GPU extraction queue once manifests are stable. Use one GPU-heavy process at a time initially; competing training and ESM jobs can reduce throughput or cause out-of-memory failures.
- Keep downloads bounded and resumable. Write checkpoints atomically; keep optimizer, scheduler, RNG, sampler and preprocessing state needed to resume exactly. Log each completed stage and actual GPU use.
- Do not disable thermal protections, overclock, alter power policies or promise a training duration. Profile real batches first. The laptop can establish a vertical implementation; the full multi-seed/baseline campaign may be better run on the A100.

The user's request to multitask means independent preparation work may run concurrently after go. It does not change the scientific task into mixed-endpoint multi-task learning. Respect dependencies: splits before learned preprocessing, sequence training before geometry training, frozen predictors before adaptation. Execution has since been authorized.

## Evidence needed before calling this an improvement

- Original PerceiverCPI reproduction/parity is reported separately from the new common-split comparison.
- Train original PerceiverCPI and the new sequence/geometry models using matched endpoint, partitions and evaluation cases. Report additional pretrained features, parameter counts, tuning budgets and support access.
- Report zero-shot performance with K=0. A few-shot method beating a support-free baseline alone does not establish architectural superiority; include support-enabled baseline adaptation.
- Compare full adaptation D against identical calibration-only B, plus residual methods, sequence-capacity controls and verified few-shot methods.
- Use independent protein-group test data, paired support draws, three training seeds and cluster-aware uncertainty. Report absolute pKd effects, the provisional 2% relative RMSE rule and per-target degradation.
- A paper-worthy claim requires a justified contribution beyond known local weighting/calibration, convincing comparisons, reproducible code/data policy and an honest limitations analysis. Publication cannot be guaranteed in advance. If the geometry/adaptation claim fails, report that result and reassess on development data without repeatedly testing against the same held-out set.

## Immediate resumption checklist

When the user says go: inspect partial state → create isolated environment → verify DAVIS data and upstream adapter → implement and test corrected sequence path → audit BindingDB eligibility in parallel with independent feature preparation → train geometry → cache predictions → compare adaptation variants. Update this file with actual evidence after each milestone. A phase is complete only when its outputs and checks exist.

The full architecture and experiment contract follows. Its implementation settings are starting commitments, not completed experiments. The dated status above supersedes the historical inventory.

---

# DynaCPI v4.0 — implementation architecture and experiment contract

**Decision date:** 4 November 2026, Asia/Kolkata.  
**Status:** Architecture selected for implementation. This is a research specification, not a trained or benchmark-validated model.  
**Authority:** This document supersedes the v3 architecture for the next implementation. The earlier v3 files remain historical references and a later comparator. The 3 November redesign report supplies the broader literature assessment.

## 1. The decision

Implement **two complete affinity predictions from one shared, frozen sequence foundation**, with a trainable protein-geometry branch and a small, explicit support adapter. First train the sequence foundation; freeze it; then train the geometry branch; freeze everything before support adaptation. Experimental support labels enter only the adapter during the main mechanism comparison.

The research question is: **Does a small measured support panel identify useful changes in the contribution of protein geometry for new compounds on held-out targets, beyond ordinary affinity calibration?**

This fixes the implementation direction. It does not establish novelty, superiority or publication readiness. Local regression, shrinkage, expert selection and support conditioning already have substantial prior art. A contribution would require reproducible evidence about this geometry-specific question, strong controls, and useful performance under the declared shifts.

### Fixed primary scope

- Continuous affinity regression; verified exact Kd converted to pKd for the scientific study.
- Compound graph plus Morgan fingerprint; frozen ESM-2 sequence features; trainable GVP geometry.
- One sequence predictor and one structure-enhanced predictor, sharing immutable foundation weights.
- Compound-conditioned attention over actual residue tokens before the pooled PerceiverCPI interaction block.
- Regularized support calibration plus centered disagreement adaptation; bounded output corrections.
- Protein-group holdout, fixed support/query episodes and explicit missing-input behavior.
- DAVIS for integration and legacy reproduction; a curated BindingDB Kd cohort is the candidate main study.

Four experts, a learned reliability controller, a support transformer, docking, multiple conformers, uncertainty heads and censored-label optimization are outside v4.0. They are possible later studies only if a concrete result justifies them. Protein 3D remains part of v4.0.

## 2. What the source audit resolved

The official PerceiverCPI source was inspected at commit **755f076ef51099a90332fd1ec8437f94e3750c0b**. GVP-PyTorch was inspected at **316eea7cd6446dd928e5975634152ec3b28e65f5**. These commits are the reproduction references; no upstream code was modified.

| Audited behavior | Implementation consequence |
|---|---|
| `InteractionModel.forward()` compresses the protein to one vector before `CAB`; `AttentionBlock` forms an outer-product attention matrix over feature channels. It is not residue-token attention. | Add an explicit residue readout in the extension. Describe this as a change to the protein interface, not unchanged checkpoint behavior. |
| `InteractionModel.featurize()` uses the molecular encoder and FFN but omits protein/CAB computation. | Expose the real pair feature immediately before `self.ffn(output)` in `forward()`. Do not use the generic method as a compound–protein representation. |
| `MPNEncoder` averages a padded incoming-message array; padding width is the maximum degree in the batch. | In the extension, sum only valid incoming messages and divide by the atom's actual degree, clamped to at least one. Require batch-composition invariance. Preserve the original calculation in the legacy reproduction track. |
| The model uses hard-coded CUDA allocations and min–max normalization without a zero-range guard. | Port device placement without altering legacy arithmetic; use guarded normalization in the extension. Record the differences. |
| The training code applies label-range-dependent weights; the default sequence length is 500. | Reproduce these only in the legacy track. The new study uses declared target-balanced MSE and complete sequence handling. |

The padded-mean issue is an inference directly supported by the two source operations. An isolated numerical check gives means 3 and 1.5 for messages `[2,4]` padded to widths 2 and 4; valid-degree averaging gives 3 in both cases. A full trained-model batch-invariance test is still required.

The pinned implementation uses summed incoming messages inside bond updates and a padded mean at the atom readout. Follow code when reproducing; do not silently replace all aggregation with the paper's prose description.

**Two tracks:** `legacy_perceiver` preserves upstream behavior for parity/reproduction; `dynacpi_v4` contains explicit engineering changes and is trained anew. A parity pass is not a reproduction of published metrics. The corrected sequence-only extension is the matched baseline for geometry and support claims.

## 3. Exact forward architecture

Let batch size be B, compound hidden width H=256, and protein length L. Each batch carries explicit masks and construct identifiers. All affinity values below are in pKd after inverse-transforming any training standardization.

| Module | Fixed initial configuration | Output |
|---|---|---|
| Compound D-MPNN | Upstream atom/bond features; directed bonds; depth 3; hidden 256; mean molecular readout; corrected valid-degree atom aggregation | `[B,256]` |
| Morgan branch | Radius 2, 2048 binary bits; chirality enabled for the extension; Linear(2048,256), ReLU, dropout 0.1 | `[B,256]` |
| Compound fusion | Upstream CAB's fingerprint-to-graph attention followed by its self-attention, preserving tied attention weights | `c: [B,256]` |
| ESM-2 | `esm2_t33_650M_UR50D`, layer 33, frozen/eval; 1280 features per residue | `[B,L,1280]` |
| Sequence projection | Linear(1280,256), LayerNorm(256) | `S: [B,L,256]` |
| Residue readout | Four attention heads, Q from compound `c`, K/V from residue tokens; masked mean residual | `p: [B,256]` |
| Pair interaction | Upstream CAB's final channel-attention operation with `c` as query and `p` as key/value | `z: [B,256]` |
| Affinity head | Linear(256,256), ReLU, dropout 0.1, Linear(256,1) | one scalar per pair |
| Geometry encoder | Three GVP convolution layers; node hidden `(128 scalars,16 vectors)`, edge hidden `(32 scalars,1 vector)`; dropout 0.1 | invariant scalars `[N_valid,128]` |
| Geometry projection | LayerNorm(128), Linear(128,256); final linear weight and bias initialized to zero | scattered residue residual `[B,L,256]` |

The 650M sequence checkpoint replaces the earlier budget-oriented 35M starting proposal because extraction is frozen and an A100 is available. This is a capacity choice, not evidence of an accuracy gain. A small checkpoint can be used for software development but its results must not be combined with the primary model's results.

### Sequence prediction

Use one instance of the original feature-channel attention operator, called A, for the three CAB operations. Its hidden width is 256 and its number of heads is one. Add no extra self-attention stack.

```text
graph = safe_minmax(DMPNN(compound))
fp = dropout(ReLU(Wfp * Morgan(compound)))
c1 = graph + A(fp, graph, graph)
c  = A(c1, c1, c1)

S = LayerNorm(Wseq * ESM(sequence))
r(S,c) = MHA(query=c[:,None,:], key=S, value=S, key_padding_mask)
p(S,c) = LayerNorm(masked_mean(S) + r(S,c).squeeze(query_axis))
zS = A(c, p(S,c), p(S,c))
s = inverse_training_label_scale(head(zS))
```

`safe_minmax` operates per molecular vector with denominator `max(max-min,1e-6)`. Empty/invalid compounds are rejected before this stage. The extension MPN has bias-free message projections; all padding messages must stay zero and be masked in reductions. There are no BatchNorm layers in new modules. The original port remains a separate path.

The residue MHA is a new extension, used identically in both predictions. It has one compound query and L residue keys, so its score tensor is `[B,4,1,L]`. The original CAB attention remains channel attention. Neither attention map is evidence of a physical binding site.

### Structure-enhanced prediction

```text
G = invariant_scalar_readout(GVP(valid_backbone_graph))
deltaH = scatter_to_construct(Wgeom * LayerNorm(G))
HG = S + valid_geometry_mask * deltaH
pG = p(HG,c)
zG = A(c, pG, pG)
g = inverse_training_label_scale(head(zG))
```

Only GVP and its geometry projection are trainable in this stage. `c`, S, the readout, A and the head retain the exact frozen sequence checkpoint. The pair operations remain differentiable with respect to HG: **do not wrap the geometry-dependent readout/head in `torch.no_grad()` merely because their parameters are frozen.** Detach/cache the sequence-only inputs, then let gradients flow through frozen operators to the geometry branch.

Zero-initializing Wgeom starts g=s. On the first update gradients reach the projection; gradients reach preceding geometry layers once that projection becomes nonzero. Verify this rather than declaring a failed GVP on the first step.

There are two complete predictors but one shared foundation. The difference `g-s` is induced by the added geometry pathway. It can still reflect extra capacity, structural metadata or optimization effects, so a sequence-capacity control remains necessary.

No eligible geometry: skip GVP and set g=s explicitly. For partial eligible structures, missing residue residuals are zero and all sequence residues remain in the readout. Freeze the geometry model after training; run both paths in eval mode during adaptation.

## 4. Feature and structure contracts

### Sequence extraction

The assay construct sequence is authoritative. Use the named ESM checkpoint and record its weight checksum, software revision, tokenizer, representation layer and sequence hash. Remove BOS/EOS/padding tokens and assert one output row per construct residue.

Extract in eval/inference mode and cache FP16 or BF16 embeddings with the stored dtype. For long sequences, use windows of 1000 residues with 128-residue overlap, append a final window covering the terminus, and average embeddings at overlapping positions. Record all offsets. This is an explicit approximation to full-sequence context, not a claim that the model has a hard 1000-residue architectural limit. No silent truncation. Downstream readout uses all reconstructed residue rows; unusually long inputs use smaller batches or exact chunked attention.

### Source and mapping

The primary geometry study uses **one sequence-matched AlphaFold-model source policy** to reduce source and co-crystal-selection confounding. Pin the downloaded snapshot/model versions. Experimental PDB structures are a separate sensitivity analysis, not an undeclared replacement source. Do not fold proteins or select structures using query affinities.

Map the structure polymer sequence to the actual construct. Require unique chain/mapping, exact agreement at mapped standard residues and complete finite N/CA/C/O coordinates for at least 80% of construct positions. This is a project QC threshold, not a biological guarantee. Reject ambiguous or mismatched geometry while retaining the record for sequence prediction. Canonical structures do not automatically represent mutants or alternative constructs.

Store construct index, polymer index, chain, author residue number, insertion code, residue identity, altloc selection and coordinate-validity flags. Select altlocs by highest occupancy with a deterministic identifier tie-break. Known modified residues require an explicit parent mapping. Experimental B-factors must never be interpreted as pLDDT.

### Graph

Build nodes only for complete finite backbone coordinates. For each node, use up to 30 nearest valid CA neighbors within 20 Å, plus continuous peptide neighbors. For n nodes set k<=n-1; handle n<=1 explicitly. Remove self and duplicate edges. A peptide neighbor requires adjacent construct and polymer positions, same chain, and a C–N distance between 1.0 and 2.0 Å. These graph choices are fixed engineering defaults.

Compute angle/orientation features before compacting the graph, or preserve original sequence indices throughout. Never connect a dihedral or orientation across a sequence gap merely because its retained nodes are adjacent in memory. Positional edge encoding uses original construct-index differences.

| Geometry tensor | Dimensions and content |
|---|---|
| Node scalars | 14: six sin/cos dihedral components, three dihedral-valid flags, three vector-valid flags, normalized pLDDT and its availability flag |
| Node vectors | `[N,3,3]`: forward/backward CA orientations and virtual side-chain direction |
| Edge scalars | 37: 16 distance RBFs, 16 sequence-offset features, two directed PAE values plus two availability flags, one peptide-edge flag |
| Edge vectors | `[E,1,3]`: normalized inter-CA displacement, consistent with edge direction |
| Edge index | `[2,E]`, confined to the relevant protein graph |

Use the reference RBF convention (16 centers over 0–20 Å). Normalize pLDDT by 100; clip PAE/30 to [0,1] while retaining raw metadata. Unknown values use zero placeholders and false availability flags. Invalid angles/vectors are zeroed before message passing and carry false validity flags. Clean invalid coordinates before arithmetic; `0 * NaN` is not masking.

Project node tuples `(14,3)->(128,16)` and edge tuples `(37,1)->(32,1)` through GVP input layers with tuple LayerNorm. Apply three GVPConvLayer blocks, then GVP `(128,16)->(128,0)` for invariant output. Scatter only valid scalar outputs. Never flatten directional vector components into an ordinary scalar MLP. Training gradients and inference invariance under proper rotations/translations are mandatory acceptance checks.

## 5. Support adapter: fixed mathematical contract

Use compatible, deduplicated, exact-pKd support records for the same construct and verified assay context as the task. Each support compound is evaluated by both frozen predictors. Neither path is trained on test support labels in this primary method.

Let s_i and g_i be support predictions, s_q and g_q query predictions, and y_i measured support pKd. Define `d_i=g_i-s_i`, `d_q=g_q-s_q` and the support-free prior `m0(q)=s_q+w0*d_q`, with `0<=w0<=1`.

Use the same chirality-aware binary Morgan fingerprint to calculate Tanimoto similarity T_iq. Fix `k_i(q)=T_iq^4`; do not normalize k to sum to one. Absolute weight retains the distinction between several near neighbors and uniformly irrelevant supports. Identical compounds are resolved through record grouping; empty fingerprints/invalid chemistry are rejected. No learned similarity metric in v4.0.

For W=sum(k)>0:

```text
dbar = sum(k*d)/W
x = d-dbar
u = y-s-w0*d
V = sum(k*x*x)
b = clip(sum(k*u)/(W+lambda_b), -2, 2)
theta = clip(sum(k*x*u)/(V+lambda_theta), -w0, 1-w0)
delta = clip(theta*(d_q-dbar), -1, 1)
prediction = m0(q) + b + delta
```

For W=0, return m0 directly with b=theta=delta=0. Compute these reductions in float64 on cached predictions. Reject nonfinite valid inputs; do not turn model failures into apparently valid zero-support episodes.

Before clipping the query differential, b and theta exactly minimize the centered, box-constrained ridge objective:

\[
\sum_i k_i(u_i-b-\theta x_i)^2+\lambda_b b^2+\lambda_\theta\theta^2,
\quad |b|\le2,\quad -w_0\le\theta\le1-w_0.
\]

Both regularizers must be positive. The additional **query differential cap of 1 pKd** is a deliberate refinement of the 3 November design, which left that term unbounded. The offset cap is 2 pKd, so the total correction relative to m0 is at most 3 pKd. These are provisional engineering limits, not confidence intervals or guarantees of lower error. Report how often each cap activates. With clipping active, call the result a bounded residual adapter, not an exact unconstrained affine prediction or a pure convex mixture.

Centering makes b identical in calibration-only and full variants when their settings are matched. A common additive shift of every support label cannot change theta or delta. With one support or constant support disagreements, theta is zero: such evidence can calibrate the baseline but cannot distinguish an offset from a different geometry preference. If all k tend to zero, both corrections vanish for bounded inputs. V is evidence variation, not a confidence probability.

### Fixed fitting procedure

1. Fit w0 on the adaptation-training cohort by minimizing the group-weighted zero-shot squared error. It is a single scalar, not a target lookup. For residual e=s-y, the solution is `clip(-sum(a*d*e)/sum(a*d*d),0,1)` with equal cluster/construct/task weighting a; use zero if the denominator vanishes.
2. On adaptation validation, choose lambda_b from `{1,5,20}` using the calibration-only primary metric; freeze it.
3. With the identical prior, kernel and offset, choose lambda_theta from `{0.01,0.1,1,10}` using the full model's primary metric. These numbers operate in raw pKd coordinates, never silently standardized labels.
4. Fix the caps at 2 and 1 and the similarity exponent at 4 for this version. Do not tune them on test results. An unclipped differential is a declared sensitivity comparison only.

Ties within numerical tolerance select the larger regularizer. This is at most seven single-hyperparameter settings after fitting the prior; the main study does not train a routing network. Refit nothing after opening test results.

## 6. Data and evaluation units

Convert positive Kd in molar units with `pKd=-log10(Kd_M)`; for nM, use `9-log10(Kd_nM)`. Keep the original value, units, relation and provenance. Inequalities and intervals are excluded from the exact-label mechanism study and separately counted; they are not converted into exact labels.

Group observations by canonical stereochemistry-preserving compound identity, actual construct, endpoint and verified assay context. Aggregate exact replicates within the same context using median pKd and retain count/spread. Conflicting or unresolved measurement types require QC, not averaging across assays. A shared publication or protein name alone does not establish assay compatibility.

The primary adaptation task is a construct–assay-context combination. If one construct has several tasks, average task metrics within that construct before protein-cluster aggregation. Do not count its assays as independent proteins. This resolves an ambiguity in earlier target-level wording.

Initially require at least 15 unique exact compounds per task: reserve 10 query compounds by a deterministic identity-based random selection and at least 5 disjoint support candidates. Query choice does not inspect labels. For K=5, use ten predetermined support draws from the remaining pool; tasks with exactly five candidates have no support-draw diversity and must be reported. For the full K={0,1,2,5,10,20} curve, use a separate fixed cohort with at least 30 compounds, the same ten queries and nested support prefixes. Report the loss of eligible tasks.

| Dataset | Role and restriction |
|---|---|
| DAVIS | Engineering integration and original-protocol reproduction. Audit boundary-value provenance. Strict protein clustering may leave insufficient independent groups. |
| BindingDB exact Kd | Candidate primary study after construct, assay, duplicate and eligibility auditing. Counts have not yet been established. |
| METZ | Secondary replication only after confirming its specific endpoint and qualifier semantics. |
| KIBA | Separate score prediction task with separately trained models/configuration; never pooled with pKd. |

Keep all qualified affinity records for sequence evaluation regardless of structure availability. Report the main geometry-mechanism endpoint on the predeclared geometry-eligible test subset, plus overall availability-aware performance on all eligible tasks. Eligibility uses input provenance/QC only. Always show both counts to expose structural-selection bias.

## 7. Splits and training stages

Build connected components of protein-similarity edges at >=30% identity over >=80% of the shorter construct. Pin aligner, identity definition, coverage mode and all settings. Audit local domain/family homology separately. This operational threshold does not establish unrelated biology. Check the largest component and family composition before assigning partitions.

Use **five distinct protein-group roles**, targeting percentages by independent group counts: base train 60%, base validation 10%, adaptation train 10%, adaptation validation 10%, final test 10%. Exact percentages can deviate to keep components intact; report achieved counts. This explicitly separates base checkpoint selection from adaptation fitting and validation. Do not move a difficult group after inspecting its errors.

If the audited corpus cannot support these roles, the main experiment is not viable under this protocol. DAVIS can still verify engineering. A smaller-cohort cross-fitting protocol would require its own preregistered design before testing; do not silently substitute random pair splits. Use development data to assess expected interval precision and power; no fixed number of groups alone guarantees adequacy.

| Stage | Trainable parameters | Data and stop rule |
|---|---|---|
| 0. Legacy audit | Original model as needed | Original protocol; separately verify adapter parity at FP32 rtol=1e-5, atol=1e-6. |
| 1. Sequence foundation | D-MPNN, FP projection, sequence projection, residue readout, CAB, scalar head | Base train; checkpoint selected on base validation. ESM always frozen. |
| 2. Geometry branch | GVP plus invariant output/projection only | Geometry-eligible base train; checkpoint selected on geometry-eligible base validation. All foundation parameters and dropout states frozen/eval. |
| 3. Fixed predictions | Nothing | Cache s,g and fixed fingerprints for adaptation/test tasks with checkpoint and feature hashes. |
| 4. Adaptation | w0 scalar on adaptation train; regularizers selected on adaptation validation | The fitting procedure in Section 5. No expert parameter changes. |
| 5. Final evaluation | Nothing | One locked experiment on final-test groups, three independent trained seed pairs, ten paired support draws. |

Initial optimization settings for stages 1–2: AdamW, learning rate 3e-4, weight decay 1e-4, gradient norm clip 1, 100-epoch ceiling, early stopping after 15 validation checks without an absolute RMSE improvement of 1e-4. Use target/task-balanced sampling and MSE on labels standardized using base-training statistics only; invert before caching predictions. Validation uses raw pKd. Geometry training retains those same normalization statistics.

Start with effective batch size 64 pairs, obtained by gradient accumulation if necessary. A training epoch is `ceil(number_of_base_train_pairs/64)` effective updates using the balanced sampler; record repeated sampling. Bucket by protein length and graph nodes without silently omitting oversized examples. The frozen backbone's dropout stays off in stage 2; geometry dropout is on during training. The final evaluation has all dropout off.

These settings are implementation defaults, not tuned claims. One declared development-only learning-rate check `{1e-4,3e-4}` may be used equally for matched models. Do not conduct an open-ended architecture search after inspecting test results.

Cache ESM features, raw graph features and frozen molecular/sequence features where valid. Do not cache trainable GVP outputs across optimizer updates. Use FP32 GVP geometry initially; enable mixed precision only after finite-gradient and invariance checks. No A100 runtime or memory fit has been measured. Profile a real maximum-size batch before scheduling the campaign.

## 8. Comparisons that decide the claim

Run the following on exactly the same predictions, episodes and offset settings:

| Variant | Prediction | Question |
|---|---|---|
| A | m0 | Zero-shot default |
| B | m0+b | Calibration alone |
| C | m0+delta | Differential geometry adaptation alone |
| D | m0+b+delta | Full method |

**D versus B is primary.** D versus A alone cannot establish geometry adaptation. Also compare s, g, uniform averaging, fitted global mixing, support-global (k=1) adaptation, nearest-neighbor and kernel residual correction, and a query-only router with equivalent support calibration. The latter is an explicit competitor, not another module in D.

Add simple five-step support fine-tuning of the sequence predictor as a distinct comparator, with step size/layers fixed on development episodes. Include FS-CAP, MetaDTA if its implementation can be verified, and AdaMBind under the common endpoint/splits/support budget. Never treat published scores from another protocol as matched results. An adapted MINE-style local competence baseline should be labeled as adapted, not a faithful reproduction of all its original settings.

For the geometry capacity control, replace the GVP residual branch with a sequence-only residual network using the same frozen foundation, matching trainable parameters within 5% and applying the same geometry-eligible subset mask. Train it with the same stage-2 protocol. It need not have identical compute, but report parameters and FLOPs. Also include a metadata-only residual control to determine whether coverage/pLDDT explain the gain. Neither control may receive coordinates or graph edges derived from them.

The v3 learned controller is a later comparator if the simple method survives the pilot. Do not add it to the v4.0 forward path to rescue a negative result.

### Metrics

Primary: at K=5, calculate RMSE over each fixed task query panel, average tasks within construct, constructs within protein cluster, then clusters equally. First average repeated draws/seeds consistently for the reported estimator; paired bootstrap resamples protein clusters and retains all their paired method results. Report the exact aggregation implementation, three-seed dispersion and target-macro/pooled alternatives.

Use 10,000 paired cluster-bootstrap resamples for a 95% interval on D-minus-B. The provisional project success rule remains at least 2% relative primary-RMSE reduction and an interval excluding zero. This is a project decision threshold, not a biological standard. Show absolute pKd differences and development-based precision estimates as well.

Secondary: MAE, within-task Spearman where defined, K curves, per-target harm, cap-hit frequency, W/V distributions, structure source/coverage strata, and compound ranking/enrichment at a predefined budget. Zero-shot A versus s reports the value of the fixed geometry prior separately from support adaptation.

The principal test is target-group holdout; shared chemistry across different targets is allowed and disclosed. A separately constructed dual-shift evaluation removes training-scaffold overlap for both support and query molecules. Additionally report query-scaffold separation from support. These are different claims and require separate eligibility manifests.

## 9. Failure behavior and API contract

| Condition | Required behavior |
|---|---|
| No support or all Tanimoto weights zero | Return m0 exactly. |
| One compatible support | Offset may change; differential stays zero. |
| Constant disagreement across support | Differential stays zero. |
| No eligible geometry | Internally g=s; sequence prediction plus compatible support calibration remains available. |
| Incompatible/bounded support | Exclude from v4.0 adapter; return exclusion counts/reasons. |
| Query compound already measured in the same task | Identify as an observed-record lookup; do not score it as a new-compound prediction. |
| Invalid compound/sequence or nonfinite valid model features | Explicit input/model error; no fabricated affinity. |
| Large fitted correction | Apply fixed caps and report cap activation. |

The inference API takes query SMILES, actual construct sequence, an optional mapped structure record, task/assay context and typed support observations. It returns `pKd`, `sequence_pKd`, `geometry_pKd` (null when unavailable), `prior_pKd`, `offset_pKd`, `differential_pKd`, support-used/excluded counts, W, V, structure status and model/data version IDs. It does not return invented confidence percentages or a predicted binding pose. Batch calls must give the same result as separate calls within tolerance.

## 10. Acceptance tests and evidence gates

Before full training:

1. Legacy adapter exposes the real post-CAB feature and passes checkpoint parity.
2. Extension molecular features and predictions are invariant to batch composition, padding and ordering in eval mode.
3. Sequence-token counts, window overlaps and structure-to-construct maps are verified on gaps, insertion codes, ambiguous chains and mutants.
4. No geometry yields exact sequence fallback; zero projection yields g=s; frozen foundation parameters/statistics never change.
5. A nonzero geometry projection allows gradients through the frozen readout/head into GVP. Rotation/translation leaves scalar output unchanged within rtol=1e-4, atol=1e-5, rebuilding the graph and accounting for distance ties.
6. Support order, duplicate grouping, K=0, K=1, constant disagreement, vanishing weights, label-shift invariance and correction bounds pass.
7. Query labels cannot enter support selection, input features, training, routing or checkpoint selection; modifying them changes scoring only.
8. Checkpoint/cache identity includes sequence, compound, structure, graph policy, preprocessing, weights and split versions.

**Pilot decision:** run A–D, individual predictions, simple residual correction and the capacity control first. Examine geometry complementarity on development data. If D only beats A but not B, there is no demonstrated differential-adaptation benefit. If a sequence-capacity control explains the geometry gain, there is no demonstrated geometry-specific benefit. If gains vanish under group holdout, narrow the claim to the supported setting. Do not keep adding modules until one test improves.

**Verification performed during this analysis:** the scalar b/theta solution matched a constrained SciPy optimizer on 200 random problems, with maximum objective difference `1.4210854715202004e-14`. Permutation and common-shift invariance, sparse/missing evidence, missing geometry and total-correction bounds passed. An isolated padded-mean counterexample was checked. These were NumPy/SciPy checks, not execution or training of the neural architecture. PyTorch/real-graph integration, full GVP invariance, checkpoint parity and molecular benchmarks remain unperformed.

## 11. Repository and implementation order

```text
dynacpi/
  configs/v4_exact_kd.yaml
  data/records.py, standardize.py, task_compatibility.py, splits.py, episodes.py
  features/compound.py, esm_cache.py, structure_map.py, backbone_graph.py
  models/legacy_adapter.py, compound_encoder.py, residue_readout.py
  models/sequence_model.py, geometry_branch.py, frozen_pair_predictor.py
  adaptation/centered_ridge.py
  train/train_sequence.py, train_geometry.py, fit_prior.py, select_adapter.py
  eval/cache_predictions.py, factorial.py, metrics.py, bootstrap.py
  tests/
  manifests/
```

Use a separate environment/project from legacy PerceiverCPI. Prefer a Linux CUDA environment on the A100 for the extension. Pin actual tested PyTorch/PyG/scatter/cluster, ESM, RDKit and GVP versions after an installation smoke test; this analysis does not invent a working dependency lock. Vendor only needed upstream components with their licenses and commit references; do not wholesale replace the bundled Chemprop pipeline with current Chemprop.

Implementation order is fixed: **data viability + legacy interface audit → sequence baseline → geometry branch → cached adapter/factorial study → strong comparisons → locked test → frontend integration.** Dataset counts and the hardware/dependency smoke test are the outstanding feasibility checks. A complete specification cannot substitute for them.

## 12. Research claim and scope of novelty

The defensible proposed contribution is an experimentally tested decomposition of support adaptation into shared calibration and a geometry-dependent correction, with explicit behavior when support cannot identify that difference. The estimator is regularized local regression with safeguards; its algebra is not claimed to be a new statistical method.

The strongest paper would establish when geometry helps, when sparse support can detect that benefit, and when adaptation fails, across properly separated targets and chemistry. Measured gains over calibration, residual learners, sequence-capacity controls and modern few-shot baselines are necessary. A new architectural name, extra experts or improved pooled RMSE on a convenient split would not establish that contribution.

The selected architecture is final enough to implement. Hyperparameter performance, data viability, the contribution of geometry and the strength of the novelty claim remain empirical questions. Change the architecture only for a documented integration failure or a development experiment; preserve the untouched final test for the selected version.

## Primary sources and provenance

- [Earlier complete redesign assessment](https://chatgpt.com/library?file_id=libfile_f5f06047a1548191a314d59d77965b78&account_id=personal).
- [PerceiverCPI model and actual forward/featurize paths](https://github.com/dmis-lab/PerceiverCPI/blob/755f076ef51099a90332fd1ec8437f94e3750c0b/chemprop/models/model.py).
- [PerceiverCPI channel-attention implementation](https://github.com/dmis-lab/PerceiverCPI/blob/755f076ef51099a90332fd1ec8437f94e3750c0b/chemprop/models/CAB.py).
- [PerceiverCPI message aggregation](https://github.com/dmis-lab/PerceiverCPI/blob/755f076ef51099a90332fd1ec8437f94e3750c0b/chemprop/models/mpn.py) and [batch graph padding](https://github.com/dmis-lab/PerceiverCPI/blob/755f076ef51099a90332fd1ec8437f94e3750c0b/chemprop/features/featurization.py).
- [PerceiverCPI training weights](https://github.com/dmis-lab/PerceiverCPI/blob/755f076ef51099a90332fd1ec8437f94e3750c0b/chemprop/train/train.py) and [configuration defaults](https://github.com/dmis-lab/PerceiverCPI/blob/755f076ef51099a90332fd1ec8437f94e3750c0b/chemprop/args.py).
- [Official ESM documentation](https://github.com/facebookresearch/esm/blob/main/README.md); model identifier and dimensions verified, weight checksum still to be recorded at extraction.
- [GVP reference featurization](https://github.com/drorlab/gvp-pytorch/blob/316eea7cd6446dd928e5975634152ec3b28e65f5/gvp/data.py). The gap-aware masks, QC thresholds and added quality features above are our specified extensions.
- [MINE](https://doi.org/10.1016/j.ins.2020.07.056), [Local prediction pools](https://doi.org/10.1002/for.3030), [FS-CAP](https://doi.org/10.1021/acs.jcim.4c00485), [AdaMBind](https://doi.org/10.1038/s41467-026-70554-5), [MetaDTA official implementation](https://github.com/Standigm/MetaDTA), and [AssayRouter preprint](https://arxiv.org/abs/2609.37285): prior-art context from the preceding review. MetaDTA's full PDF was not available for direct full-text review.

All layer sizes, bounds, sampling rules, training defaults and decision thresholds in this document are explicit project decisions unless attributed to an upstream source. No trained DynaCPI performance result is asserted.