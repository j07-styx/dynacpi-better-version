# TransferCPI: source-family evidence continuation and historical training lock

**8 November 2026 | DynaCPI v4 | technical research continuation, no new model training**

## Reason for intervention

The previous metadata-only preflight identified **seven** publication-source families crossing the provisional BindingDB training, inner-development and reserved-test protein component roles (25 affected protein components). The historical matched five-arm TransferCPI/PerceiverCPI experiments were completed **before** this source-family conflict was identified. They remain *exploratory pilot results* and must not be promoted as publication-independent generalization evidence.

The scientific cohort gate remains `BLOCKED_UNAPPROVED_NOT_TRAINING_READY`. Source bibliographic indexing, keyword tags and PubChem/patent assay annotations are **not** experimental row or protein construct review.

## Work executed

1. Prepared a confidential, metadata-only seven-family review queue based entirely on the already recorded source-family preflight and the earlier confidential 25-component role conflict worksheet. **Seven of seven families have at least one sampled shared source identifier**, while **five** have a ranked pre-existing 40-lead source-review candidate and **two** have no prior ranked review lead. The two additional bibliographic source identifiers are recorded locally for primary-method inspection. Source ID examples are a truncated subset, not complete family provenance.
2. Checked public bibliographic abstracts/assay annotations for the seven source leads. Notable general concerns include tagged or truncated protein constructs, combined experimental methods in one source, and multi-target study families. Nothing was promoted to a verified BindingDB source row. The source methods worksheet, exact cross-role family mapping and reviewer action checklist remain local under `data/audit/`.
3. Added `dynacpi/provisional_transfer_training_lock.py` and integrated it into the historical `scripts/train_transfer_cpi_dev.py` entrypoint **before GPU leasing, the affinity-label loader and experiment output creation**. The lock verifies source-audit and split-manifest checksums, replays the metadata-only overlap report, and fail-closes on absent or changed files.
4. Deliberately **retired new runs of the old provisional TransferCPI pilot protocol**. Even a future change to the metadata role conflict is not an authorization to retrain on the old unreviewed cohort. Any approved future experiment needs a separate source-reviewed, preregistered protocol and training entrypoint.
5. Preserved all existing checkpoints, five-arm comparisons and raw/quarantined scientific data without rerunning model training, evaluating any prospective test outcomes or assigning any source approval.

## Automated verification

- `tests_v4/source_family_evidence_queue_test.py`: 3 tests; check metadata projection excludes labels/structures, false family mappings fail, saved queue remains unreviewed.
- `tests_v4/provisional_transfer_training_lock_test.py`: 6 tests; reject real overlap, changed/missing manifests, unapproved synthetic cases and new train invocation before GPU/labels.
- `tests_v4/provisional_source_split_leakage_test.py`: 7 tests; previously established source-role audit.
- `tests_v4/bindingdb_publication_group_preflight_test.py`: 7 tests; existing source-family guard.
- `tests_v4/scientific_cohort_gate_test.py`: pass, real approval remains blocked.

These are software/data-guard tests, not experimental source validation.

## Confidential local outputs — never commit

- `data/audit/PROVISIONAL_SOURCE_SPLIT_REVIEW_PRIVATE_20261108.csv` — previously generated role-sensitive 25-component conflict mapping.
- `data/audit/PROVISIONAL_SOURCE_FAMILY_EVIDENCE_QUEUE_PRIVATE_20261108.json` — seven source-family review records, sampled source identifiers and reviewer checklist.
- `data/audit/PRIMARY_SOURCE_SEVEN_FAMILIES_METHODS_PRIVATE_20261108.md` — source abstracts and assay-method flags, still unverified.

The public source-only review script `scripts/build_source_family_evidence_queue_private_dev.py` writes exclusively to a local confidential path and refuses overwrites. Do not move source-family role mappings into public reports or repository commits.

## Scientific decision and next gate

**STOP new training on this provisional split. KEEP prospective-test outcomes unexamined.**

Independent source reviewers must now inspect the seven source families' original experiments, confirm exact constructs and readouts, reconcile source tables with BindingDB rows and record evidence-backed decisions. Then a **new** protein-homology-plus-publication-source family disjoint study design must be preregistered and frozen without inspecting the previous prospective-test outcomes. Validate that all prospective training entrypoints use the scientific cohort gate **and** the source-family split guard. Do not silently repair role memberships or reselect on previously inspected DEV outcomes.

**Current assessment:** source identification completed at bibliography/assay annotation level; record-level adjudication **NOT STARTED**; all tasks remain UNREVIEWED; no new model or score claim.
