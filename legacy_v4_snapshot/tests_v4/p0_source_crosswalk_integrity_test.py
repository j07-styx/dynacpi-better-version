"""Fail closed on four P0 source-metadata evidence leads; no assay approval.

The existence of bibliographic sources and abstract/registry observations
is NOT a source-reviewed exact-Kd cohort or independent model benchmark.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"research/source_access_v4"
A=ROOT/"data/audit"
Q=A/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
source=Q.read_bytes()
q={r["task_id"]:r for r in csv.DictReader(Q.open(encoding="utf-8-sig",newline=""))}
d=json.loads((R/"p0_source_crosswalk.json").read_text(encoding="utf-8"))
triage=json.loads((R/"assay_metadata_risk_triage_30.json").read_text(encoding="utf-8"))
idx=json.loads((R/"source_access_index.json").read_text(encoding="utf-8"))
assert len(q)==30 and all(x["curation_decision"]=="UNREVIEWED" for x in q.values())
assert d["status"]=="P0_EXTERNAL_SOURCE_CORROBORATION_LEADS_NO_SCIENTIFIC_APPROVAL"
assert d["task_count"]==4 and d["approved_count"]==0 and d["rejected_count"]==0
assert d["training_permission"] is False and d["compound_affinity_values_read"] is False
assert d["current_original_review_csv_sha256"]==hashlib.sha256(source).hexdigest()
assert d["triage_source_sha256"]==hashlib.sha256((R/"assay_metadata_risk_triage_30.json").read_bytes()).hexdigest()
assert d["access_index_source_sha256"]==hashlib.sha256((R/"source_access_index.json").read_bytes()).hexdigest()
assert triage["canonical_review_csv_sha256"]==d["current_original_review_csv_sha256"]
assert idx["review_file_sha256"]==d["current_original_review_csv_sha256"]
assert triage["priority_counts"]=={
    "P0_REVIEW_METADATA_SEMANTICS":4,
    "P1_REVIEW_CONSTRUCT_AND_ENDPOINT":12,
    "P2_STANDARD_PRIMARY_SOURCE_REVIEW":14}
ids={r["task_id"] for r in triage["entries"] if r["manual_review_priority"]=="P0_REVIEW_METADATA_SEMANTICS"}
assert ids=={r["task_id"] for r in d["records"]}
assert len(d["records"])==4
assert idx["source_status_counts"]["PRIMARY_DOI_ONLY_METHODS_PENDING"]==1
assert "MISSING_PRIMARY_SOURCE_IDENTIFIER" not in idx["source_status_counts"]
byname={x["recorded_target_name"]:x for x in d["records"]}
ca=byname["Carbonic anhydrase 2"]
assert ca["candidate_source_category"]=="PRIMARY_DOI_ONLY_METHODS_PENDING"
assert any(i.startswith("DOI:") for i in ca["source_ids"])
cf=byname["Cystic fibrosis transmembrane conductance regulator"]
assert cf["observation_source_tier"]=="PUBMED_PUBLICATION_ABSTRACT"
assert "PMID:12852954" in cf["source_ids"]
gl=byname["Glucagon-like peptide 1 receptor"]
assert gl["observation_source_tier"]=="PUBCHEM_BINDINGDB_ASSAY_REGISTRY"
assert "BINDINGDB_ASSAY:11490_1" in gl["source_ids"]
pd=byname["Programmed cell death 1 ligand 1"]
assert "US20240010642" in pd["source_ids"]
for record in d["records"]:
    assert q[record["task_id"]]["assay_key"]==record["assay_key"]
    assert record["review_decision"]=="UNREVIEWED"
    assert record["physical_assay_construct_verified"] is False
    assert record["each_numeric_Kd_measurement_verified"] is False
    assert record["human_review_approval"] is False
    assert record["record_exclusion_decision"] is None
    assert len(record["required_record_level_checks"])>=2
    assert "measured_pKd" not in record and "raw_affinity" not in record
assert "DOI located" in (R/"REVIEW_ACTIONS.md").read_text(encoding="utf-8")
assert "PRIMARY_DOI_ONLY_METHODS_PENDING" in (R/"SOURCE_ACCESS_SUMMARY.md").read_text(encoding="utf-8")
assert Q.read_bytes()==source
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"
result={
 "status":"PASS_4_P0_SOURCE_LEADS_NOT_SCIENTIFIC_APPROVAL",
 "p0_tasks":4,"corrected_doi_only_source_count":1,
 "source_review_queue_sha256_unchanged":True,
 "no_numeric_affinity_values_accessed":True,
 "source_approved_tasks":0,"scientific_gate":"BLOCKED"
}
(A/"p0_source_crosswalk_integrity.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
