"""Verify factual provenance flags while keeping the scientific gate closed."""
from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"data/audit"
SOURCE=ROOT/"research/source_access_v4"
file=SOURCE/"construct_alignment_risk_screen.json"
d=json.loads(file.read_text(encoding="utf-8"))
assert d["status"]=="PASS_SOURCE_LOCATOR_CONSTRUCT_RISK_ONLY_NOT_APPROVED"
assert d["reconciled_task_count"]==2 and d["scientifically_approved_task_count"]==0
assert d["full_30_task_review_queue_unchanged"] is True
assert d["cohort_training_gate"]=="BLOCKED"
review=AUDIT/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
assert hashlib.sha256(review.read_bytes()).hexdigest()==d["canonical_review_sha256"]
rows=list(csv.DictReader(review.open(encoding="utf-8-sig",newline="")))
assert len(rows)==30 and all(x["curation_decision"]=="UNREVIEWED" for x in rows)
assert len({r["task_id"] for r in d["records"]})==2
expected={"P17342":(541,27,541),"O14786":(923,273,427)}
for item in d["records"]:
    acc=item["protein_accession"]
    full,start,end=expected[acc]
    assert item["candidate_full_reference_length"]==full
    assert item["paper_reported_construct_region"]==[start,end]
    assert item["nominal_fragment_length"]==end-start+1
    assert item["reference_vs_reported_region_differs"] is True
    assert item["nominal_fragment_length"]<full
    assert item["human_review_status"]=="UNREVIEWED"
    assert all(item[k] is False for k in ("scientific_approval",
        "verified_physical_construct_sequence","individual_affinity_values_reviewed",
        "source_assay_compatibility_verified"))
note=(SOURCE/"PRELIMINARY_CONSTRUCT_ALIGNMENT_REVIEW.md").read_text(encoding="utf-8")
assert "PMC9014859" in note and "PMC5957473" in note
assert "UNVERIFIED" in note and "not an approved" in note.lower()
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"
out={"status":"PASS_CONSTRUCT_RISK_ENGINEERING_ONLY",
     "source_reviewed_task_count":0,
     "source_locators_documented":2,
     "canonical_review_queue_unchanged":True,
     "cohort_training_gate":"BLOCKED"}
(AUDIT/"construct_alignment_risk_integrity.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
