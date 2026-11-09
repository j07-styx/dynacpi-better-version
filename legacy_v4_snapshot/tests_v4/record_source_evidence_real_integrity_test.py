"""Real private 921-record source evidence worksheet integrity, not approval."""
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/build_record_source_evidence_worklist_v4.py"
spec=importlib.util.spec_from_file_location("record_evidence_v4",SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

QUEUE=ROOT/"data/audit"/m.QUEUE_NAME
PRIVATE=ROOT/"data/audit"/m.WORKLIST_NAME
old_queue_sha=m.sha(QUEUE)
old_worklist_sha=m.sha(PRIVATE)
summary=m.validate(ROOT)
assert summary["status"]=="REVIEW_WORKLIST_UNREVIEWED_NOT_TRAINING_READY"
assert summary["source_record_count"]==921
assert summary["task_count"]==30
assert summary["self_reported_review_state_counts"]=={"UNREVIEWED":921}
assert summary["fully_declared_task_count"]==0
assert summary["source_scientifically_approved_task_count"]==0
assert summary["main_scientific_training_authorized"] is False
assert summary["human_primary_source_independently_audited"] is False
assert summary["private_local_worklist_not_committed"] is True
assert summary["contains_individual_affinity_values"] is False
assert summary["local_private_worklist_created_this_run"] is False
assert m.sha(QUEUE)==old_queue_sha and m.sha(PRIVATE)==old_worklist_sha

with PRIVATE.open(encoding="utf-8",newline="") as f:
    reader=csv.DictReader(f)
    assert reader.fieldnames==list(m.ALL_FIELDS)
    rows=list(reader)
assert len(rows)==921
assert len({x["reactant_set_id"] for x in rows})==921
assert len({x["task_id"] for x in rows})==30
assert all(x["review_decision"]=="UNREVIEWED" for x in rows)
assert all(not x.get("reviewer_identity","").strip() for x in rows)
assert sum(bool(x["source_method_locator"].strip()) for x in rows)==48
assert sum(bool(x["source_assay_method"].strip()) for x in rows)==48
assert sum("numeric-only candidate" in x["source_table_or_figure_locator"] for x in rows)==6
assert all(not x["original_record_mapping_verified"] and not x["direct_equilibrium_kd_verified"] and not x["physical_construct_verified"] and not x["assay_conditions_compatible"] for x in rows)
assert all("pKd" not in x and "Kd (nM)" not in x and "smiles" not in x for x in rows)

lineage=json.loads((ROOT/"research/source_access_v4/record_level_provenance_30.json").read_text(encoding="utf-8"))
assert lineage["source_reactant_references"]==summary["source_record_count"]==921
assert lineage["review_queue_sha256"]==summary["input_review_queue_sha256"]
assert lineage["critical_lineage_flags"]=={}
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"

result={
    "status":"PASS_PRIVATE_RECORD_WORKLIST_UNREVIEWED_NO_TRAINING",
    "source_record_count":921,"task_count":30,
    "local_private_review_file_sha256":old_worklist_sha,
    "canonical_review_queue_unchanged":True,
    "individual_affinity_values_in_worklist":False,
    "source_approved_tasks":0,"scientific_gate":"BLOCKED"}
(ROOT/"data/audit/source_record_evidence_real_integrity.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k!="local_private_review_file_sha256"},indent=2))
