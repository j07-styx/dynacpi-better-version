"""Real local 30-task reactant lineage integrity, never scientific approval."""
import csv, hashlib, importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/audit_30_record_level_provenance_v4.py"
spec=importlib.util.spec_from_file_location("dynacpi_source_lineage",SCRIPT)
audit_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)

QUEUE=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
before=QUEUE.read_bytes()
review=list(csv.DictReader(QUEUE.open(encoding="utf-8-sig",newline="")))
assert len(review)==30 and all(x["curation_decision"]=="UNREVIEWED" for x in review)

report=audit_module.audit(ROOT)
assert report["status"]=="TECHNICAL_SOURCE_LINEAGE_AUDIT_NOT_SCIENTIFIC_APPROVAL"
assert (report["tasks"],report["compound_groups"],report["source_reactant_references"])==(30,919,921)
assert report["original_reactant_ids_found"]==921
assert report["unaggregated_reactant_ids_found"]==921
assert report["duplicate_source_id_rows"]==report["duplicate_unaggregated_id_rows"]==0
assert report["critical_lineage_flags"]=={}
assert report["summary_counts"]["RAW_KD_STRICT_POSITIVE_NUMERIC"]==921
assert report["summary_counts"]["COMPOUND_GROUPS_WITH_REPLICATES"]==2
assert report["approved_tasks"]==0 and report["main_training_permitted"] is False
assert report["source_biological_review_completed"] is False
assert hashlib.sha256(before).hexdigest()==report["review_queue_sha256"]
assert QUEUE.read_bytes()==before
assert len(report["per_task"])==30
assert all(x["review_state"]=="UNREVIEWED" and x["scientifically_approved"] is False
           and x["physical_assayed_construct_verified"] is False
           and x["experimentally_exact_kd_verified"] is False
           for x in report["per_task"])
serialized=(ROOT/"research/source_access_v4/record_level_provenance_30.json").read_text(encoding="utf-8")
assert '"Kd (nM)"' not in serialized and '"median_pKd"' not in serialized
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"
print(json.dumps({"status":"PASS_TECHNICAL_REACTANT_LINEAGE_NOT_SOURCE_APPROVED",
    "tasks":30,"groups":919,"reactants":921,"missing_records":0,
    "critical_issues":0,"scientifically_approved":0,
    "cohort_gate":"BLOCKED"},indent=2))
