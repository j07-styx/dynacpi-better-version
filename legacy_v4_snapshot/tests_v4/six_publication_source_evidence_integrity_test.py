"""Source-publication lead integrity, no scientific source approval."""
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
q=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
before=q.read_bytes()
r=json.loads((ROOT/"research/source_access_v4/six_task_publication_evidence.json").read_text(encoding="utf-8"))
assert r["status"]=="SIX_TASKS_EXTERNAL_PRIMARY_SOURCE_EVIDENCE_NO_APPROVAL"
assert r["target_task_count"]==6 and r["candidate_record_count"]==147
assert r["priority_spr_assay"]["records"]==48
assert r["priority_spr_assay"]["derivative_registry_entry_count"]==48
assert r["kinase_original_publication_domain_assays"]==3
assert r["kinase_candidate_reactant_count"]==63
assert r["known_review_citation_not_original_experiment"]==1
assert r["source_approved_task_count"]==0
assert r["scientific_training_authorized"] is False
assert r["chemicals_or_affinity_values_exported"] is False
assert r["original_review_csv_sha256"]==hashlib.sha256(before).hexdigest()
rows=list(csv.DictReader(q.open(encoding="utf-8-sig",newline="")))
by_id={x["task_id"]:x for x in rows}
assert len(rows)==30 and all(x["curation_decision"]=="UNREVIEWED" for x in rows)
assert len(r["records"])==6 and len({x["task_id"] for x in r["records"]})==6
for x in r["records"]:
    old=by_id[x["task_id"]]
    assert (x["target"],x["assay_key"])==(old["target_names"],old["assay_key"])
    assert x["source_record_count"]==int(old["record_count"])
    assert x["human_source_approved"] is False
    assert x["all_numeric_values_verified_against_source"] is False
    assert x["assayed_physical_construct_verified"] is False
    assert x["training_eligibility"] is False
    assert x["source_primary_url"].startswith("https://")
    assert x["independent_registry_or_context_url"].startswith("https://")
assert next(x for x in r["records"] if x["target"]=="PA-I galactophilic lectin")["verification_stage"]=="ORIGINAL_PER_RECORD_SOURCE_NOT_IDENTIFIED_REVIEW_CITATION_RED_FLAG"
assert q.read_bytes()==before
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"
print(json.dumps({"status":"PASS_PUBLICATION_EVIDENCE_LEADS_ONLY","n_tasks":6,"candidate_records":147,"source_approved":0,"science_gate":"BLOCKED"}))
