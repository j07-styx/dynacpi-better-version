"""Fail-closed triage contract; warning flags must never become approvals."""
from pathlib import Path
import csv,hashlib,importlib.util,json,sys

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/triage_30_assay_metadata_v4.py"
spec=importlib.util.spec_from_file_location("triage30",SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module.extract_regions("human NPR-C (27 to 541 residues)") == [
    {"start":27,"end":541,"nominal_length":515,"source":"ASSAY_DESCRIPTION_ONLY"}]
assert module.extract_regions("pH (3.0-11.0) and temperature 37 C")==[]
assert module.extract_regions("VHL (54 to 213 residues)/Elongin B/C")[0]["nominal_length"]==160
assert module.extract_regions("compound tested on kinase domain, no region available")==[]
assert module.extract_regions("amyloid beta (1 to 40) fibrils")[0]["nominal_length"]==40

Q=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
before=Q.read_bytes()
rows=list(csv.DictReader(Q.open(encoding="utf-8-sig",newline="")))
assert len(rows)==30 and all(r["curation_decision"]=="UNREVIEWED" for r in rows)
out=json.loads((ROOT/"research/source_access_v4/assay_metadata_risk_triage_30.json").read_text(encoding="utf-8"))
assert out["status"]=="PRELIMINARY_METADATA_RISK_TRIAGE_NOT_SOURCE_ADJUDICATION"
assert out["task_count"]==30 and out["approved_tasks"]==0 and out["reviewed_tasks"]==0
assert out["raw_affinity_values_used"] is False
assert out["canonical_review_csv_sha256"]==hashlib.sha256(before).hexdigest()
assert out["metadata_read_only"] is True
assert len(out["entries"])==30 and len({x["task_id"] for x in out["entries"]})==30
assert sum(out["priority_counts"].values())==30
assert out["priority_counts"]=={
   "P0_REVIEW_METADATA_SEMANTICS":4,
   "P1_REVIEW_CONSTRUCT_AND_ENDPOINT":12,
   "P2_STANDARD_PRIMARY_SOURCE_REVIEW":14
}
assert out["risk_flag_counts"]["REGION_SHORTER_THAN_CANONICAL_REFERENCE"]==6
assert out["risk_flag_counts"]["NUMERIC_REGION_END_EXCEEDS_TARGET_REFERENCE_CHECK_CHAIN_ASSOCIATION"]==1
assert out["risk_flag_counts"]["PRIMARY_SOURCE_FRAGMENT_VS_REFERENCE_FLAG"]==2
assert out["risk_flag_counts"]["KD_FIELD_DESCRIBED_AS_HALF_EFFECTIVE_CONCENTRATION"]==1
assert out["risk_flag_counts"]["POTENTIAL_TARGET_DESCRIPTION_SEMANTIC_MISMATCH"]==1

byname={x["target_name"]:x for x in out["entries"]}
assert "KD_FIELD_DESCRIBED_AS_HALF_EFFECTIVE_CONCENTRATION" in byname[
   "Cystic fibrosis transmembrane conductance regulator"]["metadata_risk_flags"]
assert "POTENTIAL_TARGET_DESCRIPTION_SEMANTIC_MISMATCH" in byname[
   "Glucagon-like peptide 1 receptor"]["metadata_risk_flags"]
assert "DESCRIPTION_MISSING" in byname["Programmed cell death 1 ligand 1"]["metadata_risk_flags"]
assert "DOI_ONLY_SOURCE_METHODS_PENDING" in byname["Carbonic anhydrase 2"]["metadata_risk_flags"]
elongin=byname["Elongin-C"]
assert "NUMERIC_REGION_END_EXCEEDS_TARGET_REFERENCE_CHECK_CHAIN_ASSOCIATION" in elongin["metadata_risk_flags"]
assert elongin["description_numeric_regions"][0]["end"]==213
assert elongin["candidate_protein_reference_length"]==112

for item in out["entries"]:
 assert item["curation_decision"]=="UNREVIEWED"
 assert item["all_assay_validity_attributes_unverified"] is True
 assert item["experimental_kd_record_approved"] is False
 assert item["source_construct_sequence_approved"] is False
 assert "median_pKd" not in item and "raw_affinity" not in item
 assert item["original_assay_description_hash"] and len(item["original_assay_description_hash"])==64

assert hashlib.sha256(Q.read_bytes()).hexdigest()==out["canonical_review_csv_sha256"]
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"
report={
 "status":"PASS_30_TASK_TRIAGE_METADATA_ONLY",
 "task_count":30,
 "priority_counts":out["priority_counts"],
 "canonical_review_unchanged":True,
 "scientifically_approved_tasks":0,
 "source_approval_gate":"BLOCKED",
 "synthetic_region_parser_tests":True,
 "warning":"Flags are nonbinding review leads, not scientific defect verdicts."}
(ROOT/"data/audit/assay_metadata_triage_integrity.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
