"""Create a provenance-linked, read-only P0 assay source-evidence brief.

These four cases are literature/registry LEADS, NOT researcher-adjudicated exact
Kd datasets. The builder does not access compound-level affinity values, infer
assay approval, modify the canonical review CSV, or access trial/test data.

Source assertions reflect the 2026-11-06 external source inspection snapshot.
A qualified reviewer must independently confirm sources and record-level data.
"""
from __future__ import annotations
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"data/audit"
SOURCE=ROOT/"research/source_access_v4"
QUEUE=AUDIT/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
TRIAGE=SOURCE/"assay_metadata_risk_triage_30.json"
ACCESS=SOURCE/"source_access_index.json"
OUT=SOURCE/"p0_source_crosswalk.json"
MD=SOURCE/"P0_SOURCE_CROSSWALK.md"

# Verbatim publication and registry TEXT is not reproduced.
# These are structured, limited observations from primary identifiers.
FACTS={
 "00711d68562f88f0479db57d7cb5c2f79a802c306a489b788c96c75d06c79c5b":{
  "target":"CFTR","independent_source_type":"PUBMED_PUBLICATION_ABSTRACT",
  "source_ids":["PMID:12852954","DOI:10.1016/S0960-894X(03)00482-7"],
  "source_locator":"pubmed.ncbi.nlm.nih.gov/12852954",
  "observation":"The corresponding publication abstract describes a cell-based functional transport response and half-maximal stimulation, not a directly demonstrated equilibrium Kd for the selected records.",
  "specific_risk":"FUNCTIONAL_RESPONSE_COULD_BE_MISREPRESENTED_AS_KD",
  "open_questions":["Verify which original published table matches the exact BindingDB reactant IDs.",
                    "Determine whether any independent biochemical binding Kd assay exists.",
                    "Reconcile recorded protein species and experimental construct with target sequence."]
 },
 "3a9e7fe2a56588436c4f0cbde17aea8b8099b57fc56ddac4d72e338132162881":{
  "target":"Carbonic anhydrase II",
  "independent_source_type":"CROSSREF_BIBLIOGRAPHIC_METADATA",
  "source_ids":["DOI:10.1016/0960-894X(96)00069-8"],
  "source_locator":"doi.org/10.1016/0960-894X(96)00069-8",
  "observation":"The DOI in the original review queue resolves to a 1996 publication about carbonic-anhydrase binding-site probes. This is an existing DOI-only source locator, not a missing source identifier.",
  "specific_risk":"PRIMARY_METHODS_NOT_REVIEWED_AND_KD_SEMANTICS_UNVERIFIED",
  "open_questions":["Obtain and inspect the publication methods and tables legally.",
                    "Verify how the individual values were determined and whether they are directly measured exact Kd.",
                    "Verify actual protein construct and comparable measurement conditions."]
 },
 "83c3383d774c088ede95378857d532d6106b6543bd0116618c061fac063be820":{
  "target":"PD-L1",
  "independent_source_type":"PRIMARY_PATENT_METADATA",
  "source_ids":["US20240010642"],
  "source_locator":"patents.google.com/patent/US20240010642A1/en",
  "observation":"The primary patent titled Modulators of programmed death-ligand-1 is located; the candidate assay description remains missing from the original review record.",
  "specific_risk":"MISSING_ASSAY_PROTOCOL_AND_RECORD_SOURCE_METHOD_MAPPING",
  "open_questions":["Locate the specific patent example and measurement table for the candidate assay.",
                    "Verify primary experimental method, construct and Kd endpoint.",
                    "Resolve record-level provenance and qualifying inequalities."]
 },
 "da40c59768b57b10fa19165e2783e244b9a1ebe7f116999933408861e6e09993":{
  "target":"GLP-1 receptor",
  "independent_source_type":"PUBCHEM_BINDINGDB_ASSAY_REGISTRY",
  "source_ids":["US11767332","PUBCHEM_AID:1919769","BINDINGDB_ASSAY:11490_1"],
  "source_locator":"pubchem.ncbi.nlm.nih.gov/bioassay/1919769",
  "observation":"The registry explicitly links to BindingDB assay 11490_1 but describes a colorimetric glucose-binding experiment involving boronate/carbohydrate, while the candidate target is recorded as GLP-1 receptor.",
  "specific_risk":"ASSAY_PARTICIPANT_OR_TARGET_ASSOCIATION_REQUIRES_SOURCE_RECONCILIATION",
  "open_questions":["Verify whether GLP-1 receptor was physically present in the measured mixture.",
                    "Check the original patent method and per-record table for the exact binding participant.",
                    "Verify whether the Kd quantity represents carbohydrate interaction rather than GLP-1 receptor binding."]
 }
}

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
 before=QUEUE.read_bytes()
 with QUEUE.open(encoding="utf-8-sig",newline="") as f:
  queue={r["task_id"]:r for r in csv.DictReader(f)}
 triage=json.loads(TRIAGE.read_text(encoding="utf-8"))
 access=json.loads(ACCESS.read_text(encoding="utf-8"))
 assert len(queue)==30 and all(r["curation_decision"]=="UNREVIEWED" for r in queue.values())
 assert triage["canonical_review_csv_sha256"]==hashlib.sha256(before).hexdigest()
 assert access["review_file_sha256"]==hashlib.sha256(before).hexdigest()
 tasks={r["task_id"]:r for r in triage["entries"]}
 ids={i for i,r in tasks.items() if r["manual_review_priority"]=="P0_REVIEW_METADATA_SEMANTICS"}
 if ids!=set(FACTS):raise RuntimeError("P0 task identities changed: source evidence needs re-review")
 source_categories={r["task_id"]:r["evidence_access_status"] for r in access["tasks"]}
 assert source_categories["3a9e7fe2a56588436c4f0cbde17aea8b8099b57fc56ddac4d72e338132162881"]=="PRIMARY_DOI_ONLY_METHODS_PENDING"
 out=[]
 for task in sorted(FACTS):
  row=queue[task];v=FACTS[task]
  assert row["assay_key"]==tasks[task]["assay_key"]
  assert row["target_names"]==tasks[task]["target_name"]
  if v["target"]=="CFTR":
   assert row["pmids"]=="12852954"
   assert "half effective concentration" in row["assay_description"].lower()
  if v["target"]=="Carbonic anhydrase II":
   assert row["article_dois"].casefold()=="10.1016/0960-894x(96)00069-8"
  if v["target"]=="PD-L1":
   assert row["patent_numbers"]=="US20240010642"
   assert row["assay_description"]=="Not found"
  if v["target"]=="GLP-1 receptor":
   assert row["assay_key"]=="11490_1" and row["patent_numbers"]=="US11767332"
   assert "glucose" in row["assay_description"].lower()
  out.append({
   "task_id":task,"assay_key":row["assay_key"],
   "candidate_source_category":source_categories[task],
   "recorded_target_name":row["target_names"],
   "observation_source_tier":v["independent_source_type"],
   "source_ids":v["source_ids"],"source_locator":v["source_locator"],
   "source_observation_paraphrase":v["observation"],
   "investigate_risk":v["specific_risk"],
   "required_record_level_checks":v["open_questions"],
   "physical_assay_construct_verified":False,
   "each_numeric_Kd_measurement_verified":False,
   "human_review_approval":False,"review_decision":"UNREVIEWED",
   "record_exclusion_decision":None
  })
 if QUEUE.read_bytes()!=before:raise PermissionError("Canonical review metadata changed")
 report={
  "status":"P0_EXTERNAL_SOURCE_CORROBORATION_LEADS_NO_SCIENTIFIC_APPROVAL",
  "created_utc":datetime.now(timezone.utc).isoformat(),
  "current_original_review_csv_sha256":hashlib.sha256(before).hexdigest(),
  "triage_source_sha256":sha(TRIAGE),"access_index_source_sha256":sha(ACCESS),
  "task_count":4,"approved_count":0,"rejected_count":0,
  "compound_affinity_values_read":False,"training_permission":False,
  "sources_last_manually_checked_date":"2026-11-06",
  "records":out,
  "interpretation":"The two source-method mismatch leads should be resolved at record level; DOI and patent discovery are only locators. None authorizes source-reviewed Kd training.",
  "cautions":["The publication abstract may not describe every measurement reported elsewhere.",
              "A registry target annotation and actual assay participant can differ.",
              "DOI or patent matching is bibliographic identity only, not verified experimental Kd.",
              "All thirty candidate assays remain UNREVIEWED; zero source-approved study tasks."]
 }
 OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 lines=["# Four P0 BindingDB source-evidence crosswalk (6 November 2026)","",
  "**Source-level leads, not researcher-approved experimental data.** No compound-level affinity values were read. All thirty canonical review decisions are unchanged and UNREVIEWED.","",
  "| Candidate | Source evidence level | Verification concern |",
  "| --- | --- | --- |"]
 for rec in out:
  lines.append("| "+rec["recorded_target_name"]+" | "+rec["observation_source_tier"]+" | "+rec["investigate_risk"]+" |")
 lines+=["","## Source-specific observations and next checks",""]
 for rec in out:
  lines+=["### "+rec["recorded_target_name"],"",
   "Source identifiers: "+", ".join(rec["source_ids"]),
   "Source locator: "+rec["source_locator"],"",
   rec["source_observation_paraphrase"],"",
   "Unresolved review requirements:"]
  lines.extend("- "+q for q in rec["required_record_level_checks"])
  lines+=["","**Approved: NO. Individual Kd measurements verified: NO.**",""]
 lines+=["## Corrected bibliographic interpretation","",
  "The carbonic anhydrase II row has a DOI but no PMID/patent field. A prior metadata-only index incorrectly called it missing a source identifier. The corrected index assigns PRIMARY_DOI_ONLY_METHODS_PENDING; this does not imply the actual article methods have been reviewed.",
  "",
  "Do not automatically reject the CFTR or GLP-1 receptor rows without original table-to-record checks. Their source-level red flags warrant urgent review, not an inferred numeric validation result.",
  "",
  "No source body text, compounds, potency values, or affinity-label arrays appear in this artifact."
 ]
 MD.write_text("\n".join(lines)+"\n",encoding="utf-8")
 print(json.dumps({"status":report["status"],"n_P0":4,
   "review_queue_unchanged":True,"approved":0,
   "doi_only_corrected":True,"source_tiers":[x["observation_source_tier"] for x in out]},indent=2))
if __name__=="__main__":run()
