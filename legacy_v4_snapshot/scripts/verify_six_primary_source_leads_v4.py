"""Human-inspected primary publication and registry evidence leads, 2026-11-06.

Six fixed original BindingDB review tasks (147 source records).
Bibliographic/source-assay evidence only; NO individual affinity labels, drugs,
chemical structures, researcher-approved tasks, model tuning or training.
Reproducible immutable key checks against canonical source-review CSV.
"""
from __future__ import annotations
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
Q=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
OUT=ROOT/"research/source_access_v4"
DATA=[
 {
 "target":"Annexin A3","assay":"50023141_1","n":48,"pmid":"40013713",
 "source_kind":"original experimental paper, independent curated SPR registry",
 "primary":"https://pubs.acs.org/doi/10.1021/acs.jmedchem.4c02403",
 "registry":"https://pubchem.ncbi.nlm.nih.gov/bioassay/2234142/",
 "published_registry_aggregate_count":48,
 "direct_source_finding":"Original article studies ANXA3 binding and lists supporting SPR data. Independently indexed PubChem assay describes a human recombinant ANXA3 C-terminal 6xHis-tagged SPR dissociation-constant experiment and 48 results, matching the fixed local 48-record assay count.",
 "review_stage":"PROMISING_SPR_SOURCE_RECORD_TABLE_MAPPING_PENDING",
 "unresolved":"Original paper/SI numeric rows and individual experimental construct, charge state, assay conditions, source-row correspondence and qualifiers still need independent verification."
 },
 {
 "target":"Serine/threonine-protein kinase RIO2","assay":"50034019_161","n":20,
 "pmid":"22037378",
 "source_kind":"original kinase binding screen, domain construct",
 "primary":"https://www.nature.com/articles/nbt.1990",
 "registry":"https://pubmed.ncbi.nlm.nih.gov/22037378/",
 "direct_source_finding":"Original 2011 kinase interaction paper has supplementary Table 4 reporting binding Kd in nM across 72 molecules and 442 kinase assays. Local record is annotated RIOK2 kinase domain, not necessarily full-length protein.",
 "review_stage":"PUBLISHED_KD_ASSAY_METHOD_LOCATED_DOMAIN_CONSTRUCT_UNVERIFIED",
 "unresolved":"Match each of 20 raw records to the original supplementary table and exact kinase domain, sequence and assay protocol; determine source qualifiers."
 },
 {
 "target":"SRSF protein kinase 3","assay":"50034019_51","n":19,
 "pmid":"22037378",
 "source_kind":"original kinase binding screen, domain construct",
 "primary":"https://www.nature.com/articles/nbt.1990",
 "registry":"https://pubmed.ncbi.nlm.nih.gov/22037378/",
 "direct_source_finding":"Same original Davis 2011 binding-Kd supplementary table. Local records refer specifically to SRPK3 kinase domain.",
 "review_stage":"PUBLISHED_KD_ASSAY_METHOD_LOCATED_DOMAIN_CONSTRUCT_UNVERIFIED",
 "unresolved":"Verify original supplementary numeric row mapping, real recombinant kinase-domain sequence, truncations and assay equivalence for all 19 entries."
 },
 {
 "target":"Interleukin-1 receptor-associated kinase 3","assay":"50034019_522","n":24,
 "pmid":"22037378",
 "source_kind":"original kinase binding screen, domain construct",
 "primary":"https://www.nature.com/articles/nbt.1990",
 "registry":"https://pubmed.ncbi.nlm.nih.gov/22037378/",
 "direct_source_finding":"Same original Davis 2011 supplementary Kd table, but local rows describe IRAK3 kinase domain rather than a source-verified complete protein.",
 "review_stage":"PUBLISHED_KD_ASSAY_METHOD_LOCATED_DOMAIN_CONSTRUCT_UNVERIFIED",
 "unresolved":"Reconcile 24 raw records against published values and physical kinase-domain construct before scientific inclusion."
 },
 {
 "target":"Phosphoglycerate kinase","assay":"50036674_1","n":18,
 "pmid":"9804684",
 "source_kind":"original binding/NMR paper",
 "primary":"https://pubmed.ncbi.nlm.nih.gov/9804684/",
 "registry":"https://pubs.acs.org/doi/10.1021/jm970839y",
 "direct_source_finding":"1998 original study describes ligand binding to phosphoglycerate kinase and NMR-based binding analysis; local assay reports Kd varying across pH 3-11, requiring condition-specific mapping.",
 "review_stage":"BINDING_NMR_SOURCE_FOUND_NUMERIC_METHODS_UNVERIFIED",
 "unresolved":"Independent original full paper table/method review is needed to determine whether 18 numerical Kd values are measured, derived, pH-comparable and construct-aligned."
 },
 {
 "target":"PA-I galactophilic lectin","assay":"50018971_9","n":18,
 "pmid":"26804741",
 "source_kind":"cited publication is a REVIEW, not the underlying experimental report",
 "primary":"https://pubmed.ncbi.nlm.nih.gov/26804741/",
 "registry":"https://pubmed.ncbi.nlm.nih.gov/27488655/",
 "direct_source_finding":"The exact PMID cited in the local raw review metadata (26804741) resolves to a 2016 review article about Pseudomonas infection treatment strategies, NOT an original 18-record LecA Kd study. A different 2016 LecA competitive binding paper is PMID 27488655 and discusses IC50; it cannot be silently substituted as the original Kd source.",
 "review_stage":"ORIGINAL_PER_RECORD_SOURCE_NOT_IDENTIFIED_REVIEW_CITATION_RED_FLAG",
 "unresolved":"Locate actual original experimental citation for assay 50018971_9 and verify direct Kd rather than potency or comparative IC50. Do not mark the existing task approved or replace its original PMID silently."
 },
]
def run():
 before=Q.read_bytes()
 with Q.open(encoding="utf-8-sig",newline="") as f:
  original=list(csv.DictReader(f))
 assert len(original)==30 and len({r["task_id"] for r in original})==30
 assert all(r["curation_decision"]=="UNREVIEWED" for r in original)
 by={(r["target_names"],r["assay_key"]):r for r in original}
 inspected=[]
 for item in DATA:
  key=(item["target"],item["assay"])
  if key not in by:
   raise RuntimeError("Fixed reviewed source identity no longer matches: "+str(key))
  r=by[key]
  if r["pmids"]!=item["pmid"] or int(r["record_count"])!=item["n"]:
   raise RuntimeError("Original source PMID or count mismatch: "+str(key))
  inspected.append({
   "task_id":r["task_id"],"target":r["target_names"],"assay_key":r["assay_key"],
   "source_record_count":int(r["record_count"]),"review_priority":"P2_METADATA_QUEUE_BEFORE_EXTERNAL_METHOD_CORROBORATION",
   "source_evidence_category":item["source_kind"],
   "source_primary_url":item["primary"],
   "independent_registry_or_context_url":item["registry"],
   "source_level_finding":item["direct_source_finding"],
   "verification_stage":item["review_stage"],
   "record_level_remaining":item["unresolved"],
   "human_source_approved":False,
   "all_numeric_values_verified_against_source":False,
   "assayed_physical_construct_verified":False,
   "training_eligibility":False
  })
 if Q.read_bytes()!=before:raise RuntimeError("Source reviewer queue modified")
 total=sum(x["source_record_count"] for x in inspected)
 assert total==147 and len(inspected)==6
 report={
  "status":"SIX_TASKS_EXTERNAL_PRIMARY_SOURCE_EVIDENCE_NO_APPROVAL",
  "observed_utc":datetime.now(timezone.utc).isoformat(),
  "original_review_csv_sha256":hashlib.sha256(before).hexdigest(),
  "target_task_count":6,"candidate_record_count":total,
  "priority_spr_assay":{"assay_key":"50023141_1","records":48,"source_PubChem_AID":"2234142",
       "derivative_registry_entry_count":48},
  "kinase_original_publication_domain_assays":3,
  "kinase_candidate_reactant_count":63,
  "known_review_citation_not_original_experiment":1,
  "source_approved_task_count":0,"scientific_training_authorized":False,
  "chemicals_or_affinity_values_exported":False,
  "records":inspected,
  "limits":[
   "Bibliographic/article/assay-registry corroboration is NOT original row-by-row numerical data verification.",
   "Matching the PubChem assay and local entry counts does not verify every value or correct experimental protein construct.",
   "Kinase domain experiments cannot be silently treated as full-length protein binding.",
   "A cited review article is not evidence its binding constants were personally measured.",
   "All 30 human-review tasks remain UNREVIEWED and primary scientific training is BLOCKED."
  ]
 }
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/"six_task_publication_evidence.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 lines=[
  "# Six candidate binding-assay tasks: publication-source verification (6 November 2026)",
  "",
  "**Source-level public research only. Not an independent per-record Kd audit; zero of 30 tasks scientifically approved.**",
  "",
  f"The fixed research review queue contains {total} original source rows across the six inspected tasks. No chemical structures or numeric affinities were used.",
  "",
  "| Candidate source | Original records | Source-evidence decision |",
  "| --- | ---: | --- |"
 ]
 for x in inspected:
  lines.append(f"| {x['target']} | {x['source_record_count']} | {x['verification_stage']} |")
 lines+=["","## Specific source-level findings",""]
 for x in inspected:
  lines+=["### "+x["target"],"",
      "- Original source: "+x["source_primary_url"],
      "- Additional registry/context: "+x["independent_registry_or_context_url"],
      "- Local assay key: "+x["assay_key"],
      "- Original candidate records: "+str(x["source_record_count"]),
      "",
      x["source_level_finding"],"",
      "**Still unresolved:** "+x["record_level_remaining"],"",
      "**Approved for scientific model fitting:** NO.",""]
 lines+=["## Explicit priority decision","",
    "Start with ANXA3 SPR source methods and individual measurement/table-to-row matching because it has a coherent 48-entry public assay registry and original 2025 SPR supporting data, but do not claim record-level approval from a match in counts.",
    "",
    "Next map kinase-domain assays to the Davis et al. 2011 supplementary binding-Kd table and the real physical kinase domain sequences. Distinguish domain constructs from full-length UniProt references.",
    "",
    "The LecA/PA-I task source PMID 26804741 is a review, not primary experimental data. This is an explicit citation-provenance anomaly and the direct assay source needs discovery; the separate competitive LecA PMID 27488655 must NOT be conflated with the cited review.",
    "",
    "No assay source review or data approval was modified; no model weights were retrained, scientific cohort released, or independent protein-group split frozen.",
    "",
    "All findings are metadata/publication-level and are not scientific validation of an independent model-performance result."
 ]
 (OUT/"SIX_TASK_PUBLICATION_EVIDENCE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print(json.dumps({"status":report["status"],"tasks":6,"records":147,
        "primary_spr_candidates":48,"kinase_domain_candidates":63,
        "review_article_citation_flags":1,"scientific_approvals":0,
        "raw_review_file_unchanged":True},indent=2))
if __name__=="__main__":run()
