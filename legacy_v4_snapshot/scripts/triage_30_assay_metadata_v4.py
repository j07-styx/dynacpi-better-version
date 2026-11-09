"""Read-only triage of 30 preliminary BindingDB assay descriptions.

Metadata warning signals are NOT biological approval/rejection, do not access
individual compound affinities, and MUST NOT change the human review CSV.
All candidate assay tasks remain unapproved pending primary-source review.
"""
from __future__ import annotations
import csv
import hashlib
import json
import re
import sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
REFS=ROOT/"data/audit/bindingdb_assay_review_30_uniprot_construct_screen.json"
INDEX=ROOT/"research/source_access_v4/source_access_index.json"
KNOWN=ROOT/"research/source_access_v4/construct_alignment_risk_screen.json"
OUT=ROOT/"research/source_access_v4"
RANGE=re.compile(r"(?<!\d)(\d{1,4})\s*(?:to|-|\u2013|\u2014)\s*(\d{1,4})\s+residues?\b",re.I)
PAREN_REGION=re.compile(r"(?:amyloid[\s-]*beta|protein|peptide)\s*\(\s*(\d{1,4})\s*(?:to|-|\u2013)\s*(\d{1,4})\s*\)",re.I)
DOMAIN=re.compile(r"\b(?:domain|fragment|truncat(?:ed|ion)|catalytic domain|kinase domain)\b",re.I)
TAG=re.compile(r"\b(?:His[\s-]?\d*[\s-]?tag(?:ged)?|GST[\s-]?tag(?:ged)?|His\d+[\s-]?tagged|NT647|biotinylated|fluorescen\w*[\s-]?tagged)\b",re.I)
MULTI=re.compile(r"\b(?:Elongin|LSD1/CoREST|protein complex|heterotrimer|heterodimer|fibrils|subunit|cofactor|NADPH)\b",re.I)
ACTIVITY=re.compile(r"\b(?:TR[\s-]?FRET|hydrolysis|catalys(?:is|es)|reaction rates|colorimetric|half effective concentration|AMP production|phosphorolysis|enzyme inhibition|inhibition affinity)\b",re.I)
COMPETITION=re.compile(r"\b(?:competitive binding|competitive fluorescence|displacement assay|SPR inhibition in solution|competition)\b",re.I)
MISSING=re.compile(r"^\s*(?:not found|unknown|na|n/a|none|)\s*$",re.I)
EXPLICIT_BIOCHEM=re.compile(r"\b(?:binding affinity|dissociation constant|equilibrium dissociation|surface plasmon resonance|Biacore|isothermal titration calorimetry)\b",re.I)
SPECIES=re.compile(r"\b(?:human|mouse|rat|bovine|pseudomonas|escherichia coli)\b",re.I)

def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def source_rows():
 raw=QUEUE.read_bytes()
 rows=list(csv.DictReader(QUEUE.open(encoding="utf-8-sig",newline="")))
 if len(rows)!=30 or len({r["task_id"] for r in rows})!=30:
  raise RuntimeError("Expected exactly 30 unique preselected review rows")
 if any(r.get("curation_decision")!="UNREVIEWED" for r in rows):
  raise PermissionError("Human review state has changed; do not relabel")
 return raw,rows

def extract_regions(text):
 hits=[]
 for m in (*list(RANGE.finditer(text)),*list(PAREN_REGION.finditer(text))):
  start,end=map(int,m.groups())
  if start<=0 or end<start or end-start>5000:continue
  hits.append({"start":start,"end":end,"nominal_length":end-start+1,
               "source":"ASSAY_DESCRIPTION_ONLY"})
 return sorted({(x["start"],x["end"]):x for x in hits}.values(),key=lambda d:(d["start"],d["end"]))

def flags_for(row,full_len,source_status,known_region=None):
 desc=row["assay_description"]
 flags=set();details=[]
 regions=extract_regions(desc)
 if MISSING.fullmatch(desc):flags.add("DESCRIPTION_MISSING")
 if ACTIVITY.search(desc):flags.add("SURROGATE_ACTIVITY_OR_ENDPOINT_LANGUAGE")
 if COMPETITION.search(desc):flags.add("COMPETITION_OR_DISPLACEMENT_PROTOCOL")
 if DOMAIN.search(desc):flags.add("DOMAIN_OR_FRAGMENT_LANGUAGE")
 if TAG.search(desc):flags.add("TAGGED_OR_LABELED_REAGENT_LANGUAGE")
 if MULTI.search(desc):flags.add("COMPLEX_COFAC_TOPOLOGY_LANGUAGE")
 if not EXPLICIT_BIOCHEM.search(desc):flags.add("NO_EXPLICIT_DIRECT_BINDING_METHOD_PHRASE")
 if regions:
  flags.add("NUMERIC_REGION_IN_METADATA")
  for r in regions:
   if r["nominal_length"]<full_len:
    flags.add("REGION_SHORTER_THAN_CANONICAL_REFERENCE")
   if r["end"]>full_len:
    flags.add("NUMERIC_REGION_END_EXCEEDS_TARGET_REFERENCE_CHECK_CHAIN_ASSOCIATION")
 if DOMAIN.search(desc) and not regions:
  flags.add("DOMAIN_OR_FRAGMENT_WITHOUT_NUMERIC_BOUNDARIES")
 if source_status=="PRIMARY_PATENT_NEEDED":
  flags.add("PRIMARY_PATENT_METHODS_NOT_REVIEWED")
 if source_status=="SOURCE_ARTICLE_ACCESS_PENDING":
  flags.add("PRIMARY_ARTICLE_METHODS_ACCESS_PENDING")
 if source_status=="MISSING_PRIMARY_SOURCE_IDENTIFIER":
  flags.add("PRIMARY_SOURCE_IDENTIFIER_MISSING")
 if source_status=="PRIMARY_DOI_ONLY_METHODS_PENDING":
  flags.add("DOI_ONLY_SOURCE_METHODS_PENDING")
 if known_region:
  flags.add("PRIMARY_SOURCE_FRAGMENT_VS_REFERENCE_FLAG")
  details.append("Primary open-access paper method-section locator exists; individual Kd table/construct still UNVERIFIED")
 # Catch two particularly ambiguous source-description combinations in their
 # metadata, but do not infer that Kd labels are invalid.
 if ("CFTR" in row["target_names"] or "Cystic fibrosis transmembrane conductance regulator" in row["target_names"]) and re.search("half effective concentration",desc,re.I):
  flags.add("KD_FIELD_DESCRIBED_AS_HALF_EFFECTIVE_CONCENTRATION")
 if "Glucagon-like peptide 1 receptor" in row["target_names"] and re.search("glucose",desc,re.I):
  flags.add("POTENTIAL_TARGET_DESCRIPTION_SEMANTIC_MISMATCH")
 if re.search("amyloid[\s-]*beta",desc,re.I) and "Amyloid-beta precursor protein" in row["target_names"]:
  flags.add("POTENTIAL_PROCESSED_PEPTIDE_VS_PRECURSOR_TARGET")
 if re.search(r"\bmouse\b|\brat\b",desc,re.I):
  flags.add("SOURCE_SPECIES_NEEDS_REFERENCE_ALIGNMENT")
 if {"KD_FIELD_DESCRIBED_AS_HALF_EFFECTIVE_CONCENTRATION","POTENTIAL_TARGET_DESCRIPTION_SEMANTIC_MISMATCH","DESCRIPTION_MISSING","PRIMARY_SOURCE_IDENTIFIER_MISSING","DOI_ONLY_SOURCE_METHODS_PENDING"} & flags:
  priority="P0_REVIEW_METADATA_SEMANTICS"
 elif {"REGION_SHORTER_THAN_CANONICAL_REFERENCE","PRIMARY_SOURCE_FRAGMENT_VS_REFERENCE_FLAG",
       "COMPLEX_COFAC_TOPOLOGY_LANGUAGE","SURROGATE_ACTIVITY_OR_ENDPOINT_LANGUAGE"} & flags:
  priority="P1_REVIEW_CONSTRUCT_AND_ENDPOINT"
 else:priority="P2_STANDARD_PRIMARY_SOURCE_REVIEW"
 return priority,sorted(flags),regions,details

def run():
 before,rows=source_rows()
 references=json.loads(REFS.read_text(encoding="utf-8"))
 assert references["task_count"]==30
 refs={r["task_id"]:r for r in references["tasks"]}
 index=json.loads(INDEX.read_text(encoding="utf-8"))
 assert index["status"]=="AUTOMATED_SOURCE_INDEX_ONLY_NOT_BIOLOGICAL_VALIDATION"
 assert index["review_file_sha256"]==sha(before)
 tasks={r["task_id"]:r for r in index["tasks"]}
 assert set(tasks)=={r["task_id"] for r in rows}==set(refs)
 known=json.loads(KNOWN.read_text(encoding="utf-8"))
 assert known["status"]=="PASS_SOURCE_LOCATOR_CONSTRUCT_RISK_ONLY_NOT_APPROVED"
 assert known["canonical_review_sha256"]==sha(before)
 known_flags={r["task_id"]:r for r in known["records"]}
 entries=[]
 for row in sorted(rows,key=lambda r:r["task_id"]):
  task=row["task_id"];ref=refs[task]
  match=ref["sequence_reference_matches"]
  assert len(match)>=1
  full=int(ref["assayed_target_sequence_length"])
  assert full>0 and any(m["status"]=="EXACT_REFERENCE_SEQUENCE" and int(m["reference_length"])==full for m in match)
  priority,flags,regions,details=flags_for(row,full,tasks[task]["evidence_access_status"],known_flags.get(task))
  entries.append({
   "task_id":task,
   "assay_key":row["assay_key"],
   "target_name":row["target_names"],
   "candidate_protein_reference_length":full,
   "canonical_uniprot_accessions":sorted({m["accession"] for m in match if m["status"]=="EXACT_REFERENCE_SEQUENCE"}),
   "original_assay_description_hash":sha(row["assay_description"].encode("utf-8")),
   "source_access_category":tasks[task]["evidence_access_status"],
   "manual_review_priority":priority,
   "metadata_risk_flags":flags,
   "description_numeric_regions":regions,
   "source_paper_fragment_lead":bool(task in known_flags),
   "source_paper_region":known_flags[task]["paper_reported_construct_region"] if task in known_flags else None,
   "reviewer_leads":details,
   "all_assay_validity_attributes_unverified":True,
   "source_construct_sequence_approved":False,
   "experimental_kd_record_approved":False,
   "curation_decision":"UNREVIEWED"
  })
 if QUEUE.read_bytes()!=before:raise PermissionError("Immutable review queue changed")
 assert len(entries)==30
 counts=Counter(e["manual_review_priority"] for e in entries)
 flagcounts=Counter(f for e in entries for f in e["metadata_risk_flags"])
 output={
  "status":"PRELIMINARY_METADATA_RISK_TRIAGE_NOT_SOURCE_ADJUDICATION",
  "created_utc":datetime.now(timezone.utc).isoformat(),
  "canonical_review_csv_sha256":sha(before),
  "reference_screen_file_sha256":sha(REFS.read_bytes()),
  "source_access_index_sha256":sha(INDEX.read_bytes()),
  "two_source_fragment_evidence_sha256":sha(KNOWN.read_bytes()),
  "task_count":len(entries),
  "reviewed_tasks":0,
  "approved_tasks":0,
  "risk_flag_counts":dict(sorted(flagcounts.items())),
  "priority_counts":dict(sorted(counts.items())),
  "metadata_read_only":True,
  "raw_affinity_values_used":False,
  "review_action":"HUMAN_SOURCE_METHODS_AND_RECORD_MATCHING_REQUIRED",
  "entries":entries,
  "limitations":[
   "Flags indicate language in source metadata, not defects proven in scientific source records.",
   "The short-range detector only recognizes explicit numeric residue descriptions; missing flags do not imply a full-length construct.",
   "In multiprotein assays, a residue range may describe a partner chain, not the named database target; never auto-assign it to the target reference.",
   "Reference-sequence length equals UniProt reference, NOT a verified experimental chain.",
   "Assay terms such as competition or fluorescence may coexist with correctly estimated equilibrium Kd; primary paper methods decide endpoint eligibility.",
   "This priority routing is not a source authorization, not an exclusion decision and never a basis for blind model fitting.",
   "Source articles and original individual compound measurement tables must be checked by a qualified reviewer."
  ]}
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/"assay_metadata_risk_triage_30.json").write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf-8")
 with (OUT/"assay_metadata_risk_triage_30.csv").open("w",encoding="utf-8",newline="") as f:
  cols=["task_id","assay_key","target_name","candidate_protein_reference_length",
        "manual_review_priority","source_access_category","metadata_risk_flags",
        "description_numeric_regions","source_paper_fragment_lead","curation_decision"]
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for e in entries:
   w.writerow({k:("|".join(e[k]) if k=="metadata_risk_flags" else
         json.dumps(e[k],separators=(",",":")) if k=="description_numeric_regions" else e[k])
            for k in cols})
 summary=["# Thirty-assay metadata-risk triage (6 November 2026)","",
 "**Source-review planning only. ZERO approved tasks, no measured affinities accessed.**",
 "",
 "This scan uses only the already-fixed human-review CSV, reference length screen, original source-access index and two published source-method region leads. It does **not** verify or reject any BindingDB Kd measurement.",
 "",
 "| Review priority | Tasks |","| --- | ---: |"]
 for k,v in sorted(counts.items()):
  summary.append("| "+k+" | "+str(v)+" |")
 summary+=["","## Metadata warning indicators (overlap permitted)","",
          "| Indicator | Tasks |","| --- | ---: |"]
 for k,v in sorted(flagcounts.items(),key=lambda x:(-x[1],x[0])):
  summary.append("| "+k+" | "+str(v)+" |")
 summary+=["","## Priority source-review queue","",
           "| Candidate task prefix | Target | Reference length | Priority | Key flags |",
           "| --- | --- | ---: | --- | --- |"]
 order={"P0_REVIEW_METADATA_SEMANTICS":0,"P1_REVIEW_CONSTRUCT_AND_ENDPOINT":1,
        "P2_STANDARD_PRIMARY_SOURCE_REVIEW":2}
 for e in sorted(entries,key=lambda x:(order[x["manual_review_priority"]],x["target_name"],x["task_id"])):
  summary.append("| "+e["task_id"][:12]+" | "+e["target_name"].replace("|","/").replace("\n"," ")[:65]+
  " | "+str(e["candidate_protein_reference_length"])+" | "+e["manual_review_priority"]+
  " | "+", ".join(e["metadata_risk_flags"])[:180]+" |")
 summary+=["",
 "### Required interpretation","",
 "A flag never means the assay is invalid or scientifically approved. Competitive and enzymatic assays can report equilibrium Kd in some protocols; the original methods and individual values must decide. Conversely, a description without a flag can still conceal construct differences.",
 "Resolve primary paper/patent, exact physical target/complex chain, sequence boundaries and tags, Kd identity and censoring, assay conditions and per-compound source table.",
 "Keep all 30 human decisions UNREVIEWED until evidence-backed independent adjudication.",
 ""]
 (OUT/"ASSAY_METADATA_TRIAGE.md").write_text("\n".join(summary),encoding="utf-8")
 print(json.dumps({"status":output["status"],"task_count":len(entries),
    "approved_tasks":0,"priority_counts":dict(counts),
    "flag_counts":dict(flagcounts),
    "example_risks":[{"target":e["target_name"],
      "flags":e["metadata_risk_flags"]} for e in entries
       if any(f in e["metadata_risk_flags"] for f in ("KD_FIELD_DESCRIBED_AS_HALF_EFFECTIVE_CONCENTRATION","POTENTIAL_TARGET_DESCRIPTION_SEMANTIC_MISMATCH"))]},indent=2,ensure_ascii=True))
if __name__=="__main__":run()
