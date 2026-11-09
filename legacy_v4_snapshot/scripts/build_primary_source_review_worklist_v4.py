"""Prepare research reviewer worklist from metadata-only source-access audit.

No support labels, query labels, compound identities, raw article paragraphs,
decision/approval creation, or edits to the human-reviewed source CSV.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"data/audit"
ROOTINDEX=ROOT/"research/source_access_v4"
INPUT=ROOTINDEX/"source_access_index.json"
QUEUE=AUDIT/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
OUT=ROOTINDEX/"REVIEW_ACTIONS.md"

def source_hash(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build():
 index=json.loads(INPUT.read_text(encoding="utf-8"))
 if index["status"]!="AUTOMATED_SOURCE_INDEX_ONLY_NOT_BIOLOGICAL_VALIDATION":
  raise RuntimeError("Source-access report not suitable for review worklist")
 if source_hash(QUEUE)!=index["review_file_sha256"]:
  raise PermissionError("Review CSV hash changed; manual review must take priority")
 with QUEUE.open(encoding="utf-8-sig",newline="") as f:
  source={r["task_id"]:r for r in csv.DictReader(f)}
 assert len(source)==30
 assert index["scientifically_approved_tasks"]==0
 pubs={p["pmid"]:p for p in index["publications"]}
 groups={}
 for r in index["tasks"]:
  if r["task_id"] not in source:raise RuntimeError("Task absent in existing review queue")
  if source[r["task_id"]]["curation_decision"]!="UNREVIEWED":
   raise PermissionError("Cannot generate unreviewed checklist for adjudicated source")
  groups.setdefault(r["evidence_access_status"],[]).append(r)
 labels=[
  ("PUBLIC_OPEN_ACCESS_INDEXED","Open-access methods section available for manual inspection"),
  ("SOURCE_ARTICLE_ACCESS_PENDING","Locate the original article or licensed supplementary methods"),
  ("PRIMARY_PATENT_NEEDED","Primary patent protocol requires manual review"),
  ("PRIMARY_DOI_ONLY_METHODS_PENDING","DOI located; original paper methods still need review"),
  ("MISSING_PRIMARY_SOURCE_IDENTIFIER","Source identifier must be resolved before any Kd approval")
 ]
 lines=[
  "# DynaCPI v4 — 30-task source evidence worklist",
  "",
  "**Automated review preparation; ZERO tasks approved. No chemicals or measured affinities included.**",
  "",
  "Source review queue SHA-256: "+index["review_file_sha256"],
  "",
  "A PubMed/UniProt match or a positive keyword signal never establishes exact Kd or experimental construct identity.",
  "",
  "For each task, a qualified reviewer must inspect source methods and supplements, verify the actual measured protein construct, endpoint/units and assay comparability, and record traceable evidence in the original human-review CSV. This worklist is not an approval form.",
  ""
 ]
 for status,heading in labels:
  entries=sorted(groups.get(status,[]),key=lambda r:r["task_id"])
  lines.extend(["## "+heading+" ("+str(len(entries))+")",""])
  for i,r in enumerate(entries,1):
   task=r["task_id"];row=source[task]
   lines.append("### "+str(i)+". Task "+task[:16])
   lines.append("")
   lines.append("- Full task ID: "+task)
   lines.append("- BindingDB internal assay key: "+r["assay_key"])
   lines.append("- Source access category: "+status)
   lines.append("- Source identifiers (for matching only): "+(", ".join(r["pmids"]) if r["pmids"] else "no PubMed identifier"))
   lines.append("- Recorded source DOI(s): "+(", ".join(r.get("doi_identifiers",[])) or "none recorded"))
   lines.append("- Patent metadata source: "+("listed in original source queue" if r["patent_ids"] else "none recorded"))
   flags=row.get("quality_flags","").strip()
   lines.append("- Original metadata quality flags: "+(flags or "none"))
   pids=[pubs[x] for x in r["pmids"]]
   for p in pids:
    lines.append("- Publication PMID "+p["pmid"]+"; metadata: "+p["metadata_access"]+"; OA section indexing: "+p["fulltext_access"])
    if p.get("pmcid"):
     lines.append("  - PMCID: "+p["pmcid"]+" (source access locator; not scientific verification)")
    for sec in p.get("section_headings_and_keyword_counts",[])[:5]:
     if sec["method_heading_hint"] or sec["keyword_hit_counts"]:
      lines.append("  - Inspect source section "+sec["section_id"]+" (heading: "+sec["section_heading"]+")")
   lines.append("- Construct identity from original source: **UNVERIFIED**")
   lines.append("- Exact Kd and qualifier semantics from original source: **UNVERIFIED**")
   lines.append("- Measurement comparability/assay conditions: **UNVERIFIED**")
   lines.append("- Reviewer reference and reasoning: **NOT PROVIDED**")
   lines.append("")
 lines.extend([
  "## Explicit approval restrictions",
  "",
  "- Do not convert section keywords to positive assertions, scientific labels, or an approved cohort.",
  "- Do not pool inhibition/activity endpoints with actual exact Kd.",
  "- Do not assume a full-length UniProt sequence is the physical protein construct.",
  "- Do not use this worklist for model training, split freezing or affinity inference.",
  "- The canonical reviewer file and source-level evidence must be independently adjudicated, then checked by validate_assay_reviews.py.",
  "- The primary scientific-cohort gate remains blocked without source-approved cohort and frozen protein-group roles.",
  ""
 ])
 OUT.write_text("\n".join(lines),encoding="utf-8")
 result={"status":"READ_ONLY_SOURCE_WORKLIST_CREATED_NOT_APPROVED",
         "task_count":sum(map(len,groups.values())),
         "group_counts":dict(Counter({k:len(v) for k,v in groups.items()})),
         "scientific_approvals":0,"canonical_review_file_unchanged":source_hash(QUEUE)==index["review_file_sha256"],
         "worklist_path":str(OUT)}
 (ROOTINDEX/"worklist_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 print(json.dumps(result,indent=2))
if __name__=="__main__":build()
