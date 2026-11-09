"""Local-only per-reactant primary-source evidence worksheet for DynaCPI v4.

This creates a blank human-review worklist for 921 Reactant_set_ids from the
fixed 30 UNREVIEWED BindingDB tasks, with original source metadata anchors.
It does not include affinity values or SMILES, review primary-source papers,
approve assays, create cohorts/splits, or run model training.

The LOCAL_PRIVATE worksheet is never committed to GitHub. The committed
readiness report contains counts, hashes, and no individual reactant IDs.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
QUEUE_NAME = "bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
GROUP_NAME = "bindingdb_full_kd_compound_median_candidates.jsonl"
SOURCE_NAME = "bindingdb_full_kd_source_provenance.jsonl"
WORKLIST_NAME = "source_record_evidence_worklist_30_LOCAL_PRIVATE.csv"
OUT_NAME = "record_source_evidence_readiness_30.json"

ANCHORS = (
    "task_id", "assay_key", "reactant_set_id", "protein_component_id",
    "protein_sha256", "original_pmid", "original_doi", "original_patent",
    "original_pubchem_aid", "original_source_organism",
    "original_curator_source", "source_row_metadata_sha256"
)
REVIEW_FIELDS = (
    "source_method_locator", "source_table_or_figure_locator",
    "physically_assayed_target", "physical_construct_description",
    "physical_construct_sequence_sha256", "source_assay_method",
    "source_record_endpoint", "source_value_qualifier",
    "original_record_mapping_verified", "direct_equilibrium_kd_verified",
    "physical_construct_verified", "assay_conditions_compatible",
    "reviewer_identity", "review_date", "review_notes", "review_decision"
)
ALL_FIELDS = ANCHORS + REVIEW_FIELDS
FLAGS = (
    "original_record_mapping_verified", "direct_equilibrium_kd_verified",
    "physical_construct_verified", "assay_conditions_compatible"
)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try:yield json.loads(line)
                except json.JSONDecodeError as ex:raise ValueError(f"{path.name}:{n} malformed") from ex

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != list(ALL_FIELDS):
            raise ValueError("Local source-evidence CSV schema mismatch")
        return list(reader)

def metadata_fingerprint(raw):
    # No raw values, source reactant ID, compound structure or affinity included.
    chosen={k:raw.get(k,"") for k in (
        "BindingDB Entry DOI","UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
        "protein_construct_sequence_length",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
        "Target Name","Date of publication")}
    return hashlib.sha256(json.dumps(chosen,sort_keys=True).encode()).hexdigest()

def expected_rows(root):
    audit=Path(root)/"data/audit"
    q=audit/QUEUE_NAME
    if not q.exists():raise FileNotFoundError(q)
    with q.open(encoding="utf-8-sig",newline="") as f:
        review=list(csv.DictReader(f))
    if len(review)!=30 or len({r["task_id"] for r in review})!=30:
        raise ValueError("The original 30-task review queue was changed")
    if any(r["curation_decision"]!="UNREVIEWED" for r in review):
        raise PermissionError("Canonical human review has changed; suspend worksheet build")
    bytask={r["task_id"]:r for r in review}
    seen={}
    for group in jsonl(audit/GROUP_NAME):
        tid=group["task_id"]
        if tid not in bytask:continue
        for rid in group["reactant_ids"]:
            rid=str(rid)
            if not re.fullmatch(r"[0-9]{1,18}",rid) or rid in seen:
                raise ValueError("Duplicate or invalid source reactant identifier")
            seen[rid]=tid
    if not seen or len(seen)!=921:
        raise ValueError("Fixed 921-record provenance scope changed")
    src={}
    for item in jsonl(audit/SOURCE_NAME):
        rid=str(item.get("BindingDB Reactant_set_id",""))
        if rid in seen:
            if rid in src:raise ValueError("Source reactant identity duplicated")
            src[rid]=item
    if set(src)!=set(seen):
        raise ValueError("Source reactant records missing or unexpected")
    expected={}
    for rid,tid in sorted(seen.items(),key=lambda x:int(x[0])):
        q=bytask[tid]
        raw=src[rid]
        if raw.get("protein_construct_sequence_sha256")!=q["protein_sha256"]:
            raise ValueError("Source reference hash changed")
        expected[rid]={
            "task_id":tid,"assay_key":q["assay_key"],"reactant_set_id":rid,
            "protein_component_id":q["protein_component_id"],
            "protein_sha256":q["protein_sha256"],
            "original_pmid":str(raw.get("PMID","")),
            "original_doi":str(raw.get("Article DOI","")),
            "original_patent":str(raw.get("Patent Number","")),
            "original_pubchem_aid":str(raw.get("PubChem AID","")),
            "original_source_organism":str(raw.get("Target Source Organism According to Curator or DataSource","")),
            "original_curator_source":str(raw.get("Curation/DataSource","")),
            "source_row_metadata_sha256":metadata_fingerprint(raw)
        }
    return expected,sha(audit/QUEUE_NAME),sha(audit/GROUP_NAME),sha(audit/SOURCE_NAME)

def create_if_missing(root,expected):
    path=Path(root)/"data/audit"/WORKLIST_NAME
    if path.exists():
        return False,path
    # Create atomically: never overwrite a reviewer-edited file.
    with path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=ALL_FIELDS)
        writer.writeheader()
        for rid,row in expected.items():
            writer.writerow({**row,**{name:("UNREVIEWED" if name=="review_decision" else "") for name in REVIEW_FIELDS}})
    return True,path

def _review_date(value):
    try:day=date.fromisoformat(value)
    except (TypeError,ValueError) as ex:raise ValueError("Reviewer date must be ISO YYYY-MM-DD") from ex
    if day>datetime.now(timezone.utc).date():raise ValueError("Future reviewer date")
    return day

def validate(root):
    expected,reviewsha,groupsha,sourcesha=expected_rows(root)
    created,worklist=create_if_missing(root,expected)
    rows=read_csv(worklist)
    if len(rows)!=len(expected):
        raise ValueError("Source-evidence worklist count mismatch")
    seen=set();counts=Counter();per_task=defaultdict(Counter)
    for item in rows:
        rid=item["reactant_set_id"]
        if rid in seen or rid not in expected:
            raise ValueError("Duplicate/unknown source record in reviewer worklist")
        seen.add(rid)
        if any(item.get(k)!=expected[rid][k] for k in ANCHORS):
            raise PermissionError("Immutable source metadata altered in review worksheet")
        decision=item["review_decision"].strip().upper()
        if decision not in ("UNREVIEWED","APPROVE","REJECT"):
            raise ValueError("Unknown review decision")
        if decision=="UNREVIEWED":
            if any(item.get(f,"").strip().upper()=="YES" for f in FLAGS):
                raise ValueError("Unreviewed row asserts an affirmative evidence verification")
            if item.get("reviewer_identity","").strip() or item.get("review_date","").strip():
                raise ValueError("Unreviewed row must not contain a reviewer signoff")
        else:
            if not item.get("reviewer_identity","").strip():
                raise ValueError("Reviewed row missing reviewer identity")
            _review_date(item.get("review_date",""))
            if len(item.get("review_notes","").strip())<30:
                raise ValueError("Reviewed row requires substantive justification")
            if decision=="APPROVE":
                for flag in FLAGS:
                    if item[flag].strip().upper()!="YES":
                        raise ValueError("Approval missing affirmative verification: "+flag)
                for field in (
                    "source_method_locator","source_table_or_figure_locator",
                    "physically_assayed_target","physical_construct_description",
                    "physical_construct_sequence_sha256","source_assay_method",
                    "source_record_endpoint","source_value_qualifier"):
                    if not item[field].strip():
                        raise ValueError("Approval missing external evidence field: "+field)
                if not re.fullmatch(r"[0-9a-f]{64}",item["physical_construct_sequence_sha256"]):
                    raise ValueError("Physical protein construct sequence hash invalid")
                if item["source_record_endpoint"].strip().upper()!="KD":
                    raise ValueError("Only directly verified experimental Kd eligible for this worklist")
                if item["source_value_qualifier"].strip().upper()!="EXACT":
                    raise ValueError("Censored/qualified endpoint not eligible as exact Kd")
        counts[decision]+=1
        per_task[item["task_id"]][decision]+=1
    if seen!=set(expected):
        raise ValueError("Worklist omits original source records")
    audit=Path(root)/"data/audit"
    if sha(audit/QUEUE_NAME)!=reviewsha or sha(audit/GROUP_NAME)!=groupsha or sha(audit/SOURCE_NAME)!=sourcesha:
        raise PermissionError("Original input changed during evidence readiness audit")
    task_rows=[]
    all_complete=0
    for tid in sorted(per_task):
        cnt=per_task[tid]
        expected_count=sum(x["task_id"]==tid for x in expected.values())
        if sum(cnt.values())!=expected_count:
            raise ValueError("Incomplete task source record coverage")
        full=cnt["APPROVE"]==expected_count
        if full:all_complete+=1
        task_rows.append({
            "task_id":tid,"source_records":expected_count,
            "record_declarations_unreviewed":cnt["UNREVIEWED"],
            "record_declarations_approved":cnt["APPROVE"],
            "record_declarations_rejected":cnt["REJECT"],
            "fully_source_record_declared":full,
            "scientifically_source_approved":False
        })
    total=len(expected)
    status="REVIEW_WORKLIST_UNREVIEWED_NOT_TRAINING_READY" if counts["UNREVIEWED"]==total else "REVIEW_DECLARATIONS_EXIST_REQUIRE_INDEPENDENT_SCIENTIFIC_AUDIT"
    report={
        "status":status,
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "input_review_queue_sha256":reviewsha,
        "input_compound_median_source_sha256":groupsha,
        "input_raw_provenance_sha256":sourcesha,
        "local_private_worklist_sha256":sha(worklist),
        "local_private_worklist_created_this_run":created,
        "source_record_count":total,
        "task_count":len(per_task),
        "self_reported_review_state_counts":dict(counts),
        "fully_declared_task_count":all_complete,
        "source_scientifically_approved_task_count":0,
        "human_primary_source_independently_audited":False,
        "main_scientific_training_authorized":False,
        "contains_individual_affinity_values":False,
        "contains_chemical_structures":False,
        "contains_raw_reactant_identifiers":False,
        "private_local_worklist_not_committed":True,
        "by_task":task_rows,
        "limitations":[
            "This checks source record coverage and completeness of reviewer declarations, not independent source authenticity.",
            "No populated field is proof of direct equilibrium Kd until primary data, actual protein construct and assay conditions are independently verified.",
            "No individual reactant IDs, potency values or SMILES are published by this aggregate readiness report.",
            "The local-only private worksheet must never be staged in GitHub, even after reviewer notes are entered.",
            "Neither this script nor the worksheet can authorize approved cohort generation or scientific model training."
        ]
    }
    out=Path(root)/"research/source_access_v4"
    out.mkdir(parents=True,exist_ok=True)
    (out/OUT_NAME).write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines=[
        "# DynaCPI v4 — local source-evidence worksheet readiness",
        "",
        "**Source-review preparation only; no scientific approval.**",
        "",
        f"Referenced original BindingDB records: **{total}** across **{len(per_task)}** fixed human-review tasks.",
        "",
        "| Current self-reported record state | Records |",
        "| --- | ---: |"
    ]
    for k in ("UNREVIEWED","APPROVE","REJECT"):
        lines.append(f"| {k} | {counts[k]} |")
    lines += [
        "",
        f"Tasks with every source record declared APPROVE (not independently verified): {all_complete}.",
        "**Scientifically approved tasks: 0. Main training permitted: NO.**",
        "",
        "Original review CSV and raw source files were SHA-256 checked. The 921-record private review worksheet exists ONLY on the authorized Windows PC and is not a GitHub artifact.",
        "",
        "Record reviewers must identify original paper/patent table and methods, directly measured experimental endpoint and exact units/qualifiers, physically assayed construct, and compatible assay conditions before declaring any reviewed record. Rejection also requires an evidence-backed rationale.",
        "",
        "The readiness report contains aggregate counts only; it does not include raw reactant IDs, potency values, compounds or chemical structures.",
    ]
    (out/"RECORD_SOURCE_EVIDENCE_READINESS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args()
    report=validate(args.root)
    print(json.dumps({k:report[k] for k in (
        "status","source_record_count","task_count",
        "self_reported_review_state_counts","fully_declared_task_count",
        "source_scientifically_approved_task_count",
        "main_scientific_training_authorized",
        "local_private_worklist_created_this_run"
    )},indent=2))

if __name__=="__main__":
    main()
