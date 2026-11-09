"""Fail-closed approval interface for human-adjudicated BindingDB task metadata.

This gate checks completeness, consistency and reviewer declarations only.
It cannot independently determine whether a published assay measured true Kd.
Do NOT run model training from an UNREVIEWED / REJECT result.
"""
from pathlib import Path
from collections import Counter
from datetime import date
import argparse
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"data/audit"
BASE=AUDIT/"bindingdb_assay_manual_review_queue_30.csv"
ENRICHED=AUDIT/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
REVIEW_FIELDS={"construct_verified","assay_semantics_verified","records_compatible",
               "adjudicator_notes","reviewer_identity","evidence_reference",
               "review_date","curation_decision","curation_reason"}

def read_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def evaluate_reviews(original,reviewed):
    base={x["task_id"]:x for x in original}
    seen=set()
    reviewed_approved=[]
    counts=Counter()
    for row in reviewed:
        task=row["task_id"]
        if task in seen or task not in base:
            raise ValueError("Unrecognized or duplicated task ID")
        seen.add(task)
        for key,value in base[task].items():
            if key not in REVIEW_FIELDS and value != row.get(key):
                raise ValueError("Nonreview source metadata changed: "+task+" field "+key)
        decision=row.get("curation_decision","").upper().strip()
        if decision not in {"APPROVE","REJECT","UNREVIEWED"}:
            raise ValueError("Unknown curation_decision: "+decision)
        counts[decision]+=1
        if decision=="UNREVIEWED":
            if any(row.get(s,"").upper().strip() in {"YES","APPROVE"} for s in
                   ("construct_verified","assay_semantics_verified","records_compatible")):
                raise ValueError("Unreviewed task contains affirmative verification fields")
            continue
        if not row.get("reviewer_identity","").strip() or not row.get("curation_reason","").strip():
            raise ValueError("Reviewed task missing reviewer or justification")
        day=row.get("review_date","")
        try:
            reviewed_day=date.fromisoformat(day)
        except ValueError:
            raise ValueError("Reviewed task must have ISO calendar review_date")
        if reviewed_day > date.today():
            raise ValueError("Future review date")
        if decision=="APPROVE":
            for s in ("construct_verified","assay_semantics_verified","records_compatible"):
                if row.get(s,"").upper().strip() != "YES":
                    raise ValueError("Approval requires YES in "+s)
            if not row.get("evidence_reference","").strip():
                raise ValueError("Approval missing evidence reference")
            if len(row["curation_reason"].strip())<15:
                raise ValueError("Approval justification too brief")
            reviewed_approved.append({"task_id":task,"protein_component_id":row["protein_component_id"],
                                       "assay_key":row["assay_key"],"reviewer_identity":row["reviewer_identity"],
                                       "evidence_reference":row["evidence_reference"]})
    if seen != set(base):
        raise ValueError("Reviewed file omits tasks from original queue")
    return {"counts":dict(counts),
            "approved_tasks":len(reviewed_approved),
            "approved_protein_components":len(set(x["protein_component_id"] for x in reviewed_approved)),
            "approved":reviewed_approved}

def main(path=ENRICHED):
    base=read_rows(BASE)
    reviews=read_rows(path)
    result=evaluate_reviews(base,reviews)
    result["status"]="REVIEW_INCOMPLETE_NOT_TRAINING_READY" if result["counts"].get("UNREVIEWED",0) else "ALL_REVIEWED_REQUIRES_ADDITIONAL_COHORT_QC"
    result["source_file"]=str(path)
    result["source_sha256"]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result["scientific_cohort_approved"]=False
    result["reason"]="Review declarations alone cannot certify construct/assay semantics, and a cohort-wide protein-group split is not frozen."
    out=AUDIT/"bindingdb_assay_manual_review_status.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="approved"},indent=2),flush=True)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--review-file",type=Path,default=ENRICHED)
    args=parser.parse_args()
    main(args.review_file)
