"""Synthetic checks for DAVIS source-review prioritization."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.triage_davis_source_review_dev import summarize

def main():
    p={"status":"MECHANICAL_EVIDENCE_ONLY_NOT_SOURCE_APPROVAL","source_doi":"doi",
       "records":3,"counts":{"UNRESOLVED_CHEMICAL_MAPPING":2},
       "by_target":{"X":{"EXACT_NUMERIC_MATCH":1,"UNRESOLVED_CHEMICAL_MAPPING":2}}}
    r={"scope":"Registry evidence only; no chemical stereochemistry or source approval",
       "records":2,"registry_status_counts":{"REGISTRY_LINK_WITH_PRIMARY_COLUMN":1},
       "by_target":{"X":{"REGISTRY_LINK_WITH_PRIMARY_COLUMN":1,"NO_UNIQUE_CACHED_REGISTRY_LINK":1}}}
    result=summarize(p,r)
    assert result["approval"] is False
    assert result["tasks"][0]["potential_evidence_count_not_approved"]==2
    assert result["tasks"][0]["additional_records_to_reach_15_exact_matches"]==14
    r["records"]=3
    try:summarize(p,r)
    except ValueError:pass
    else:raise AssertionError("Universe mismatch was accepted")
    print("PASS: status, evidence partition, minimum count, mismatch rejection")
if __name__=="__main__":main()

