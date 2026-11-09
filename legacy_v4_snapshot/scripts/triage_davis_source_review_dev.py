"""Aggregate source-evidence review triage; never changes assay approvals.

Joins mutually exclusive source reconciliation and unresolved-only registry reports.
Registry links remain unverified, especially stereochemistry and constructs.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"research/bindingdb_davis_source_cells_20261107.json"
B=ROOT/"research/davis_registry_crosswalk_20261107.json"
DEST=ROOT/"research/davis_source_review_priority_20261107.json"

def summarize(primary,registry):
    if primary["status"]!="MECHANICAL_EVIDENCE_ONLY_NOT_SOURCE_APPROVAL":
        raise ValueError("Unexpected source evidence status")
    if registry["scope"]!="Registry evidence only; no chemical stereochemistry or source approval":
        raise ValueError("Registry evidence is not unreviewed-only")
    if registry["records"]!=primary["counts"]["UNRESOLVED_CHEMICAL_MAPPING"]:
        raise ValueError("Registry review universe must equal unresolved source universe")
    result=[]
    for target,pc in sorted(primary["by_target"].items()):
        rc=registry["by_target"].get(target)
        if rc is None:raise ValueError("Registry target missing")
        exact=pc.get("EXACT_NUMERIC_MATCH",0)
        unresolved=pc.get("UNRESOLVED_CHEMICAL_MAPPING",0)
        linked=rc.get("REGISTRY_LINK_WITH_PRIMARY_COLUMN",0)
        still=rc.get("NO_UNIQUE_CACHED_REGISTRY_LINK",0)
        if unresolved!=linked+still:raise ValueError("Inconsistent unresolved partition")
        result.append({
          "target":target,"records":exact+unresolved,
          "exact_structure_numeric_matches":exact,
          "registry_links_not_stereo_verified":linked,
          "no_unique_registry_link":still,
          "potential_evidence_count_not_approved":exact+linked,
          "additional_records_to_reach_15_exact_matches":max(0,15-exact),
          "conditional_15_possible_if_all_registry_links_verified":exact+linked>=15,
          "still_missing_after_conditional_registry_links":max(0,15-exact-linked),
          "source_review_status":"UNREVIEWED"
        })
    if sum(x["records"] for x in result)!=primary["records"]:
        raise ValueError("Source record count mismatch")
    if sum(x["registry_links_not_stereo_verified"] for x in result)!=registry["registry_status_counts"]["REGISTRY_LINK_WITH_PRIMARY_COLUMN"]:
        raise ValueError("Registry count mismatch")
    return {
      "status":"EVIDENCE_TRIAGE_ONLY_ZERO_APPROVED",
      "primary_source_doi":primary["source_doi"],
      "source_records":primary["records"],
      "exact_structure_numeric_matches":sum(x["exact_structure_numeric_matches"] for x in result),
      "conditional_registry_links_not_stereo_verified":sum(x["registry_links_not_stereo_verified"] for x in result),
      "unresolved_without_unique_registry_link":sum(x["no_unique_registry_link"] for x in result),
      "tasks":result,
      "review_order":[x["target"] for x in sorted(result,key=lambda x:(-int(x["conditional_15_possible_if_all_registry_links_verified"]),x["still_missing_after_conditional_registry_links"],-x["exact_structure_numeric_matches"]))],
      "approval":False,
      "limitations":[
        "Registry matches do not prove stereochemical equivalence or compound identity.",
        "Numeric agreement does not verify original assay method or experimental protein construct.",
        "None of the 30 BindingDB source-review tasks is approved.",
        "The 15-compound minimum is only a task eligibility threshold, not scientific approval.",
        "Do not use this triage as training data or a frozen test split."
      ]
    }

def main():
    if DEST.exists():raise FileExistsError("Refuse to overwrite existing triage")
    result=summarize(json.loads(A.read_text()),json.loads(B.read_text()))
    DEST.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

