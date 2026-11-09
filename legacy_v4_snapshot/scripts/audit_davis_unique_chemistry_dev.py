"""Audit unique chemical identifiers behind the 63 DAVIS source records.

This is a count-only, label-blind provenance audit. Local reactant/monomer
identifiers and private record evidence never leave the workstation.
"""
import json
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/"data/audit/davis_primary_20261107"
SOURCE=ROOT/"data/audit/bindingdb_full_kd_source_provenance.jsonl"
OUT=ROOT/"research/davis_source_unique_chemistry_audit_20261107.json"

def summarize(source_records,registry_records,monomer_by_reactant):
    registry={str(r["reactant_set_id"]):r for r in registry_records}
    exact=defaultdict(set);conditional=defaultdict(set)
    source_ids=set()
    for row in source_records:
        rid=str(row["reactant_set_id"])
        if rid in source_ids:raise ValueError("Duplicate source reactant ID")
        source_ids.add(rid)
        if rid not in monomer_by_reactant:raise ValueError("Source monomer missing")
        target=row["target"];monomer=monomer_by_reactant[rid]
        if row["status"]=="EXACT_NUMERIC_MATCH":
            exact[target].add(monomer);conditional[target].add(monomer)
        elif row["status"]=="UNRESOLVED_CHEMICAL_MAPPING":
            r=registry.get(rid)
            if r is None or r["target"]!=target:raise ValueError("Registry evidence missing/misaligned")
            if r["registry_status"]=="REGISTRY_LINK_WITH_PRIMARY_COLUMN":
                conditional[target].add(monomer)
        else:raise ValueError("Unexpected evidence state")
    if set(registry)!={str(r["reactant_set_id"]) for r in source_records if r["status"]=="UNRESOLVED_CHEMICAL_MAPPING"}:
        raise ValueError("Registry universe mismatch")
    targets=sorted({r["target"] for r in source_records})
    results={}
    for target in targets:
        results[target]={
          "exact_unique_bindingdb_monomer_ids":len(exact[target]),
          "conditional_unique_bindingdb_monomer_ids":len(conditional[target]),
          "duplicate_exact_record_count":sum(r["target"]==target and r["status"]=="EXACT_NUMERIC_MATCH" for r in source_records)-len(exact[target]),
          "conditional_reaches_15_distinct_monomers":len(conditional[target])>=15,
          "still_needed_after_conditional":max(0,15-len(conditional[target]))
        }
    return {"status":"UNIQUE_MONOMER_COUNT_AUDIT_NOT_CHEMICAL_APPROVAL",
            "source_records":len(source_records),"targets":results,
            "notes":["BindingDB MonomerID is a registry grouping, not proof of stereo identity or independent chemistry.",
                     "No source records, labels, compounds or identifiers are exported.",
                     "Conditional registry evidence remains unverified.",
                     "No assay approval or split created."]}

def main():
    if OUT.exists():raise FileExistsError("Preserve existing output")
    source=json.loads((PRIVATE/"bindingdb_source_cells_LOCAL_PRIVATE.json").read_text())
    registry=json.loads((PRIVATE/"bindingdb_registry_crosswalk_LOCAL_PRIVATE.json").read_text())
    wanted={str(r["reactant_set_id"]) for r in source}
    ids={}
    with SOURCE.open(encoding="utf-8") as f:
        for line in f:
            row=json.loads(line)
            rid=str(row["BindingDB Reactant_set_id"])
            if rid in wanted:
                if rid in ids:raise ValueError("Duplicate provenance record")
                ids[rid]=str(row["BindingDB MonomerID"])
    result=summarize(source,registry,ids)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

