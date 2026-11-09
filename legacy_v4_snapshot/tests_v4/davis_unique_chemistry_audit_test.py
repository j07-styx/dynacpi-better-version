"""Synthetic unique-monomer audit checks, no private data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_davis_unique_chemistry_dev import summarize

def main():
    s=[
      {"reactant_set_id":"1","target":"X","status":"EXACT_NUMERIC_MATCH"},
      {"reactant_set_id":"2","target":"X","status":"EXACT_NUMERIC_MATCH"},
      {"reactant_set_id":"3","target":"X","status":"UNRESOLVED_CHEMICAL_MAPPING"}
    ]
    reg=[{"reactant_set_id":"3","target":"X","registry_status":"REGISTRY_LINK_WITH_PRIMARY_COLUMN"}]
    result=summarize(s,reg,{"1":"a","2":"a","3":"b"})
    assert result["targets"]["X"]["exact_unique_bindingdb_monomer_ids"]==1
    assert result["targets"]["X"]["duplicate_exact_record_count"]==1
    assert result["targets"]["X"]["conditional_unique_bindingdb_monomer_ids"]==2
    assert result["status"]=="UNIQUE_MONOMER_COUNT_AUDIT_NOT_CHEMICAL_APPROVAL"
    try:summarize(s,reg,{"1":"a","2":"a"})
    except ValueError:pass
    else:raise AssertionError("Missing monomer mapping accepted")
    try:summarize(s,reg+[{"reactant_set_id":"4","target":"X"}],{"1":"a","2":"a","3":"b"})
    except ValueError:pass
    else:raise AssertionError("Extra registry evidence accepted")
    print("PASS: deduplication, conditional grouping, missing provenance, registry mismatch")
if __name__=="__main__":main()

