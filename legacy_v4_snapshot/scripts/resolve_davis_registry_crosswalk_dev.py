"""Resolve registry links without treating synonyms as structure approval."""
import json,sys,hashlib
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from reconcile_bindingdb_davis_source_dev import norm,compare
def main():
    p=ROOT/"data/audit/davis_primary_20261107"
    old=json.loads((p/"bindingdb_source_cells_LOCAL_PRIVATE.json").read_text())
    unresolved={e["reactant_set_id"]:e for e in old if e["status"]=="UNRESOLVED_CHEMICAL_MAPPING"}
    tables=json.loads((p/"extracted_tables.json").read_text())
    syn=json.loads((p/"pubchem_synonyms.json").read_text())["InformationList"]["Information"]
    by_bdbm=defaultdict(set);cid_columns=defaultdict(set)
    for entry in syn:
        cid=str(entry["CID"]);names={norm(s) for s in entry.get("Synonym",[])}
        for s in entry.get("Synonym",[]):
            if s.startswith("BDBM") and s[4:].isdigit():by_bdbm[s[4:]].add(cid)
        for col,name in enumerate(tables["table4_headers"][3:],3):
            if any(norm(alias) in names for alias in [name]+name.split("/")):cid_columns[cid].add(col)
    source_rows={r[2]:r for r in tables["selected_table4_rows"]}
    evidence=[]
    for line in (ROOT/"data/audit/bindingdb_full_kd_source_provenance.jsonl").open():
        r=json.loads(line);rid=str(r["BindingDB Reactant_set_id"])
        if rid not in unresolved:continue
        e=dict(unresolved[rid]);hits=by_bdbm.get(str(r["BindingDB MonomerID"]),set())
        e["registry_status"]="NO_UNIQUE_CACHED_REGISTRY_LINK"
        if len(hits)==1:
            cid=next(iter(hits));cols=cid_columns.get(cid,set())
            e["pubchem_cid"]=cid
            if len(cols)==1 and cid!="4908365":
                col=next(iter(cols));value=source_rows[e["target"]][col]
                e.update(registry_status="REGISTRY_LINK_WITH_PRIMARY_COLUMN",
                         primary_compound=tables["table4_headers"][col],
                         numeric_comparison=compare(value,r["Kd (nM)"]),
                         basis="Exact BDBM identifier synonym in cached PubChem response; normalized primary column name")
            else:e["registry_status"]="REGISTRY_LINK_WITHOUT_UNIQUE_PRIMARY_COLUMN"
        e["structure_approval"]=False
        evidence.append(e)
    assert len(evidence)==40 and all(e["review_decision"]=="UNREVIEWED" for e in evidence)
    private=p/"bindingdb_registry_crosswalk_LOCAL_PRIVATE.json"
    public=ROOT/"research/davis_registry_crosswalk_20261107.json"
    if private.exists() or public.exists():raise FileExistsError("Preserve evidence")
    report={"scope":"Registry evidence only; no chemical stereochemistry or source approval","records":40,
      "registry_status_counts":dict(Counter(e["registry_status"] for e in evidence)),
      "numeric_comparison_counts":dict(Counter(e["numeric_comparison"] for e in evidence if "numeric_comparison" in e)),
      "by_target":{t:dict(Counter(e["registry_status"] for e in evidence if e["target"]==t)) for t in source_rows if any(e["target"]==t for e in evidence)},
      "synonyms_sha256":hashlib.sha256((p/"pubchem_synonyms.json").read_bytes()).hexdigest(),
      "new_approved_records":0,"limits":["Registry synonyms are supporting evidence, not proof of stereochemical equivalence","No reviewer-owned file or training dataset changed"]}
    with private.open("x") as f:json.dump(evidence,f,indent=2)
    with public.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
