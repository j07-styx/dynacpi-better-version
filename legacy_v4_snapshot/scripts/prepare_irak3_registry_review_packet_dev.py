"""Prepare private IRAK3 source-review packet from existing provenance.

Never auto-approve registry synonyms, stereo identity, assay methods or constructs.
Private per-record evidence stays in data/audit and is never committed.
"""
import csv,json,hashlib
from pathlib import Path
from collections import Counter
from decimal import Decimal,InvalidOperation
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/"data/audit/davis_primary_20261107"
SRC=ROOT/"data/audit/bindingdb_full_kd_source_provenance.jsonl"
PRIVATE=P/"irak3_registry_review_packet_LOCAL_PRIVATE.csv"
PUBLIC=ROOT/"research/irak3_registry_review_packet_summary_dev.json"
COLUMNS=("reactant_set_id","bindingdb_monomer_id","primary_compound","pubchem_cid",
         "primary_table_cell_kd_nm","bindingdb_kd_nm","numeric_match",
         "compound_identity_verified","stereo_verified","protein_construct_verified",
         "assay_method_verified","review_decision","reviewer_evidence_notes")
def make(leads,provenance,tables):
    rows={r[2]:r for r in tables["selected_table4_rows"]}
    headers=tables["table4_headers"]
    selected=[]
    for e in leads:
        if e["target"]!="IRAK3" or e["registry_status"]!="REGISTRY_LINK_WITH_PRIMARY_COLUMN":continue
        if e["review_decision"]!="UNREVIEWED" or e["structure_approval"] is not False:raise ValueError("Prior approval unexpected")
        rid=str(e["reactant_set_id"])
        record=provenance.get(rid)
        if record is None:raise ValueError("Missing provenance")
        col=[i for i,v in enumerate(headers) if v==e["primary_compound"]]
        if len(col)!=1:raise ValueError("Ambiguous primary column")
        raw=rows["IRAK3"][col[0]]
        try:
            source=Decimal(str(raw));binding=Decimal(str(record["Kd (nM)"]).strip())
        except InvalidOperation as exc:raise ValueError("Non-numeric Kd") from exc
        if not source.is_finite() or source<=0 or binding!=source:raise ValueError("Primary Kd mismatch")
        if e["numeric_comparison"]!="EXACT_NUMERIC_MATCH":raise ValueError("Unexpected numeric evidence")
        selected.append({
            "reactant_set_id":rid,
            "bindingdb_monomer_id":str(record["BindingDB MonomerID"]),
            "primary_compound":e["primary_compound"],
            "pubchem_cid":str(e["pubchem_cid"]),
            "primary_table_cell_kd_nm":str(source),
            "bindingdb_kd_nm":str(binding),
            "numeric_match":"EXACT",
            "compound_identity_verified":"NO",
            "stereo_verified":"NO",
            "protein_construct_verified":"NO",
            "assay_method_verified":"NO",
            "review_decision":"UNREVIEWED",
            "reviewer_evidence_notes":""
        })
    if len(selected)!=7:raise ValueError("Expected exactly seven leads")
    if len({r["reactant_set_id"] for r in selected})!=7 or len({r["bindingdb_monomer_id"] for r in selected})!=7:
        raise ValueError("Duplicate evidence records")
    return sorted(selected,key=lambda r:r["primary_compound"])
def main():
    if PRIVATE.exists() or PUBLIC.exists():raise FileExistsError("Refusing to overwrite review packet")
    leads=json.loads((P/"bindingdb_registry_crosswalk_LOCAL_PRIVATE.json").read_text())
    tables=json.loads((P/"extracted_tables.json").read_text())
    wanted={str(e["reactant_set_id"]) for e in leads if e["target"]=="IRAK3" and e["registry_status"]=="REGISTRY_LINK_WITH_PRIMARY_COLUMN"}
    provenance={}
    with SRC.open(encoding="utf-8") as f:
        for line in f:
            r=json.loads(line);rid=str(r["BindingDB Reactant_set_id"])
            if rid in wanted:
                if rid in provenance:raise ValueError("Duplicate provenance")
                provenance[rid]=r
    packet=make(leads,provenance,tables)
    with PRIVATE.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(packet)
    summary={
      "status":"PRIVATE_REVIEW_PACKET_PREPARED_ZERO_APPROVAL",
      "target":"IRAK3","review_leads":len(packet),
      "unique_bindingdb_monomer_ids":len({r["bindingdb_monomer_id"] for r in packet}),
      "primary_source_numeric_matches":sum(r["numeric_match"]=="EXACT" for r in packet),
      "compound_identity_approved":0,"stereochemistry_approved":0,
      "protein_construct_approved":0,"assay_method_approved":0,
      "local_packet_relative_path":"data/audit/davis_primary_20261107/irak3_registry_review_packet_LOCAL_PRIVATE.csv",
      "source_table_sha256":hashlib.sha256((P/"extracted_tables.json").read_bytes()).hexdigest(),
      "limits":["Private packet has identifiers and source Kd cells; it must not be committed.",
                "Numerical equality and PubChem synonyms are not chemical identity or assay approval.",
                "Human reviewer must verify structures, stereo, protein constructs and assay methods."]
    }
    PUBLIC.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()

