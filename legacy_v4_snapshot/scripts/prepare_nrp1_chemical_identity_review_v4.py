"""Prepare a private chemical-identity review packet for six NRP1 Table 3 candidates.
Includes BindingDB ligand structures ONLY in LOCAL_PRIVATE file, never public report.
"""
import csv,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];audit=root/"data/audit"
source=json.loads((audit/"nrp1_table3_numeric_candidates_LOCAL_PRIVATE.json").read_text(encoding="utf-8"))
assert len(source["paper_values"])==6
lookup={}
for p in source["paper_values"]:
 assert len(p["exact_numeric_source_record_matches"])==1
 rid=p["exact_numeric_source_record_matches"][0]
 assert rid not in lookup
 lookup[rid]=p
records={}
with (audit/"bindingdb_full_kd_source_provenance.jsonl").open(encoding="utf-8") as f:
 for line in f:
  r=json.loads(line);rid=str(r.get("BindingDB Reactant_set_id",""))
  if rid in lookup:
   assert rid not in records
   records[rid]=r
assert set(records)==set(lookup)
packet=[]
for rid,p in lookup.items():
 r=records[rid]
 assert r["PMID"]=="29648813" and r["Article DOI"].lower()=="10.1021/acs.jmedchem.8b00210"
 assert float(r["Kd (nM)"])==p["paper_spr_kd_nm"]
 assert r["Ligand SMILES"].strip() and r["BindingDB MonomerID"].strip()
 packet.append({"paper_compound_id":p["paper_compound_id"],"paper_table":"Table 3","paper_substituent":p["paper_substituent"],"paper_spr_kd_nm":p["paper_spr_kd_nm"],"bindingdb_reactant_set_id":rid,"bindingdb_monomer_id":r["BindingDB MonomerID"],"bindingdb_smiles":r["Ligand SMILES"],"bindingdb_kd_nm":r["Kd (nM)"],"paper_structure_checked":False,"chemical_identity_verified":False,"assay_construct_verified":False,"review_decision":"UNREVIEWED","manual_next_step":"Compare drawn paper compound and substituent position to BindingDB SMILES; confirm stereochemistry and salts."})
packet.sort(key=lambda x:x["paper_compound_id"])
path=audit/"nrp1_table3_chemical_identity_review_LOCAL_PRIVATE.json"
path.write_text(json.dumps({"status":"CHEMICAL_IDENTITY_REVIEW_REQUIRED","records":packet},indent=2,ensure_ascii=False),encoding="utf-8")
public={"status":"SIX_CHEMICAL_IDENTITY_REVIEW_ITEMS_PREPARED","records":len(packet),"with_bindingdb_smiles":sum(bool(x["bindingdb_smiles"]) for x in packet),"chemical_identity_verified":0,"approved":0,"private_packet_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"private_packet_path":str(path)}
(root/"research/source_access_v4/NRP1_CHEMICAL_IDENTITY_REVIEW_STATUS.json").write_text(json.dumps(public,indent=2),encoding="utf-8")
print(json.dumps(public,indent=2))
