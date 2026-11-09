"""Diagnose source-vs-extraction identity differences, without changing records."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
data=root/"data/audit"
old={}
for line in (data/"bindingdb_full_kd_candidates.jsonl").open(encoding="utf-8"):
    d=json.loads(line);old[d["reactant_set_id"]]=d
mis=[]
for line in (data/"bindingdb_full_kd_source_provenance.jsonl").open(encoding="utf-8"):
    d=json.loads(line)
    rid=d["BindingDB Reactant_set_id"]
    o=old[rid]
    current_sha=d["protein_construct_sequence_sha256"]
    cleaned_sha=hashlib.sha256(o["sequence"].encode()).hexdigest()
    if current_sha!=cleaned_sha:
        mis.append({"reactant_id":rid,"source_raw_seq_len":d["protein_construct_sequence_length"],
                    "candidate_seq_len":len(o["sequence"]),
                    "name":d["Target Name"],"source":d["Curation/DataSource"],
                    "source_smiles":d["Ligand SMILES"][:90],
                    "candidate_smiles":o["smiles_raw"][:90],
                    "source_kd":d["Kd (nM)"],
                    "candidate_pKd":o["pKd"],
                    "raw_protein_id":d["UniProt (SwissProt) Primary ID of Target Chain 1"],
                    "hash_disagrees":True})
result={"count":len(mis),"mismatches":mis,
        "status":"Source-vs-candidate differences requiring source-level QC; no automatic correction"}
(data/"bindingdb_full_kd_provenance_mismatches.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
