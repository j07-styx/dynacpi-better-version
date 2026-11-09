"""Preserve original experimental metadata for the exact-looking-Kd candidates.

Reads the publisher's full Nov 2026 TSV and extracts only records already selected
by source-level screening. No biological QA or final cohort acceptance is implied.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,io,json,zipfile
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
source=audit/"bindingdb_full_kd_candidates.jsonl"
ids={}
for line in source.open(encoding="utf-8"):
    rec=json.loads(line)
    rid=rec["reactant_set_id"]
    if rid in ids:
        raise ValueError("Duplicate candidate ID in materialized input: "+rid)
    ids[rid]=rec
name="BindingDB_All.tsv"
archive=root/"data/bindingdb_202611/BindingDB_All_202611_tsv.zip"
fields=[
"BindingDB Reactant_set_id","Ligand SMILES","BindingDB MonomerID",
"Target Name","Target Source Organism According to Curator or DataSource",
"Number of Protein Chains in Target (>1 implies a multichain complex)",
"BindingDB Target Chain Sequence 1",
"UniProt (SwissProt) Primary ID of Target Chain 1",
"UniProt (TrEMBL) Primary ID of Target Chain 1",
"PDB ID(s) of Target Chain 1",
"PDB ID(s) for Ligand-Target Complex",
"Kd (nM)","Ki (nM)","IC50 (nM)","EC50 (nM)",
"pH","Temp (C)","Curation/DataSource","Article DOI",
"BindingDB Entry DOI","PMID","PubChem AID","Patent Number",
"Date of publication","Date in BindingDB"
]
renames={s:s for s in fields}
renames["BindingDB Target Chain Sequence 1"]="protein_construct_sequence_sha256"
out=audit/"bindingdb_full_kd_source_provenance.jsonl"
stats=Counter()
sources=Counter()
ids_found=set()
with zipfile.ZipFile(archive) as z:
    with io.TextIOWrapper(z.open(name),encoding="utf-8-sig",errors="replace",newline="") as inf:
        reader=csv.reader(inf,delimiter="\t")
        header=next(reader)
        col={s:header.index(s) for s in fields}
        with out.open("w",encoding="utf-8") as of:
            for row in reader:
                stats["source_rows"]+=1
                if len(row)!=len(header):
                    stats["malformed_column_count_rows"]+=1
                    continue
                rid=row[col["BindingDB Reactant_set_id"]]
                if rid not in ids:
                    continue
                if rid in ids_found:
                    stats["duplicate_candidate_reactant_ids"]+=1
                    continue
                ids_found.add(rid)
                d={renames[s]:row[i].strip() for s,i in col.items()}
                seq=d["protein_construct_sequence_sha256"]
                d["protein_construct_sequence_length"]=len(seq)
                d["protein_construct_sequence_sha256"]=hashlib.sha256(seq.encode()).hexdigest()
                if d["protein_construct_sequence_sha256"] != hashlib.sha256(ids[rid]["sequence"].encode()).hexdigest():
                    stats["sequence_hash_mismatch"]+=1
                if d["Kd (nM)"] != str(d["Kd (nM)"]).strip():
                    stats["kd_strip_anomaly"]+=1
                if not d["UniProt (SwissProt) Primary ID of Target Chain 1"] and not d["UniProt (TrEMBL) Primary ID of Target Chain 1"]:
                    stats["missing_uniprot_id"]+=1
                if not d["pH"]:
                    stats["missing_ph"]+=1
                if not d["Temp (C)"]:
                    stats["missing_temp"]+=1
                if not d["PDB ID(s) of Target Chain 1"]:
                    stats["missing_target_structure_id"]+=1
                if not d["Article DOI"]:
                    stats["missing_article_doi"]+=1
                sources[d["Curation/DataSource"]]+=1
                of.write(json.dumps(d)+"\n")
                stats["matched_candidate_rows"]+=1
                if stats["source_rows"]%500000==0:
                    print(json.dumps({"scanned":stats["source_rows"],"selected":stats["matched_candidate_rows"]}),flush=True)
stats["candidate_ids_missing_from_export"]=len(ids)-len(ids_found)
report={
  "scope":"Provenance fields for exact-looking-Kd candidate reactant IDs, full November 2026 BindingDB export",
  "status":"SOURCE PROVENANCE EXTRACTED; scientifically uncurated",
  "counts":dict(stats),
  "top_source_categories":sources.most_common(20),
  "raw_source_file_sha256":"470a3c2e4aebe3c493de3e9e46e739592eb8e29b3adafb5189c875442268a0b2",
  "fields_preserved":fields,
  "note":"Target sequence is represented by exact hash + length; full source sequence is retained in original ZIP and candidate JSONL. Physical constructs can differ despite sequence identity."
}
(audit/"bindingdb_full_kd_provenance_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="fields_preserved"},indent=2),flush=True)
