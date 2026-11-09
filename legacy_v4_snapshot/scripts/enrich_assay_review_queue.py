"""Prepare source-identifiable, label-blind manual BindingDB curation sheet.

No pKd/Kd values are copied. Human source review is required; descriptive
text is not sufficient to prove scientific assay compatibility.
"""
from collections import defaultdict
from pathlib import Path
import csv
import json
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
source=audit/"bindingdb_full_kd_source_provenance.jsonl"
candidates=audit/"bindingdb_full_kd_candidates.jsonl"
queue=audit/"bindingdb_assay_manual_review_queue_30.csv"
dest=audit/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
with queue.open(encoding="utf-8-sig",newline="") as f:
    wanted=list(csv.DictReader(f))
assert len(wanted)==30
wanted_keys={(x["protein_sha256"],x["assay_key"]) for x in wanted}
reactant_to_key={}
for line in candidates.open(encoding="utf-8"):
    row=json.loads(line)
    # sha256 is recomputed rather than trusting other JSONL identities
    import hashlib
    key=(hashlib.sha256(row["sequence"].encode()).hexdigest(),row["assay_key"])
    if key in wanted_keys:
        reactant_to_key[row["reactant_set_id"]]=key
groups=defaultdict(lambda:defaultdict(set))
for line in source.open(encoding="utf-8"):
    obj=json.loads(line)
    key=reactant_to_key.get(obj["BindingDB Reactant_set_id"])
    if key is None:continue
    g=groups[key]
    g["ids"].add(obj["BindingDB Reactant_set_id"])
    for key2,column in [
        ("source","Curation/DataSource"),("target_names","Target Name"),
        ("swissprot","UniProt (SwissProt) Primary ID of Target Chain 1"),
        ("trembl","UniProt (TrEMBL) Primary ID of Target Chain 1"),
        ("article_doi","Article DOI"),("pmid","PMID"),
        ("patents","Patent Number"),("ph","pH"),("temperature","Temp (C)"),
        ("target_pdb","PDB ID(s) of Target Chain 1"),
        ("ligand_complex_pdb","PDB ID(s) for Ligand-Target Complex")
    ]:
        s=(obj.get(column) or "").strip()
        if s:g[key2].add(s)

def summarize(g,k,max_items=8):
    values=sorted(g.get(k,set()))
    return " | ".join(values[:max_items])+(" | ...MORE" if len(values)>max_items else "")

columns=list(wanted[0])
addon=["record_count","reactant_ids_sample","data_sources","target_names",
       "swissprot","trembl","article_dois","pmids","patent_numbers","reported_ph",
       "reported_temperature","target_pdbs","ligand_complex_pdbs",
       "reviewer_identity","evidence_reference","review_date",
       "curation_decision","curation_reason"]
for column in addon:
    assert column not in columns,column
columns+=addon
if dest.exists():
    raise FileExistsError("Refusing to overwrite a possibly adjudicated manual review file: " + str(dest))
with dest.open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=columns)
    writer.writeheader()
    for rec in wanted:
        key=(rec["protein_sha256"],rec["assay_key"])
        g=groups.get(key)
        if g is None or not g["ids"]:
            raise ValueError("Review task missing source records: "+str(key))
        rec.update({
         "record_count":len(g["ids"]),
         "reactant_ids_sample":summarize(g,"ids",12),
         "data_sources":summarize(g,"source"),
         "target_names":summarize(g,"target_names"),
         "swissprot":summarize(g,"swissprot"),
         "trembl":summarize(g,"trembl"),
         "article_dois":summarize(g,"article_doi"),
         "pmids":summarize(g,"pmid"),
         "patent_numbers":summarize(g,"patents"),
         "reported_ph":summarize(g,"ph"),
         "reported_temperature":summarize(g,"temperature"),
         "target_pdbs":summarize(g,"target_pdb"),
         "ligand_complex_pdbs":summarize(g,"ligand_complex_pdb"),
         "reviewer_identity":"","evidence_reference":"","review_date":"",
         "curation_decision":"UNREVIEWED","curation_reason":""})
        writer.writerow(rec)
report={
"status":"SOURCE_REFERENCES_JOINED_REVIEW_REQUIRED",
"review_tasks":len(wanted),
"review_components":len(set(rec["protein_component_id"] for rec in wanted)),
"no_affinity_values_exported":True,
"output":str(dest),
"required_manual_decision_fields":["construct_verified","assay_semantics_verified","records_compatible",
                                    "reviewer_identity","evidence_reference","review_date","curation_decision","curation_reason"],
"warning":"No tasks have been approved; paper/source verification is essential."}
(audit/"bindingdb_assay_manual_review_queue_30_ENRICHED_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
