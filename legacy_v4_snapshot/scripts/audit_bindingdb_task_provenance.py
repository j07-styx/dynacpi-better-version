"""Source-level task consistency diagnostics, NOT biological assay approval."""
from pathlib import Path
from collections import defaultdict, Counter
import json,hashlib
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
candidate={}
for ln in (audit/"bindingdb_full_kd_candidates.jsonl").open(encoding="utf-8"):
    x=json.loads(ln)
    candidate[x["reactant_set_id"]]=x
tasks={}
for ln in (audit/"bindingdb_full_kd_task_summary.jsonl").open(encoding="utf-8"):
    t=json.loads(ln)
    tasks[(t["sequence_sha256"],t["assay_key"])]=t
groups=defaultdict(lambda: defaultdict(set))
bad=[]
for ln in (audit/"bindingdb_full_kd_source_provenance.jsonl").open(encoding="utf-8"):
    d=json.loads(ln)
    rid=d["BindingDB Reactant_set_id"]
    c=candidate[rid]
    seqsha=hashlib.sha256(c["sequence"].encode()).hexdigest()
    group=groups[(seqsha,c["assay_key"])]
    group["row_id"].add(rid)
    group["original_construct_hash"].add(d["protein_construct_sequence_sha256"])
    group["source"].add(d["Curation/DataSource"])
    for key,field in [
      ("target_name","Target Name"),
      ("uniprot_swiss","UniProt (SwissProt) Primary ID of Target Chain 1"),
      ("uniprot_trembl","UniProt (TrEMBL) Primary ID of Target Chain 1"),
      ("article_doi","Article DOI"),("patent","Patent Number"),
      ("ph","pH"),("temp","Temp (C)")]:
        val=d.get(field,"").strip()
        if val:group[key].add(val)
    if d["protein_construct_sequence_sha256"]!=seqsha:
        group["mismatch_id"].add(rid)
        bad.append(rid)
def compare(threshold):
    cnt=Counter()
    examples=[]
    for k,t in tasks.items():
        if t["unique_canonical_compounds"]<threshold:continue
        g=groups[k]
        cnt["tasks"]+=1
        cnt["original_records"]+=t["raw_records"]
        if len(g["source"])!=1:cnt["source_mixed"]+=1
        if len(g["target_name"])!=1:cnt["target_name_mixed"]+=1
        if len(g["original_construct_hash"])!=1:cnt["source_construct_hash_mixed"]+=1
        if "mismatch_id" in g:cnt["tasks_with_source_vs_canonical_sequence_mismatch"]+=1
        if len(g["uniprot_swiss"])+len(g["uniprot_trembl"])==0:cnt["missing_uniprot"]+=1
        if len(g["article_doi"])==0:cnt["without_article_doi"]+=1
        if len(g["patent"])>0:cnt["patent_derived"]+=1
        if not g["ph"]:cnt["no_ph"]+=1
        if not g["temp"]:cnt["no_temp"]+=1
        if len(g["ph"])>1:cnt["multiple_ph_values"]+=1
        if len(g["temp"])>1:cnt["multiple_temp_values"]+=1
        if len(g["article_doi"])>1:cnt["multiple_article_doi_values"]+=1
        if len(g["source"])==1:
            s=next(iter(g["source"]))
            cnt["source_category_"+s]+=1
        if ("mismatch_id" in g or len(g["source"])>1 or len(g["target_name"])>1) and len(examples)<10:
            examples.append({"seqsha":k[0],"assay":k[1],
                             "sources":sorted(g["source"]),"target_names":sorted(g["target_name"])[:5],
                             "mismatch_ids":sorted(g.get("mismatch_id",()))[:5]})
    return {"counts":dict(cnt),"examples":examples}
report={
"status":"PRELIMINARY task metadata QC, not authoritative assay compatibility",
"gt15":compare(15),
"gt30":compare(30),
"unique_source_vs_candidate_hash_mismatches":len(set(bad)),
"meaning":"Unspecified pH or temperature does not prove noncompatibility; same assay ID alone also does not prove comparability."
}
(audit/"bindingdb_full_kd_task_provenance_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
