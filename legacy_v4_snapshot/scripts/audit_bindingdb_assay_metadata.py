"""Audit how often exact-Kd candidate support tasks have interpretable assay text.

Assay descriptions are untrusted raw experimental metadata, not a guarantee of
endpoint identity or comparable experimental conditions. Outputs are descriptive.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, io, json, re, zipfile

root=Path(__file__).resolve().parents[1]
data=root/"data/audit"
candidate_file=data/"bindingdb_full_kd_candidates.jsonl"
zip_path=root/"data/bindingdb_202611/BindingDB_Assays_202611_tsv.zip"
stats=Counter()
groups=defaultdict(lambda: {"rows":0,"dois":set()})
for ln in candidate_file.open(encoding="utf-8"):
    r=json.loads(ln)
    k=(hashlib.sha256(r["sequence"].encode()).hexdigest(),r["assay_key"])
    g=groups[k]
    g["rows"]+=1
    if r["article_doi"].strip():
        g["dois"].add(r["article_doi"].strip())
    stats["candidate_records"]+=1
need_keys=set(k[1] for k in groups)
assays={}
with zipfile.ZipFile(zip_path) as z:
    member=next(m for m in z.namelist() if m.endswith(".tsv"))
    f=io.TextIOWrapper(z.open(member),encoding="utf-8-sig",errors="replace",newline="")
    for r in csv.DictReader(f,delimiter="\t"):
        stats["assay_metadata_rows"]+=1
        key=(r.get("ENTRYID") or "").strip()+"_"+(r.get("ASSAYID") or "").strip()
        if key not in need_keys:
            continue
        name=(r.get("ASSAY_NAME") or "").strip()
        desc=(r.get("DESCRIPTION") or "").strip()
        if key in assays and assays[key] != (name,desc):
            stats["conflicting_assay_description_rows"]+=1
        assays.setdefault(key,(name,desc))
    f.close()

terms={"kd":re.compile(r"(?i)(?:\bkd\b|\bdissociation constant\b|\bbinding affinity\b)"),
       "kinetics":re.compile(r"(?i)(?:\bkon\b|\bkoff\b|\bkinetics\b)"),
       "temperature":re.compile(r"(?i)(?:temperature|degrees|\btemp\b|[°]c)"),
       "ph":re.compile(r"(?i)(?:\bpH\b|buffer)"),
       "biochemical":re.compile(r"(?i)(?:biochemical|purified|recombinant|SPR|ITC|surface plasmon|calorimetric|radioligand)"),
       "cellular":re.compile(r"(?i)(?:cellular|cell.line|in vivo|cell.based|viability)"),
       "ic50":re.compile(r"(?i)\bIC50\b"),
       "ki":re.compile(r"(?i)\bKi\b")}
out_file=data/"bindingdb_full_kd_assay_contexts.jsonl"
with out_file.open("w",encoding="utf-8") as out:
    for (sha,key),g in sorted(groups.items()):
        name,desc=assays.get(key,("",""))
        joined=name+" "+desc
        signals={term:bool(regex.search(joined)) for term,regex in terms.items()}
        eligible="ge30" if g["rows"]>=30 else ("ge15_raw_rows" if g["rows"]>=15 else "other")
        stats["task_total"]+=1
        if key in assays:
            stats["assay_match_tasks"]+=1
            stats["assay_match_records"]+=g["rows"]
        if name:
            stats["assay_named_tasks"]+=1
        if desc:
            stats["assay_described_tasks"]+=1
        if not name and not desc:
            stats["assay_no_text_tasks"]+=1
        for term,found in signals.items():
            if found:
                stats["signal_"+term+"_tasks"]+=1
        if len(g["dois"])>1:
            stats["multiple_article_doi_tasks"]+=1
        out.write(json.dumps({"sequence_sha256":sha, "assay_key":key, "raw_record_count":g["rows"],
          "article_doi_count":len(g["dois"]), "name":name,"description":desc,
          "signals":signals, "preliminary_size_band":eligible})+"\n")
report={
"source":"Full November 2026 BindingDB Kd candidates and BindingDB_Assays_202611_tsv.zip",
"status":"PRELIMINARY; text pattern matches do not prove assay compatibility",
"counts":dict(stats),
"assay_keys_found":len(assays),
"assay_keys_sought":len(need_keys),
"warning":"Assay identity from ENTRYID_ASSAYID is a metadata join, not sufficient to establish comparable experimental measurements. Missing descriptions and mixed assay modalities require scientific review."
}
data.joinpath("bindingdb_full_kd_assay_context_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
