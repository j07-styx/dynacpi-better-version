"""Sequence-exact UniProt mapping of engineering-only DAVIS pilot proteins.

Pilot protein sequences may represent UniProt contiguous domains or truncations,
not full-length proteins. Only accept sequence-exact contiguous substrings for
automated coordinate mapping. This does NOT verify an assayed construct.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import urllib.parse
import urllib.request

root=Path(__file__).resolve().parents[1]
ref=json.loads((root/"data/references/proteins.txt").read_text(encoding="utf-8"))
pilot=json.loads((root/"data/engineering_pilot/rows.json").read_text(encoding="utf-8"))
stage=json.loads((root/"data/staging/sequences.json").read_text(encoding="utf-8"))
aliases={
    "ARK5":"NUAK1","MRCKA":"CDC42BPA","RSK2(KinDom.1-N-terminal)":"RPS6KA3",
    "FAK":"PTK2","MEK6":"MAP2K6","MEK5":"MAP2K5","OSR1":"OXSR1",
    "ADCK4":"COQ8B","ICK":"CILK1","MLK2":"MAP3K10","MRCKB":"CDC42BPB",
}
by_hash=defaultdict(list)
for name,seq in ref.items():by_hash[hashlib.sha256(seq.encode()).hexdigest()].append(name)
splits={}
for row in pilot:
    h=row["sequence"]
    if h in splits and splits[h]!=row["split"]:raise ValueError("Train/validation sequence overlap")
    splits[h]=row["split"]
assert len(splits)==32
cache=root/"data/reference/uniprot/pilot_mapping"
cache.mkdir(parents=True,exist_ok=True)

def lookup(h):
    seq=stage[h]
    raw=by_hash[h]
    name=raw[0]
    gene=aliases.get(name,name.split("(")[0])
    dest=cache/(h+".json")
    if dest.exists():
        r=json.loads(dest.read_text(encoding="utf-8"))
        if r.get("source_sha256")==h:return r
    query=urllib.parse.urlencode({"query":"gene_exact:"+gene+" AND organism_id:9606",
         "format":"json","size":"100","fields":"accession,sequence,gene_names"})
    url="https://rest.uniprot.org/uniprotkb/search?"+query
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"DynaCPI-v4-3D-PilotMapping/0.1"})
        with urllib.request.urlopen(req,timeout=25) as r:
            data=json.load(r)
        candidates=[]
        for item in data.get("results",[]):
            sequence=item.get("sequence",{}).get("value","")
            at=sequence.find(seq)
            if at<0:continue
            candidates.append({"accession":item["primaryAccession"],"start0":at,
                               "end0":at+len(seq),"reference_length":len(sequence),
                               "reference_sha256":hashlib.sha256(sequence.encode()).hexdigest(),
                               "is_reviewed":"reviewed" in item.get("entryType","").lower() and "unreviewed" not in item.get("entryType","").lower()})
        candidates.sort(key=lambda x:(not x["is_reviewed"],x["reference_length"],x["accession"]))
        record={"source_sha256":h,"pilot_split":splits[h],"source_name":name,
          "searched_gene":gene,"query_length":len(seq),
          "candidate_reference_count":len(candidates),
          "reference_matches":candidates[:8],
          "status":"EXACT_CONTIGUOUS_SUBSEQUENCE" if candidates else "NO_EXACT_CONTIGUOUS_UNIPROT_MATCH"}
    except Exception as exc:
        record={"source_sha256":h,"pilot_split":splits[h],"source_name":name,
                "searched_gene":gene,"query_length":len(seq),
                "status":"REFERENCE_LOOKUP_ERROR",
                "reason":type(exc).__name__+": "+str(exc)[:180]}
    dest.write_text(json.dumps(record,indent=2),encoding="utf-8")
    return record

records=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(lookup,h) for h in sorted(splits)]
    for fut in as_completed(futures):records.append(fut.result())
records.sort(key=lambda r:(r["pilot_split"],r["source_name"]))
counts=Counter(r["status"] for r in records)
by_split=defaultdict(Counter)
for r in records:by_split[r["pilot_split"]][r["status"]]+=1
report={"status":"PILOT_REFERENCE_SEQUENCE_SCREEN_ONLY",
        "checked_at_utc":datetime.now(timezone.utc).isoformat(),
        "n_proteins":len(records),"counts":dict(counts),
        "by_split":{k:dict(v) for k,v in by_split.items()},
        "records":records,
        "caveats":["UniProt exact subsequence match does not establish assayed construct identity.",
                   "Gene labels mapped to candidate accessions using manually named aliases; only literal sequence equality accepted.",
                   "Any structure must be mapped from reference positions into query-local indices before GVP.",
                   "A protein without a sequence-exact alignment must not be given fabricated geometry.",
                   "Engineering-only DAVIS pilot; no approved BindingDB cohort or research split."]}
dest=root/"data/engineering_pilot/uniprot_exact_subsequence_survey.json"
dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ("records","caveats")},indent=2),flush=True)
for r in records:
    x=r.get("reference_matches",[])
    m=x[0] if x else {}
    print(json.dumps({"name":r["source_name"],"split":r["pilot_split"],"status":r["status"],
                      "length":r["query_length"],"accession":m.get("accession"),
                      "start1":m.get("start0",0)+1 if m else None,
                      "reference_length":m.get("reference_length")}),flush=True)
if __name__=="__main__":pass
