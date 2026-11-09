"""Sensitivity of preliminary full-candidate protein grouping to alignment length.
Never use this for final split assignment; this is a QC diagnostic.
"""
from pathlib import Path
from collections import defaultdict, Counter
import json

root=Path(__file__).resolve().parents[1]
data=root/"data/audit"
fasta=data/"bindingdb_full_kd_sequences.fasta"
hits=data/"bindingdb_full_kd_mmseqs_hits.tsv"
summaries=data/"bindingdb_full_kd_task_summary.jsonl"
seqs={}
with fasta.open(encoding="utf-8") as f:
    for ln in f:
        if ln.startswith(">"):
            key=ln[1:].strip()
        else:seqs[key]=ln.strip()
keys=set(seqs)
tasks=[json.loads(x) for x in summaries.open(encoding="utf-8")]
thresholds=(0,40,80,100,150)
parent={cutoff:{x:x for x in keys} for cutoff in thresholds}
size={cutoff:{x:1 for x in keys} for cutoff in thresholds}
accepted=Counter()
def find(pa,x):
    while pa[x]!=x:
        pa[x]=pa[pa[x]]
        x=pa[x]
    return x
def union(cutoff,q,t):
    pa=parent[cutoff]
    a,b=find(pa,q),find(pa,t)
    if a==b:return
    if size[cutoff][a]<size[cutoff][b]:a,b=b,a
    pa[b]=a
    size[cutoff][a]+=size[cutoff][b]
short_edges=[]
with hits.open(encoding="utf-8") as f:
    for ln in f:
        q,t,identity,alnlen,qlen,tlen,qcov,tcov=ln.rstrip("\n\r").split("\t")
        if q==t:continue
        n=int(alnlen)
        cov=float(qcov) if int(qlen)<=int(tlen) else float(tcov)
        if float(identity)<0.30 or cov<0.80:continue
        for cutoff in thresholds:
            if n>=cutoff:
                union(cutoff,q,t)
                accepted[str(cutoff)]+=1
        if n<80 and len(short_edges)<8:
            short_edges.append({"q_len":int(qlen),"t_len":int(tlen),
                                "alignment_len":n,"identity":float(identity),"coverage_shorter":cov})
def cohort(cutoff,k):
    grouped=defaultdict(list)
    for t in tasks:
        if t["unique_canonical_compounds"]<k:continue
        grouped[find(parent[cutoff],t["sequence_sha256"])].append(t)
    return {"components":len(grouped),
            "tasks":sum(map(len,grouped.values())),
            "largest_task_counts":sorted(map(len,grouped.values()),reverse=True)[:5],
            "largest_seq_counts":sorted((len({x["sequence_sha256"] for x in a}) for a in grouped.values()),reverse=True)[:5]}
report={
  "scope":"Preliminary full dataset sequence similarity only, no final protein split",
  "accepted_directed_edges_by_min_alignment_length":dict(accepted),
  "sensitivity":{
      str(c):{"all_components":len({find(parent[c],x) for x in keys}),
              "ge15":cohort(c,15), "ge30":cohort(c,30)}
      for c in thresholds
  },
  "short_hit_examples":short_edges,
  "note":"Short local matches may spuriously merge unrelated full-length proteins; sensitivity only, threshold choice requires a documented biological criterion."
}
path=data/"bindingdb_full_kd_mmseqs_threshold_sensitivity.json"
path.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
