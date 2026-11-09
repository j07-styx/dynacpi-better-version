"""Preliminary protein-component screen, not an approved holdout split.
Similarity edges: MMseqs2 returned local hits with identity >=.30
and aligned coverage >=.80 of the shorter original sequence.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

root=Path(__file__).resolve().parents[1]
data=root/"data/audit"
fasta=data/"bindingdb_full_kd_sequences.fasta"
hits=data/"bindingdb_full_kd_mmseqs_hits.tsv"
tasksfile=data/"bindingdb_full_kd_task_summary.jsonl"
names=[]
seqs={}
with fasta.open(encoding="utf-8") as f:
    for line in f:
        if line.startswith(">"):
            name=line[1:].strip()
            names.append(name)
        else:
            seqs[name]=line.strip()
assert set(names)==set(seqs)
parent={x:x for x in names}
size={x:1 for x in names}
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]
        x=parent[x]
    return x
def union(a,b):
    x,y=find(a),find(b)
    if x==y:
        return
    if size[x]<size[y]:x,y=y,x
    parent[y]=x
    size[x]+=size[y]

stats=Counter()
with hits.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        stats["aligned_rows"]+=1
        cols=line.rstrip("\r\n").split("\t")
        if len(cols)!=8:
            stats["unexpected_alignment_columns"]+=1
            continue
        q,t,identity,alnlen,qlen,tlen,qcov,tcov=cols
        if q not in parent or t not in parent:
            stats["unknown_query_target"]+=1
            continue
        if q==t:
            stats["self_hits"]+=1
            continue
        try:
            ident=float(identity)
            qlen=int(qlen)
            tlen=int(tlen)
            coverage=float(qcov) if qlen<=tlen else float(tcov)
        except (ValueError,ZeroDivisionError):
            stats["invalid_alignment_numbers"]+=1
            continue
        if ident>=0.30 and coverage>=0.80:
            stats["accepted_directed_edges"]+=1
            union(q,t)
components=defaultdict(list)
for x in names:
    components[find(x)].append(x)
tasks=[json.loads(line) for line in tasksfile.open(encoding="utf-8")]
eligible15=[t for t in tasks if t["unique_canonical_compounds"]>=15]
eligible30=[t for t in tasks if t["unique_canonical_compounds"]>=30]
def cohort(rows):
    clusters=defaultdict(list)
    for t in rows:
        clusters[find(t["sequence_sha256"])].append(t)
    return {"task_count":len(rows),
            "unique_seqs":len(set(x["sequence_sha256"] for x in rows)),
            "eligible_components":len(clusters),
            "largest_component_tasks":sorted((len(x) for x in clusters.values()),reverse=True)[:20],
            "largest_component_distinct_sequences":sorted((len(set(t["sequence_sha256"] for t in x)) for x in clusters.values()),reverse=True)[:20]}
result={
  "scope":"Preliminary full-BindingDB exact-looking-Kd pool; no final assay/construct eligibility",
  "alignment_algorithm":"MMseqs2 easy-search all-vs-all with high sensitivity; local candidate search may miss remote homologies",
  "edge_policy":"fident >= 0.30 and qcov or tcov for the shorter seq >= 0.80; connected components, not representative-only greedy clusters",
  "stats":dict(stats),
  "all_eligible_sequence_count":len(names),
  "protein_components_all":len(components),
  "largest_component_sizes":sorted((len(c) for c in components.values()),reverse=True)[:20],
  "ge15":cohort(eligible15),
  "ge30":cohort(eligible30),
  "caveats":[
    "Sequence-based components are operational and do not establish independent local functional domains.",
    "Missing or ambiguous target-construct metadata and assay compatibility may reduce eligible tasks.",
    "Protein clustering was run over source candidates before chemical task-level assay QC.",
    "No train/validation/test assignment has been created.",
    "MMseqs search sensitivity means false negatives possible; manual domain/family challenge checks remain required."
  ]
}
out=data/"bindingdb_full_kd_mmseqs_component_screen.json"
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
with (data/"bindingdb_full_kd_protein_components.jsonl").open("w",encoding="utf-8") as f:
    for items in sorted(components.values(),key=lambda v:(-len(v),v[0])):
        canonical=min(items)
        f.write(json.dumps({"component_id":canonical,"members":sorted(items)})+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("caveats","largest_component_sizes")},indent=2),flush=True)
