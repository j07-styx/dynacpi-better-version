"""Dry-run ONLY: estimate heterogeneity of five-way protein group split sizes.

No task labels or held-out outcomes used; no split is committed or selected.
"""
from pathlib import Path
from collections import defaultdict, Counter
import json
import numpy as np
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
comp={}
for line in (audit/"bindingdb_full_kd_protein_components.jsonl").open(encoding="utf-8"):
    d=json.loads(line)
    for seq in d["members"]:
        if seq in comp:raise ValueError("Sequence assigned to multiple components")
        comp[seq]=d["component_id"]
tasks=[json.loads(line) for line in (audit/"bindingdb_full_kd_task_summary.jsonl").open(encoding="utf-8")]
eligible15=defaultdict(list)
eligible30=defaultdict(list)
for task in tasks:
    gid=comp.get(task["sequence_sha256"])
    if gid is None:raise ValueError("Unclustered candidate sequence")
    if task["unique_canonical_compounds"]>=15:
        eligible15[gid].append(task)
    if task["unique_canonical_compounds"]>=30:
        eligible30[gid].append(task)
n=len(eligible15)
groups=sorted(eligible15)
quotas=[round(n*v) for v in (0.6,0.1,0.1,0.1)]
quotas.append(n-sum(quotas))
if quotas[-1]<1:raise ValueError("No test group")
rng=np.random.default_rng(1105)
records=[]
for _ in range(1000):
    shuffled=list(rng.permutation(groups))
    starts=np.cumsum([0]+quotas)
    row={}
    for role,a,b in zip(("base_train","base_validation","adapt_train","adapt_validation","test"),starts[:-1],starts[1:]):
        selected=shuffled[int(a):int(b)]
        row[role+"_clusters"]=len(selected)
        row[role+"_tasks_ge15"]=sum(len(eligible15[x]) for x in selected)
        row[role+"_tasks_ge30"]=sum(len(eligible30.get(x,[])) for x in selected)
        row[role+"_clusters_ge30"]=sum(bool(eligible30.get(x)) for x in selected)
    records.append(row)
def dist(key):
    vs=np.array([r[key] for r in records])
    return {"min":int(vs.min()),"median":float(np.median(vs)),
            "p05":float(np.quantile(vs,.05)), "p95":float(np.quantile(vs,.95)),
            "max":int(vs.max())}
metrics={key:dist(key) for key in records[0] if key.endswith(("_tasks_ge15","_tasks_ge30","_clusters_ge30"))}
result={
"scope":"1000 hypothetical random allocations of protein components, NOT an actual split",
"source":"Full November 2026 BindingDB provisional exact-looking-Kd task pool",
"eligible_ge15":{"tasks":sum(map(len,eligible15.values())),"components":n},
"eligible_ge30":{"tasks":sum(map(len,eligible30.values())),"components":len(eligible30)},
"cluster_quotas_60_10_10_10_10":dict(zip(("base_train","base_val","adapt_train","adapt_val","test"),quotas)),
"size_distributions":metrics,
"warnings":[
 "No labels were used; simulations do not establish comparable assay conditions or fair split composition.",
 "No actual split was selected or frozen; simulations must not be used to choose a favorable test group.",
 "The large connected component introduces extreme task imbalance even when group counts are balanced.",
 "The K=20 eligible subset loses many tasks and independent clusters under filtering."
]}
(audit/"bindingdb_full_kd_split_feasibility_simulation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
