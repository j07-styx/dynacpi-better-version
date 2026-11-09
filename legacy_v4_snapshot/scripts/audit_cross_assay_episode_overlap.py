"""Label-blind audit of compound identity reused across same-protein assay tasks.

This is an episode-contamination screen, not a declaration of data leakage:
distinct assays can legitimately share compounds but should not silently
share query labels as supervised context within a test evaluation.
"""
from collections import defaultdict, Counter
from pathlib import Path
import json, hashlib
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
tasks={}
for ln in (audit/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl").open(encoding="utf-8"):
    t=json.loads(ln);tasks[t["task_id"]]=t
episodes={}
byseq=defaultdict(list)
for ln in (audit/"bindingdb_full_kd_candidate_episodes_PRE_SPLIT.jsonl").open(encoding="utf-8"):
    e=json.loads(ln);key=e["task_id"]
    episodes[key]=e
    byseq[tasks[key]["protein_seq_sha256"]].append(key)
cnt=Counter()
examples=[]
for seq,keys in byseq.items():
    if len(keys)<2:
        continue
    cnt["sequences_with_multiple_eligible_assays"]+=1
    query_index=defaultdict(set)
    support_index=defaultdict(set)
    for tid in keys:
        for c in episodes[tid]["query_ids"]:
            query_index[c].add(tid)
        for c in episodes[tid]["candidate_support_ids"]:
            support_index[c].add(tid)
    if len(query_index)==0:continue
    collision_query_support={cid for cid,qt in query_index.items() if support_index.get(cid, set()) - qt}
    collision_query_query={cid for cid,qt in query_index.items() if len(qt)>1}
    if collision_query_support or collision_query_query:
        cnt["sequences_with_cross_assay_compound_overlap"]+=1
        cnt["cross_assay_query_in_support_unique_compounds"]+=len(collision_query_support)
        cnt["cross_assay_query_in_query_unique_compounds"]+=len(collision_query_query)
        if len(examples)<10:
            examples.append({"sequence_sha256":seq,
                             "assay_task_count":len(keys),
                             "query_to_other_support_shared_compounds":len(collision_query_support),
                             "query_to_other_query_shared_compounds":len(collision_query_query)})
    cnt["candidate_eligible_assay_tasks_on_multi_assay_targets"]+=len(keys)
cnt["total_eligible_tasks"]=len(episodes)
cnt["total_eligible_sequence_targets"]=len(byseq)
report={
"status":"PRE_SPLIT_REVIEW_ONLY",
"counts":dict(cnt),
"examples":examples,
"meaning":"Same protein compound appearing across two assays is not necessarily duplicate labeling; cross-assay reuse can confound few-shot evaluation or create query-label leakage if episodes are pooled without strict task boundaries.",
"recommended_constraint":"When constructing final evaluation support episodes, preserve explicit task/assay boundaries; never make held-out query labels available through another episode or use test-target query labels for model fitting.",
"scope":"Compound identities only; no affinity values used; no scientific split allocated"}
dest=audit/"bindingdb_full_kd_cross_assay_episode_overlap.json"
dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
