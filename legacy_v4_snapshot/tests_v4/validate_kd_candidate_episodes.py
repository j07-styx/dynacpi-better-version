"""Verify quarantined support/query manifests without accessing measured labels."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
tasks={}
for ln in (audit/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl").open(encoding="utf-8"):
    x=json.loads(ln)
    assert x["task_id"] not in tasks
    tasks[x["task_id"]]=x
compounds=defaultdict(set)
for ln in (audit/"bindingdb_full_kd_compound_median_candidates.jsonl").open(encoding="utf-8"):
    x=json.loads(ln)
    key=x["task_id"]
    compound_id=x["compound_sha256"]
    assert compound_id not in compounds[key],(key,compound_id)
    assert hashlib.sha256(x["smiles_isomeric"].encode()).hexdigest()==compound_id
    assert x["replicates"]>=1 and x["range_pKd"]>=0
    compounds[key].add(compound_id)
episodes={}
for ln in (audit/"bindingdb_full_kd_candidate_episodes_PRE_SPLIT.jsonl").open(encoding="utf-8"):
    d=json.loads(ln)
    tid=d["task_id"]
    assert tid in tasks and tid not in episodes
    q=d["query_ids"]
    s=d["candidate_support_ids"]
    assert len(q)==10 and len(s)>=5 and len(set(q))==10
    assert len(set(s))==len(s) and not set(q).intersection(s)
    assert set(q).union(s)==compounds[tid]
    expected=sorted(compounds[tid],key=lambda x:hashlib.sha256(("query_selection_v4|"+tid+"|"+x).encode()).hexdigest())
    assert q==expected[:10] and s==expected[10:]
    draws=d["support_draws"]
    assert len(draws)==10
    distinct=set()
    for ix,draw in enumerate(draws):
        assert draw["draw_index"]==ix
        ranked=sorted(s,key=lambda x:hashlib.sha256(("support_draw_v4|"+tid+"|"+str(ix)+"|"+x).encode()).hexdigest())
        prefixes=draw["support_prefixes"]
        for ks,prefix in prefixes.items():
            k=int(ks)
            assert len(prefix)==k and prefix==ranked[:k]
        assert "0" in prefixes and "5" in prefixes
        distinct.add(tuple(prefixes["5"]))
    assert d["distinct_k5_support_draws"]==len(distinct)
    episodes[tid]=d
assert len(tasks)==len(compounds)
assert all(t["n_unique_compounds"]==len(compounds[tid]) for tid,t in tasks.items())
assert len(episodes)==sum(t["n_unique_compounds"]>=15 for t in tasks.values())
summary={"status":"PASS","scope":"PRE_SPLIT TECHNICAL CANDIDATE MANIFESTS ONLY; no scientific cohort or trained model",
         "tasks_validated":len(tasks),"eligible_episode_tasks":len(episodes),
         "compound_identity_groupings":sum(map(len,compounds.values())),
         "invariants":["unique task and compound identities","median replicate metadata nonnegative",
                       "10 queries and >=5 supports disjoint","deterministic label-independent SHA order",
                       "K prefixes nested","episode compound universe matches task","no train-test split is committed"]}
dest=audit/"bindingdb_full_kd_pre_split_manifest_validation.json"
dest.write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2),flush=True)
