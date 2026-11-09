"""Validate PRE-SPLIT candidate episode identities; not assay approval."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json

root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
files={
"tasks":audit/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl",
"episodes":audit/"bindingdb_full_kd_candidate_episodes_PRE_SPLIT.jsonl",
"rows":audit/"bindingdb_full_kd_compound_median_candidates.jsonl",
"components":audit/"bindingdb_full_kd_protein_components.jsonl"
}
groups={}
for line in files["components"].open(encoding="utf-8"):
    data=json.loads(line)
    for s in data["members"]:
        assert s not in groups
        groups[s]=data["component_id"]

tasks={}
for line in files["tasks"].open(encoding="utf-8"):
    obj=json.loads(line)
    task=obj["task_id"]
    assert task not in tasks
    assert obj["protein_seq_sha256"] in groups
    assert groups[obj["protein_seq_sha256"]]==obj["protein_component_id"]
    assert obj["status"]=="PRE_SPLIT_TECHNICAL_CANDIDATE_NOT_ASSAY_VERIFIED"
    assert hashlib.sha256((obj["protein_seq_sha256"]+"|"+obj["assay_key"]).encode()).hexdigest()==task
    tasks[task]=obj

observed=defaultdict(set)
counters=Counter()
for line in files["rows"].open(encoding="utf-8"):
    obj=json.loads(line)
    task=obj["task_id"]
    assert task in tasks
    compound=obj["compound_sha256"]
    assert compound not in observed[task]
    assert hashlib.sha256(obj["smiles_isomeric"].encode()).hexdigest()==compound
    assert obj["replicates"]==len(obj["reactant_ids"])>=1
    observed[task].add(compound)
    counters["compound_median_records"]+=1

episodes={}
for line in files["episodes"].open(encoding="utf-8"):
    obj=json.loads(line)
    task=obj["task_id"]
    assert task not in episodes
    assert task in tasks and tasks[task]["n_unique_compounds"]>=15
    assert obj["status"]=="DO_NOT_USE_FOR_FINAL_BENCHMARK_UNTIL_ASSAY_QC_AND_SPLIT_LOCK"
    q,s=obj["query_ids"],obj["candidate_support_ids"]
    assert len(q)==10 and len(s)>=5
    assert len(q)==len(set(q)) and len(s)==len(set(s))
    assert not set(q).intersection(s)
    assert set(q).union(s)==observed[task]
    for k in (0,1,2,5,10,20):
        if len(s)>=k:
            assert obj["support_prefixes"][str(k)]==s[:k]
        else:
            assert str(k) not in obj["support_prefixes"]
    episodes[task]=obj
    counters["episodes"]+=1
    if len(s)>=20:
        counters["episodes_K20"]+=1

for task,obj in tasks.items():
    assert obj["n_unique_compounds"]==len(observed[task])
    assert (task in episodes)==(obj["n_unique_compounds"]>=15)
    counters["tasks"]+=1
    if obj["quality_review_flags"]:
        counters["tasks_with_quality_flags"]+=1

report={
"status":"PRE_SPLIT_MANIFEST_IDENTITY_CHECK_PASSED",
"counts":dict(counters),
"checks":["unique task IDs","protein-component membership",
        "source-independent task identity hash","unique compound IDs and canonical identities",
        "10 deterministic disjoint query IDs",
        "support candidates disjoint from queries",
        "nested K prefixes","no final test split or approval status",
        "task sizes equal generated compound records"],
"scope":"Technical identity checks only; assay and construct QC remain pending"}
dest=audit/"bindingdb_full_kd_candidate_episode_manifest_validation.json"
dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
