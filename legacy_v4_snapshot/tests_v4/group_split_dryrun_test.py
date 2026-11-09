"""Engineering-only validation of group disjointness and task-count balance."""
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import json
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from dynacpi.group_split import plan_component_split,ROLES
manifest=root/"data/audit/bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl"
eligible=[]
for ln in manifest.open(encoding="utf-8"):
    item=json.loads(ln)
    if item["n_unique_compounds"]>=15:
        eligible.append({k:item[k] for k in ("task_id","protein_component_id","n_unique_compounds")})
assert len(eligible)==1146
trial=plan_component_split(eligible,seed=1105)
repeat=plan_component_split(eligible,seed=1105)
assert trial==repeat
assert trial["status"]=="DRY_RUN_UNAPPROVED"
assert set(trial["task_roles"])=={t["task_id"] for t in eligible}
assert sum(trial["component_counts"].values())==168
assert sum(trial["task_counts"].values())==1146
assert all(trial["component_counts"][r]>=3 for r in ROLES)
role_of=trial["role_of_component"]
assert len(role_of)==168
for item in eligible:
    assert trial["task_roles"][item["task_id"]]==role_of[item["protein_component_id"]]
try:
    plan_component_split(eligible+[{"task_id":"invalid","protein_component_id":"invalid",
                                    "n_unique_compounds":15,"pKd":5.5}])
except ValueError as exc:
    assert "affinity labels" in str(exc)
else:raise AssertionError("Affinity values were accepted by split policy")
k20_tasks=Counter()
for e in eligible:
    if e["n_unique_compounds"]>=30:
        k20_tasks[trial["task_roles"][e["task_id"]]]+=1
components_by_role=defaultdict(set)
for e in eligible:
    if e["n_unique_compounds"]>=30:
        components_by_role[trial["task_roles"][e["task_id"]]].add(e["protein_component_id"])
# Do not write an actual assignment: this is not a scientifically authorized split.
fingerprint=hashlib.sha256(json.dumps(sorted(role_of.items())).encode()).hexdigest()
report={
 "status":"PASS_DRY_RUN_ONLY",
 "dataset":"Full BindingDB preliminary >=15-compound candidate groups, unreviewed",
 "tasks":len(eligible),"component_count":len(role_of),
 "role_component_counts":trial["component_counts"],
 "role_task_counts":trial["task_counts"],
 "K20_role_task_counts":{r:k20_tasks[r] for r in ROLES},
 "K20_role_component_counts":{r:len(components_by_role[r]) for r in ROLES},
 "assignment_sha256_for_reproducibility":fingerprint,
 "checks":["disjoint component roles","deterministic seed","all tasks included",
           "no affinity labels accepted","no final split or assignment written"],
 "warning":"Dry run is not a benchmark; no training or test split is frozen."}
dest=root/"data/audit/bindingdb_full_kd_group_split_dryrun_policy.json"
dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
