"""Deterministic label-free split feasibility; never creates an approved split."""
from collections import Counter, defaultdict
import hashlib
import json

ROLES=("base_train","base_validation","adapter_train","adapter_validation","test")
FIELDS={"task_id","protein_component_id","compound_id","scaffold_id"}
def allocate(keys, seed, axis):
    keys=sorted(set(keys),key=lambda k:hashlib.sha256(f"{axis}|{seed}|{k}".encode()).hexdigest())
    n=len(keys)
    sizes=[int(n*.6)]+[int(n*.1)]*3
    sizes.append(n-sum(sizes))
    mapping={};offset=0
    for role,size in zip(ROLES,sizes):
        for key in keys[offset:offset+size]:mapping[key]=role
        offset+=size
    return mapping

def plan_axes(rows, seed=1105):
    if not rows:raise ValueError("Empty metadata")
    task_component={};compound_scaffold={};seen=set()
    for r in rows:
        if set(r)!=FIELDS or any(not isinstance(v,str) or not v for v in r.values()):
            raise ValueError("Expected only nonempty identity metadata; labels are forbidden")
        key=(r["task_id"],r["compound_id"])
        if key in seen:raise ValueError("Duplicate task-compound row")
        seen.add(key)
        for mapping,k,v in ((task_component,r["task_id"],r["protein_component_id"]),
                            (compound_scaffold,r["compound_id"],r["scaffold_id"])):
            if k in mapping and mapping[k]!=v:raise ValueError("Inconsistent identity mapping")
            mapping[k]=v
    protein=allocate(task_component.values(),seed,"protein")
    scaffold=allocate(compound_scaffold.values(),seed,"scaffold")
    assignments={}
    for mode in ("protein","scaffold","dual"):
        assignment={}
        for r in rows:
            p=protein[r["protein_component_id"]];s=scaffold[r["scaffold_id"]]
            role=p if mode=="protein" else s if mode=="scaffold" else p if p==s else None
            assignment[(r["task_id"],r["compound_id"])]=role
        assignments[mode]=assignment
    return assignments

def summarize(rows, assignments):
    result={}
    for mode,mapping in assignments.items():
        groups={role:defaultdict(set) for role in ROLES};role_of_id=defaultdict(set)
        for r in rows:
            role=mapping[(r["task_id"],r["compound_id"])]
            if role is None:continue
            groups[role][r["task_id"]].add(r["compound_id"])
            axes=("protein_component_id",) if mode=="protein" else ("scaffold_id","compound_id") if mode=="scaffold" else ("protein_component_id","scaffold_id","compound_id")
            for axis in axes:role_of_id[(axis,r[axis])].add(role)
        if any(len(v)>1 for v in role_of_id.values()):raise AssertionError("Cross-role leakage")
        counts={}
        for role,tasks in groups.items():
            eligible={t for t,c in tasks.items() if len(c)>=15}
            components={r["protein_component_id"] for r in rows if r["task_id"] in eligible}
            counts[role]={"pairs":sum(map(len,tasks.values())),"tasks":len(tasks),
                "tasks_with_15_compounds":len(eligible),"components_with_eligible_tasks":len(components)}
        digest=hashlib.sha256(json.dumps(sorted((t,c,v) for (t,c),v in mapping.items()),separators=(",",":")).encode()).hexdigest()
        result[mode]={"roles":counts,"excluded_cross_axis_pairs":sum(v is None for v in mapping.values()),
            "assignment_sha256":digest,"disjointness_verified":True}
    return result
