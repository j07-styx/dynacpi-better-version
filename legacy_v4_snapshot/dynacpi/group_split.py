"""Label-blind deterministic protein-component allocation.

Designed for source-approved tasks only. The module never reads affinity labels.
Its output is a proposal: a separate approval/freeze step is required before
training. Keep the same component graph for every role to block protein leakage.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from dataclasses import dataclass
import hashlib
from math import floor
from typing import Dict, Iterable, Mapping, Sequence

ROLES=("base_train","base_validation","adapter_train","adapter_validation","test")
FRACTIONS=(0.60,0.10,0.10,0.10,0.10)


def allocate_quotas(n: int, fractions=FRACTIONS) -> Dict[str,int]:
    if len(fractions)!=5 or abs(sum(fractions)-1.0)>1e-8:
        raise ValueError("Invalid group allocation fractions")
    if n<25:
        raise ValueError("Insufficient independent protein components for five-way study (need >=25)")
    fractional=[n*x for x in fractions]
    quotas=[int(floor(x)) for x in fractional]
    remainder=n-sum(quotas)
    for j in sorted(range(5),key=lambda i:(-(fractional[i]-quotas[i]),i))[:remainder]:
        quotas[j]+=1
    if min(quotas)<3:
        raise ValueError("Insufficient protein groups in a validation/test role")
    return dict(zip(ROLES,quotas))


def plan_component_split(tasks: Sequence[Mapping], seed: int = 1105) -> dict:
    """Group-level, label-blind split proposal.

    Task dictionaries must carry only IDs, per-task sample counts and component
    IDs. No 'y', 'pKd', 'affinity' or measurement features are permitted.
    """
    if not tasks:raise ValueError("No candidate tasks")
    groups=defaultdict(list)
    seen=set()
    for task in tasks:
        disallowed={"y","pkd","pKd","affinity","label","median_pKd","query_labels"}
        if disallowed & set(task):
            raise ValueError("Split inputs may not contain affinity labels")
        tid=task["task_id"]
        gid=task["protein_component_id"]
        if tid in seen:raise ValueError("Duplicate task ID")
        if not tid or not gid:raise ValueError("Missing task/component ID")
        seen.add(tid)
        n=int(task["n_unique_compounds"])
        if n<15:raise ValueError("Ineligible task has fewer than 15 distinct compounds")
        groups[gid].append((tid,n))
    quotas=allocate_quotas(len(groups))
    all_group_keys=list(groups)
    # Highest task-count components allocated first to avoid one giant group
    # unexpectedly filling a 10%-quota role. Tie-breaker is seed+component hash.
    order=sorted(all_group_keys,
      key=lambda gid:(-len(groups[gid]),
                      hashlib.sha256((f"split|{seed}|{gid}").encode()).hexdigest()))
    n_tasks=len(tasks)
    task_targets={r:n_tasks*f for r,f in zip(ROLES,FRACTIONS)}
    assigned={r:[] for r in ROLES}
    role_task_counts=Counter()
    for gid in order:
        options=[r for r in ROLES if len(assigned[r])<quotas[r]]
        if not options:raise AssertionError("Missing split capacity")
        ng=len(groups[gid])
        def score(role):
            # Encourage a balanced task tally while exactly satisfying component
            # quotas; no label statistics are used.
            prospective=role_task_counts[role]+ng
            ratio=prospective/max(task_targets[role],1.0)
            remaining_components=quotas[role]-len(assigned[role])
            return (ratio, -remaining_components, ROLES.index(role))
        # A very large group is assigned based on label-independent imbalance.
        chosen=min(options,key=score)
        assigned[chosen].append(gid)
        role_task_counts[chosen]+=ng
    if sum(map(len,assigned.values()))!=len(groups):
        raise AssertionError("Not all groups assigned")
    role_of={g:r for r,group_ids in assigned.items() for g in group_ids}
    if len(role_of)!=len(groups):
        raise AssertionError("Protein component overlaps between roles")
    output={
      "status":"DRY_RUN_UNAPPROVED",
      "seed":seed,
      "quotas":quotas,
      "task_counts":dict(role_task_counts),
      "component_counts":{r:len(ids) for r,ids in assigned.items()},
      "role_of_component":role_of,
      "task_roles":{tid:role_of[gid] for gid,items in groups.items() for tid,_ in items},
      "policy":"Greedy task-count-balancing across exact group quotas, largest component first; hash tie break; NO LABELS",
      "warnings":["Preliminary homology components may change after construct and domain review",
                  "Imbalanced large components limit role task-count matching",
                  "A separate human-reviewed approval and final split freeze are required"]}
    if set(output["task_roles"])!=seen:
        raise AssertionError("Tasks lost from allocation")
    return output
