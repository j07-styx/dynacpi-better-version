"""Simulate feasibility of five-role source-disjoint split quotas without labels.

Monte Carlo permutations are hypothetical and are NEVER saved as task-role
assignments. This evaluates size imbalance under existing provisional MMseqs
protein grouping versus conservative DOI/PMID/patent-linked source grouping.

No TRAIN/DEV/TEST modeling, affinity labels, official reserved test data, or
approved split artifacts are accessed. The source and protein groups are
scientifically provisional.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dynacpi.balm_public_data_dev import ROOT
from scripts.prioritize_bindingdb_component_source_review_dev import load_jsonl

AUDIT=ROOT/"data/audit"
REPORT=ROOT/"research/bindingdb_publication_family_split_feasibility_simulation_20261108.json"
MARKDOWN=ROOT/"research/BINDINGDB_PUBLICATION_FAMILY_SPLIT_FEASIBILITY_20261108.md"
SOURCE=ROOT/"research/bindingdb_publication_connected_protein_review_preflight_20261108.json"
ROLES=("base_train","base_validation","adapter_train","adapter_validation","test")
FRACTIONS=(.6,.1,.1,.1,.1)
DRAWS=1000
SEED=20261108

def quotas(n):
    assert n>=len(ROLES) and np.isclose(sum(FRACTIONS),1.)
    target=np.array(FRACTIONS)*n
    counted=np.floor(target).astype(int)
    missing=n-int(sum(counted))
    # Fixed tie break on ROLE ORDER; label-blind and no task composition.
    index=sorted(range(len(target)),key=lambda i:(-(target[i]-counted[i]),i))
    for i in index[:missing]:
        counted[i]+=1
    assert int(counted.sum())==n
    # The formal cohort gate requires >=3 independent groups per role. At
    # tiny n (e.g. 25), exact 60/10/10/10/10 percentages cannot both hold.
    # Move groups from the most above-target quota until minimums are met.
    for i in range(len(ROLES)):
        while counted[i]<3:
            donors=[j for j in range(len(ROLES)) if counted[j]>3]
            if not donors:
                raise ValueError("Insufficient study families to allocate at least three per role")
            donor=max(donors,key=lambda j:(counted[j]-target[j],counted[j],-j))
            counted[donor]-=1
            counted[i]+=1
    assert int(counted.sum())==n and np.all(counted>=3)
    return dict(zip(ROLES,[int(x) for x in counted]))

def simulate_group_task_distributions(taskcounts,draws=DRAWS,seed=SEED):
    """Return only aggregate distributions, NEVER the sampled assignments."""
    ids=sorted(taskcounts)
    weights=np.asarray([taskcounts[x] for x in ids],dtype=np.int32)
    assert len(ids)>=25 and len(ids)==len(set(ids))
    assert all(int(v)>0 for v in weights)
    tasktotal=int(weights.sum())
    allocations=quotas(len(ids))
    lengths=np.asarray([allocations[k] for k in ROLES],dtype=np.int32)
    cut=np.cumsum(lengths)
    rng=np.random.default_rng(seed)
    totals=np.zeros((draws,len(ROLES)),dtype=np.int32)
    largest_role=np.zeros(draws,dtype=np.int32)
    largest_index=int(weights.argmax())
    for draw in range(draws):
        permutation=rng.permutation(len(ids))
        for idx,segment in enumerate(np.split(permutation,cut[:-1])):
            totals[draw,idx]=int(weights[segment].sum())
            if largest_index in segment:
                largest_role[draw]=idx
        assert int(totals[draw].sum())==tasktotal
    result={}
    for i,key in enumerate(ROLES):
        n=totals[:,i].astype(float)
        result[key]={
            "family_quota":allocations[key],
            "expected_task_share":FRACTIONS[i],
            "task_count_min":int(n.min()),
            "task_count_p05":float(np.percentile(n,5)),
            "task_count_median":float(np.median(n)),
            "task_count_p95":float(np.percentile(n,95)),
            "task_count_max":int(n.max()),
            "task_share_p05":float(np.percentile(n/tasktotal,5)),
            "task_share_median":float(np.median(n/tasktotal)),
            "task_share_p95":float(np.percentile(n/tasktotal,95)),
            "fraction_of_draws_above_double_desired_task_share":float(
                np.mean(n/tasktotal>2*FRACTIONS[i])),
            "fraction_of_draws_below_half_desired_task_share":float(
                np.mean(n/tasktotal<FRACTIONS[i]/2)),
            "draws_with_largest_family":int(np.sum(largest_role==i)),
        }
    return {
        "families":len(ids),
        "total_technical_tasks":tasktotal,
        "largest_family_tasks":int(weights.max()),
        "largest_family_task_fraction":float(weights.max()/tasktotal),
        "family_count_quota_by_role":allocations,
        "monte_carlo_draws":draws,
        "random_seed":seed,
        "role_task_count_distributions":result,
        "no_actual_split_or_task_assignment_written":True,
    }

def main():
    if REPORT.exists() or MARKDOWN.exists():
        raise FileExistsError("Split feasibility simulation already exists")
    source=json.loads(SOURCE.read_text(encoding="utf8"))
    mapping=source["linkage"]["source_family_by_protein_component"]
    protein_counts=Counter()
    publication_counts=Counter()
    observed_taskids=set()
    for task in load_jsonl(AUDIT/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl"):
        if task["n_unique_compounds"]<15:continue
        taskid=task["task_id"]
        if taskid in observed_taskids:raise ValueError("Duplicated task ID")
        observed_taskids.add(taskid)
        component=task["protein_component_id"]
        if component not in mapping:raise ValueError("Unknown source protein family")
        protein_counts[component]+=1
        publication_counts[mapping[component]]+=1
    assert len(observed_taskids)==1146
    assert len(protein_counts)==168
    assert len(publication_counts)==142
    assert sum(protein_counts.values())==sum(publication_counts.values())==1146
    baseline=simulate_group_task_distributions(protein_counts)
    study=simulate_group_task_distributions(publication_counts)
    assert source["overview"]["largest_source_family_task_count"]==study["largest_family_tasks"]
    result={
        "status":"PRE_SPLIT_SOURCE_LINKED_TASK_COUNT_IMBALANCE_SIMULATION_NOT_APPROVED_NOT_FROZEN",
        "simulations":{
            "provisional_protein_MMseqs_groups_only":baseline,
            "publication_PMid_DOI_patent_connected_groups":study,
        },
        "comparison_axes":[
            "Task count distribution under fixed group-count quotas",
            "Probability largest source group lands in each role",
            "Unbalanced task share under random group-only allocations",
        ],
        "input_source_preflight_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_task_manifest_sha256":source["data_source_integrity_sha256"][
            "bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl"],
        "role_fractions":dict(zip(ROLES,FRACTIONS)),
        "no_actual_split_or_task_assignment_written":True,
        "scientific_gate_status":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "warnings":[
            "These are hypothetical role allocations only; none is selected, published, or frozen.",
            "A fixed NUMBER of study groups per role does not ensure balanced task or chemical coverage.",
            "The large source family could cause extreme task distribution imbalance.",
            "One publication may cover distinct assays; missing DOI/PMID/patent links may undermerge source groups.",
            "No affinity labels or model predictions used for this analysis.",
            "The prospective test, reviewed assay cohort and frozen split remain untouched.",
        ],
    }
    REPORT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    lines=[
        "# Publication-source group imbalance: hypothetical split feasibility only",
        "",
        "**No actual split was selected or frozen. No affinity labels or prediction scores used.**",
        "",
        "The candidate PRE_SPLIT group graph changes from 168 preliminary MMseqs protein components to 142 DOI/PMID/patent-connected protein families. The larger family count might seem enough for five roles, but 530 of 1,146 tasks belong to one source-connected family. Group-quantity balancing alone can leave test or validation roles dominated by one publication-linked family.",
        "",
        "We ran 1,000 deterministic RNG draws of RANDOM family-to-role allocations (no identities saved), distributing 60% of group counts to base training and 10% each to base validation, adaptation training, adaptation validation and test. Only resulting task-count distributions were saved.",
        "",
        "| Source group policy / role | Number of groups allocated | Median tasks [5th, 95th percentile] | Task share 5th–95th percentile | Draws above 2× target share |",
        "|---|---:|---|---|---:|",
    ]
    for group_type,x in (
        ("Protein groups only",baseline),
        ("Publication-linked groups",study)
    ):
        for role in ROLES:
            y=x["role_task_count_distributions"][role]
            lines.append(f"| {group_type}, {role} | {y['family_quota']} | {y['task_count_median']:.0f} [{y['task_count_p05']:.0f}, {y['task_count_p95']:.0f}] | {100*y['task_share_p05']:.1f}%–{100*y['task_share_p95']:.1f}% | {100*y['fraction_of_draws_above_double_desired_task_share']:.1f}% |")
    lines+=["",
        "## Interpretation",
        "",
        f"- Source-group largest family: {study['largest_family_tasks']} tasks, **{100*study['largest_family_task_fraction']:.1f}%** of the candidate task inventory.",
        "- A fixed group fraction is not the same as a fixed data-size fraction, and assigning a single large source family to a small validation/test role yields extreme imbalance.",
        "- Source IDs are metadata hints rather than certified true study batches. Missing DOI/PMID links can conceal additional overlap; extra group unions may also overmerge independent assays.",
        "- These results can guide future prespecified, label-blind quota and group-size policies once real assay review has approved a source-backed cohort. They must NOT be used to select favorable heldout group identities after inspecting labels.",
        "",
        "## Reproduction and status",
        "",
        "- scripts/simulate_bindingdb_publication_group_balance_dev.py",
        "- research/bindingdb_publication_family_split_feasibility_simulation_20261108.json",
        "- tests_v4/bindingdb_publication_group_preflight_test.py",
        "- No actual task role mapping, frozen split or approved cohort was generated.",
        "- Scientific cohort approval remains BLOCKED_UNAPPROVED_NOT_TRAINING_READY.",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines),encoding="utf8")
    print("SIMULATION",json.dumps({
        "protein_group_count":baseline["families"],
        "publication_family_count":study["families"],
        "source_largest_tasks":study["largest_family_tasks"],
        "source_test_role":study["role_task_count_distributions"]["test"],
        "source_base_validation":study["role_task_count_distributions"]["base_validation"],
    }),flush=True)

if __name__=="__main__":main()
