"""Reproducible, label-blind review triage. Not scientific assay approval.
Conservative flags are reason-for-review, not proof of invalid chemistry.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
source=audit/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl"
hard={"multiple_target_name_annotations","multiple_or_missing_uniprot_ids",
      "within_assay_ph_variation","within_assay_temp_variation",
      "replicates_disagree_over_0_1_pKd","some_multicomponent_smiles",
      "assay_text_unavailable"}
soft={"ph_not_reported","temperature_not_reported"}
counts=Counter()
flag_counts=Counter()
component_cohorts=defaultdict(set)
tier_examples=defaultdict(list)
with source.open(encoding="utf-8") as f:
    for ln in f:
        task=json.loads(ln)
        n=task["n_unique_compounds"]
        fset=set(task["quality_review_flags"])
        for flag in fset:
            flag_counts[flag]+=1
        if n<15:
            tier="not_K5_eligible"
        elif fset & hard:
            tier="requires_priority_review"
        elif fset & soft:
            tier="missing_conditions_review"
        else:
            tier="no_automated_flags"
        counts[tier]+=1
        if n>=15:
            counts["all_K5_tasks"]+=1
            component_cohorts["all_K5"].add(task["protein_component_id"])
            if not fset&hard:
                counts["no_hard_flag_K5_tasks"]+=1
                component_cohorts["no_hard_flag_K5"].add(task["protein_component_id"])
            if not fset:
                counts["no_flag_K5_tasks"]+=1
                component_cohorts["no_flag_K5"].add(task["protein_component_id"])
            if n>=30:
                counts["all_K20_tasks"]+=1
                component_cohorts["all_K20"].add(task["protein_component_id"])
                if not fset&hard:
                    counts["no_hard_flag_K20_tasks"]+=1
                    component_cohorts["no_hard_flag_K20"].add(task["protein_component_id"])
        if len(tier_examples[tier])<10:
            tier_examples[tier].append({"task_id":task["task_id"],
                                      "flags":task["quality_review_flags"]})
report={
"status":"LABEL_BLIND_AUTOMATED_REVIEW_TRIAGE_NOT_ASSAY_APPROVAL",
"tier_task_counts":dict(counts),
"per_flag_task_counts":dict(flag_counts),
"unique_protein_components_by_tier":{k:len(v) for k,v in component_cohorts.items()},
"review_examples":dict(tier_examples),
"rules":{
 "hard_review_flags":sorted(hard),
 "soft_missing_metadata_flags":sorted(soft),
 "no_hard_flag":"not evidence that any assay is truly comparable",
 "missing_conditions":"unknown pH or temperature is not proof of incompatible measurements"
},
"warning":"No target or assay certified. Do not train or split on triage-only output."}
dest=audit/"bindingdb_full_kd_candidate_review_triage.json"
dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="review_examples"},indent=2),flush=True)
