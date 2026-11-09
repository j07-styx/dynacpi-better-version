"""Create a blinded (no affinity values) metadata review sample across components.
Review queue is NOT an approved cohort or a scientific split.
"""
from pathlib import Path
import csv,hashlib,json
from collections import defaultdict
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
assays={}
for line in (audit/"bindingdb_full_kd_assay_contexts.jsonl").open(encoding="utf-8"):
    obj=json.loads(line)
    assays[(obj["sequence_sha256"],obj["assay_key"])]=obj

hard={"multiple_target_name_annotations","multiple_or_missing_uniprot_ids",
      "within_assay_ph_variation","within_assay_temp_variation",
      "replicates_disagree_over_0_1_pKd","some_multicomponent_smiles",
      "assay_text_unavailable"}
eligible=defaultdict(list)
for line in (audit/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl").open(encoding="utf-8"):
    task=json.loads(line)
    if task["n_unique_compounds"] < 15 or hard.intersection(task["quality_review_flags"]):
        continue
    eligible[task["protein_component_id"]].append(task)

# Sample 30 different protein components without inspecting measured affinity.
# Ordering uses a fixed public seed and a SHA of the component ID.
order=sorted(eligible,key=lambda component:hashlib.sha256(("review_queue_v1|"+component).encode()).hexdigest())
selected=[]
for component in order[:30]:
    # Prefer largest support panel within a selected component for review workload efficiency.
    task=sorted(eligible[component],key=lambda x:(-x["n_unique_compounds"],x["task_id"]))[0]
    assay=assays[(task["protein_seq_sha256"],task["assay_key"])]
    selected.append({
      "task_id":task["task_id"],"protein_component_id":component,
      "protein_sha256":task["protein_seq_sha256"],"assay_key":task["assay_key"],
      "unique_compounds":task["n_unique_compounds"],
      "assay_name":assay["name"],
      "assay_description":assay["description"],
      "has_kd_keyword":assay["signals"].get("kd",False),
      "has_ki_keyword":assay["signals"].get("ki",False),
      "has_ic50_keyword":assay["signals"].get("ic50",False),
      "quality_flags":";".join(task["quality_review_flags"]),
      "construct_verified":"UNREVIEWED","assay_semantics_verified":"UNREVIEWED",
      "records_compatible":"UNREVIEWED","adjudicator_notes":""
    })

output=audit/"bindingdb_assay_manual_review_queue_30.csv"
columns=list(selected[0]) if selected else []
with output.open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=columns)
    writer.writeheader()
    writer.writerows(selected)
report={
"status":"MANUAL_REVIEW_QUEUE_NOT_APPROVED",
"review_rows":len(selected),
"unique_protein_components":len(set(row["protein_component_id"] for row in selected)),
"input_candidate_component_count":len(eligible),
"affinity_values_accessed":False,
"policy":"SHA-selected distinct candidate components; within each choose task with largest chemistry panel, no affinity inspection.",
"output":str(output),
"warning":"Review of experimental papers/construct identity and provenance required before declaring any row valid."}
(audit/"bindingdb_assay_manual_review_queue_30_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
