"""Descriptive assay-metadata risk counts for high-support Kd candidate tasks.
Does not automatically approve or reject assay equivalence.
"""
from pathlib import Path
import json
from collections import Counter
root=Path(__file__).resolve().parents[1]
data=root/"data/audit"
tasks={}
for ln in (data/"bindingdb_full_kd_task_summary.jsonl").open(encoding="utf-8"):
    t=json.loads(ln)
    tasks[(t["sequence_sha256"],t["assay_key"])]=t
checks={}
examples=[]
for k in (15,30):
    c=Counter()
    for ln in (data/"bindingdb_full_kd_assay_contexts.jsonl").open(encoding="utf-8"):
        a=json.loads(ln)
        t=tasks.get((a["sequence_sha256"],a["assay_key"]))
        if not t or t["unique_canonical_compounds"]<k:
            continue
        c["tasks"]+=1
        c["unique_seq_count_not_calculated"]+=0
        c["records"]+=t["raw_records"]
        if a["name"]:c["name_present"]+=1
        if a["description"]:c["description_present"]+=1
        signals=a["signals"]
        for term,on in signals.items():
            if on:c["term_"+term]+=1
        if not signals["kd"]:
            c["without_explicit_Kd_wording"]+=1
            if k==15 and len(examples)<10:
                examples.append({"assay_key":a["assay_key"],
                                 "canonical_compounds":t["unique_canonical_compounds"],
                                 "name":a["name"][:160], "description_excerpt":a["description"][:300]})
        if signals["ic50"] or signals["ki"]:
            c["Ki_or_IC50_wording"]+=1
        if a["article_doi_count"]>1:c["multi_article_doi"]+=1
        if t["within_group_duplicate_compounds_gt_0_1_pKd"]>0:
            c["tasks_with_discordant_replicates"]+=1
    checks[f"ge{k}"]=dict(c)
report={
   "scope":"Full BindingDB candidate exact-looking-Kd records and published assay text",
   "status":"SCREENING ONLY; text is not a substitute for manual endpoint or assay verification",
   "cohort_flags":checks,
   "sample_missing_Kd_wording":examples,
   "interpretation":"Kd is present in a numerical source column. No Kd word in a description is not proof an observation is invalid; flag for provenance review."
}
(data/"bindingdb_full_kd_assay_risks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({"cohort_flags":checks,"sample_missing_Kd_wording":examples[:4]},indent=2),flush=True)
