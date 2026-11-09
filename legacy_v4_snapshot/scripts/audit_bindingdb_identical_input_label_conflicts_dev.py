"""Estimate within-identical-input ambiguity floor on train / old inner-dev only.

This is an in-sample mathematical error floor for deterministic models that
receive only protein sequence and chemical structure. Not a predictive score.
"""
import json, math
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"data"/"audit"
OUT=ROOT/"research"/"bindingdb_provisional_identical_input_label_conflicts_dev.json"

def read(path):
    with path.open(encoding="utf8") as f:
        for l in f:yield json.loads(l)

def summarize(rows):
    pairs=defaultdict(list)
    for r in rows:
        pairs[(r["protein_seq_sha256"],r["compound_sha256"])].append(r)
    repeated=[p for p in pairs.values() if len(p)>1]
    conflicts=[p for p in repeated if max(r["median_pKd"] for r in p)-min(r["median_pKd"] for r in p)>1.]
    sse=0.
    sse_duplicates=0.
    dup_n=0
    for p in pairs.values():
        y=np.asarray([r["median_pKd"] for r in p])
        sq=float(np.sum((y-y.mean())**2))
        sse+=sq
        if len(p)>1:
            sse_duplicates+=sq
            dup_n+=len(p)
    dist=sorted(repeated,key=lambda p:max(r["median_pKd"] for r in p)-min(r["median_pKd"] for r in p),reverse=True)
    examples=[]
    for p in dist[:5]:
        examples.append({"pair_sha256_prefix":p[0]["protein_seq_sha256"][:12]+"-"+p[0]["compound_sha256"][:12],
                         "observed_count":len(p),
                         "pKd_range":max(r["median_pKd"] for r in p)-min(r["median_pKd"] for r in p),
                         "assays":[{"key":r["assay_key"],"pKd":r["median_pKd"],
                                    "reactant_ids":r.get("reactant_ids",[])[:3]} for r in p[:6]]})
    return {
      "candidate_pairs_with_assay_repeats":len(rows),
      "unique_sequence_compound_inputs":len(pairs),
      "identical_input_groups_with_multiple_labels":len(repeated),
      "identical_input_records_in_multi_label_groups":dup_n,
      "fraction_records_in_multi_label_groups":dup_n/len(rows),
      "identical_input_groups_with_cross_assay_spread_over_1_pKd":len(conflicts),
      "unavoidable_in_sample_pooled_rmse_floor_under_sequence_plus_compound_only":math.sqrt(sse/len(rows)),
      "duplicate_group_only_rmse_floor":math.sqrt(sse_duplicates/dup_n) if dup_n else None,
      "top_conflict_examples":examples}
def main():
    m=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    tr=set(m["train_components"]);dv=set(m["inner_dev_components"]);te=set(m["test_components"])
    assert not tr&dv and not tr&te and not dv&te
    comp={}
    for g in read(A/"bindingdb_full_kd_protein_components.jsonl"):
        for s in g["members"]:comp[s]=g["component_id"]
    tasks={(r["sequence_sha256"],r["assay_key"]) for r in read(A/"bindingdb_full_kd_task_summary.jsonl")
           if r["unique_canonical_compounds"]>=15}
    train=[];val=[]
    for r in read(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if (r["protein_seq_sha256"],r["assay_key"]) not in tasks or r["range_pKd"]>.1:
            continue
        c=comp[r["protein_seq_sha256"]]
        if c in tr:train.append(r)
        elif c in dv:val.append(r)
    assert len(train)==m["train_pairs"] and len(val)==m["inner_dev_pairs"]
    report={
     "status":"IN_SAMPLE_DETERMINISTIC_INPUT_CONFLICT_FLOOR_NOT_VALIDATION_PERFORMANCE",
     "train":summarize(train),"inner_dev":summarize(val),
     "interpretation":"Any predictor that sees only protein sequence + canonical compound has the same predicted value for duplicate inputs. Squared-error optimal fitted value for those records is their within-input mean; the remaining SSE is unavoidable even with an oracle that sees split labels.",
     "caveats":["An optimistic IN-SAMPLE lower bound; not a true generalization limit and not model performance.",
                "Assay identity and source conditions are not in the input, so cross-assay conflicts imply target ambiguity.",
                "Prospective test is not included; assay provenance remains provisional."]
    }
    OUT.write_text(json.dumps(report,indent=2))
    for part in ("train","inner_dev"):
        obj=report[part]
        print(json.dumps({"partition":part,**{k:v for k,v in obj.items() if k!="top_conflict_examples"},
                          "top_example":obj["top_conflict_examples"][:1]},indent=2),flush=True)

if __name__=="__main__":main()
