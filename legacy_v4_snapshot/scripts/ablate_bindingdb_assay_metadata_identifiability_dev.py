"""How much same-input pKd conflict is conditionally resolved by assay metadata?

Pure in-sample identifiability analysis, not a predictive model.
Retains only previously used train/inner-dev, never the prospective test.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];A=ROOT/"data"/"audit"
OUT=ROOT/"research"/"bindingdb_assay_metadata_identifiability_ablation_dev.json"

def read(path):
    with path.open(encoding="utf8") as f:
        for line in f:yield json.loads(line)

def signature(s):
    s=s.lower()
    modality=("dsf" if ("thermal shift" in s or "dsf" in s)
              else "spr" if ("surface plasmon" in s or "spr" in s)
              else "itc" if ("isothermal titration" in s or "itc" in s)
              else "cell" if "cell" in s
              else "other")
    endpoint=("intrinsic" if "intrinsic" in s
              else "apparent" if "apparent" in s
              else "generic_kd" if ("dissociation constant" in s or " kd " in " "+s+" ")
              else "binding_affinity" if "binding affinity" in s
              else "other")
    return endpoint,modality

def oracle_error(rows, keys):
    groups=defaultdict(list)
    for r in rows:
        key=(r["protein_seq_sha256"],r["compound_sha256"])+tuple(keys(r))
        groups[key].append(float(r["median_pKd"]))
    sse=0.;conflicts=0;duplicate_n=0
    for yy in groups.values():
        y=np.asarray(yy)
        sse+=np.sum((y-y.mean())**2)
        if len(y)>1:duplicate_n+=len(y)
        if len(y)>1 and y.max()-y.min()>1:conflicts+=1
    return {"oracle_empirical_mse":float(sse/len(rows)),
            "oracle_rmse":float(np.sqrt(sse/len(rows))),
            "input_groups":len(groups),"duplicate_input_rows":duplicate_n,
            "duplicated_groups_spread_gt1":conflicts}

def main():
    m=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    train_ids=set(m["train_components"]);dev_ids=set(m["inner_dev_components"])
    assert train_ids.isdisjoint(dev_ids|set(m["test_components"]))
    mapping={}
    for g in read(A/"bindingdb_full_kd_protein_components.jsonl"):
        for s in g["members"]:mapping[s]=g["component_id"]
    tasks={(r["sequence_sha256"],r["assay_key"]) for r in read(A/"bindingdb_full_kd_task_summary.jsonl")
           if r["unique_canonical_compounds"]>=15}
    rows={"train":[],"inner_dev":[]}
    for r in read(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if (r["protein_seq_sha256"],r["assay_key"]) not in tasks or r["range_pKd"]>.1:continue
        c=mapping[r["protein_seq_sha256"]]
        if c in train_ids:rows["train"].append(r)
        elif c in dev_ids:rows["inner_dev"].append(r)
    allowed={(r["protein_seq_sha256"],r["assay_key"])
             for partition in rows.values() for r in partition}
    contexts={}
    for r in read(A/"bindingdb_full_kd_assay_contexts.jsonl"):
        key=(r["sequence_sha256"],r["assay_key"])
        if key in allowed:contexts[key]=r["description"] or ""
    assert len(contexts)==len(allowed)
    out={"status":"EXPLORATORY_IN_SAMPLE_ASSAY_CONDITIONAL_IDENTIFIABILITY_NO_MODEL_PREDICTIONS",
         "partitions":{}}
    for split,data in rows.items():
        meta={id(r):signature(contexts[(r["protein_seq_sha256"],r["assay_key"])]) for r in data}
        both=oracle_error(data,lambda r:())
        ep=oracle_error(data,lambda r:(meta[id(r)][0],))
        mode=oracle_error(data,lambda r:(meta[id(r)][1],))
        sig=oracle_error(data,lambda r:meta[id(r)])
        # Exact assay keys are NOT a deployable cross-target covariate;
        # included as a nearly tautological lower-bound control.
        assay=oracle_error(data,lambda r:(r["assay_key"],))
        source=both["oracle_empirical_mse"]
        out["partitions"][split]={
           "pairs":len(data),
           "endpoint_counts":dict(sorted(__import__("collections").Counter(meta[id(r)][0] for r in data).items())),
           "modality_counts":dict(sorted(__import__("collections").Counter(meta[id(r)][1] for r in data).items())),
           "sequence_compound_only":both,
           "plus_endpoint_descriptor":ep,
           "plus_modality_descriptor":mode,
           "plus_endpoint_and_modality_descriptors":sig,
           "plus_exact_assay_id_control":assay,
           "fraction_same_input_sse_explained_by_endpoint":float(1-ep["oracle_empirical_mse"]/source) if source else None,
           "fraction_same_input_sse_explained_by_endpoint_and_modality":float(1-sig["oracle_empirical_mse"]/source) if source else None}
        print(json.dumps({"partition":split,**out["partitions"][split]},indent=2),flush=True)
    out["limitations"]=[
     "All conditional lower bounds use split labels and are in-sample oracles, not model predictions.",
     "Endpoint/modality categories are heuristic keyword parsing, not verified scientific assay harmonization.",
     "Fewer repeat groups after adding assay metadata mechanically reduce the oracle lower bound.",
     "No reserved test labels or candidate metrics are accessed for selection."]
    OUT.write_text(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
