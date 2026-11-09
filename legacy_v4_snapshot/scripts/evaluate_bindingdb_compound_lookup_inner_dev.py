"""Train-only compound lookup baseline for provisional BindingDB inner-dev.

No model training, test access, or target-derived validation features.
"""
import json,math,statistics
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"data/audit"
def read(p):
    with p.open(encoding="utf-8") as f:
        for line in f:yield json.loads(line)
def main():
    manifest=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    groups={r["member"]:r["component_id"] for r in []}
    for r in read(A/"bindingdb_full_kd_protein_components.jsonl"):
        for s in r["members"]:groups[s]=r["component_id"]
    tasks={(x["sequence_sha256"],x["assay_key"]) for x in read(A/"bindingdb_full_kd_task_summary.jsonl") if x["unique_canonical_compounds"]>=15}
    tr=set(manifest["train_components"]);dev=set(manifest["inner_dev_components"])
    train=[];valid=[]
    for x in read(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if (x["protein_seq_sha256"],x["assay_key"]) not in tasks or x["range_pKd"]>.1:continue
        group=groups[x["protein_seq_sha256"]]
        if group in tr:train.append(x)
        elif group in dev:valid.append(x)
    assert len(train)==manifest["train_pairs"] and len(valid)==manifest["inner_dev_pairs"]
    by=defaultdict(list)
    for x in train:by[x["smiles_isomeric"]].append(x["median_pKd"])
    global_mean=statistics.mean(x["median_pKd"] for x in train)
    results={}
    for shrink in (0,1,5,20,100):
        sq=0;unseen=0
        for x in valid:
            vals=by.get(x["smiles_isomeric"],[])
            if not vals:unseen+=1
            pred=(sum(vals)+shrink*global_mean)/(len(vals)+shrink) if len(vals)+shrink else global_mean
            sq+=(pred-x["median_pKd"])**2
        results[str(shrink)]={"rmse":math.sqrt(sq/len(valid)),"validation_pairs_with_unseen_compound":unseen}
    out={"status":"PROVISIONAL_INNER_DEV_ONLY_NOT_TEST","train_pairs":len(train),"inner_dev_pairs":len(valid),"unique_train_compounds":len(by),"train_only_compound_mean_shrinkage":results,"warning":"Development hyperparameter comparison only; chemical IDs may not be independently verified; no prospective test evaluated."}
    dest=ROOT/"research/bindingdb_compound_lookup_inner_dev_dev.json"
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()

