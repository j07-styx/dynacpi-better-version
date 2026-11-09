"""Train-only group-held-out cross-validation for nonlinear compound-protein descriptors.

Cross-validation excludes the frozen inner-development and reserved test partitions.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from scripts.train_bindingdb_descriptor_protein_boosting_dev import (
    AMINO, AUDIT, PARAMS, ROOT, molecule_features, protein_features, records,
)

OUT = ROOT / "research" / "bindingdb_descriptor_protein_train_only_component_cv.json"

def summarize(pred, y, groups):
    error = np.square(pred.astype(np.float64) - y.astype(np.float64))
    components = np.unique(groups)
    return {
        "pooled_rmse":float(np.sqrt(error.mean())),
        "component_macro_rmse":float(np.mean([np.sqrt(error[groups == c].mean()) for c in components])),
    }

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.time()
    manifest = json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    train_components = set(manifest["train_components"])
    excluded_components = set(manifest["inner_dev_components"]) | set(manifest["test_components"])
    assert train_components.isdisjoint(excluded_components)
    seq_to_component = {}
    for component in records(AUDIT/"bindingdb_full_kd_protein_components.jsonl"):
        for seqhash in component["members"]:
            seq_to_component[seqhash] = component["component_id"]
    tasks = {(r["sequence_sha256"],r["assay_key"])
             for r in records(AUDIT/"bindingdb_full_kd_task_summary.jsonl")
             if r["unique_canonical_compounds"] >= 15}
    train = []
    for r in records(AUDIT/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if seq_to_component[r["protein_seq_sha256"]] not in train_components:
            continue  # dev/test rows excluded before inspecting any label
        if (r["protein_seq_sha256"],r["assay_key"]) in tasks and r["range_pKd"] <= .1:
            train.append(r)
    assert len(train) == manifest["train_pairs"] == 27118
    needed = {r["protein_seq_sha256"] for r in train}
    seqs = {}
    for r in records(AUDIT/"bindingdb_full_kd_candidates.jsonl"):
        s = r["sequence"]
        key = hashlib.sha256(s.encode()).hexdigest()
        if key in needed:
            seqs[key] = s
            if len(seqs) == len(needed):
                break
    assert len(seqs) == len(needed)
    protein_cache = {key:protein_features(seq) for key,seq in seqs.items()}
    chem_cache = {}
    c = np.zeros((len(train),21),dtype=np.float32)
    p = np.zeros((len(train),21),dtype=np.float32)
    for i,r in enumerate(train):
        smi = r["smiles_isomeric"]
        if smi not in chem_cache:
            chem_cache[smi] = molecule_features(smi)
        c[i] = chem_cache[smi]
        p[i] = protein_cache[r["protein_seq_sha256"]]
    both = np.column_stack([c,p])
    y = np.asarray([r["median_pKd"] for r in train],dtype=np.float32)
    groups = np.asarray([seq_to_component[r["protein_seq_sha256"]] for r in train])
    splitter=GroupShuffleSplit(n_splits=3,test_size=.2,random_state=20261108)
    folds=[]
    modes=["train_mean","chemical_only","chemical_plus_protein","chemical_plus_protein_weighted"]
    for fold, (tr,va) in enumerate(splitter.split(c,y,groups),start=1):
        inner_tr_groups=set(groups[tr]);inner_va_groups=set(groups[va])
        assert inner_tr_groups.isdisjoint(inner_va_groups)
        mu = float(np.mean(y[tr].astype(np.float64)))
        metrics={"train_mean":summarize(np.full(len(va),mu),y[va],groups[va])}
        counts={g:int(np.sum(groups[tr]==g)) for g in inner_tr_groups}
        weights=np.asarray([counts[g]**(-.5) for g in groups[tr]],dtype=np.float64)
        weights /= weights.mean()
        for mode,feature,weight in [
            ("chemical_only",c,None),
            ("chemical_plus_protein",both,None),
            ("chemical_plus_protein_weighted",both,weights)
        ]:
            model=HistGradientBoostingRegressor(**PARAMS)
            model.fit(feature[tr],y[tr],sample_weight=weight)
            metrics[mode]=summarize(model.predict(feature[va]),y[va],groups[va])
        fold_info={"fold":fold,"training_pairs":len(tr),"heldout_pairs":len(va),
                   "training_components":len(inner_tr_groups),"heldout_components":len(inner_va_groups),
                   "heldout_max_component_pair_share":float(max(np.mean(groups[va]==g) for g in inner_va_groups)),
                   "metrics":metrics}
        folds.append(fold_info)
        print(json.dumps(fold_info),flush=True)
    overall={}
    for mode in modes:
        overall[mode]={key:float(np.mean([x["metrics"][mode][key] for x in folds]))
                       for key in ("pooled_rmse","component_macro_rmse")}
    result={
        "status":"TRAIN_ONLY_COMPONENT_HOLDOUT_CV_NO_FROZEN_DEV_OR_RESERVED_TEST_METRICS",
        "original_train_pairs":len(train),"original_train_components":len(set(groups)),
        "cv_splitter":"GroupShuffleSplit(n_splits=3,test_size=.2,random_state=20261108)",
        "fixed_hyperparameters":PARAMS,
        "models":modes,"folds":folds,"mean_fold_metrics":overall,
        "seconds":round(time.time()-started,2),
        "warnings":["Only 3 overlapping group shuffle folds, no independent test",
                    "Provisional assay/construct QC, do not claim external superiority",
                    "Frozen inner development and test were not used as CV outcomes"]
    }
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("FINAL",json.dumps(overall),flush=True)

if __name__=="__main__":
    main()
