"""Training-only group CV selects chemistry/protein ensemble shrinkage; frozen dev read after lock.

No reserved test evaluation. The group-CV folds all lie within the frozen train partition.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from scripts.train_bindingdb_descriptor_protein_boosting_dev import (
    AUDIT, PARAMS, ROOT, molecule_features, protein_features, records
)

OUT = ROOT / "research" / "bindingdb_boosted_protein_shrinkage_cv_selected_dev.json"
GRID_SCALE = (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)
GRID_PROTEIN = (0.0, 0.5, 1.0)
SEED = 20261108

def metric(pred, target, components):
    e = (np.asarray(pred,dtype=np.float64)-np.asarray(target,dtype=np.float64))**2
    unique=np.unique(components)
    return {
        "pooled_rmse": float(np.sqrt(np.mean(e))),
        "component_macro_rmse": float(np.mean([np.sqrt(np.mean(e[components==g])) for g in unique]))
    }

def predict_ensemble(mean, chemistry, joint, shrink, protein_weight):
    correction = (1.0-protein_weight)*(chemistry-mean)+protein_weight*(joint-mean)
    return mean+shrink*correction

def gather():
    manifest=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    tr=set(manifest["train_components"]);dv=set(manifest["inner_dev_components"])
    te=set(manifest["test_components"])
    assert not (tr&dv or tr&te or dv&te)
    mapping={}
    for group in records(AUDIT/"bindingdb_full_kd_protein_components.jsonl"):
        for key in group["members"]:
            mapping[key]=group["component_id"]
    tasks={(x["sequence_sha256"],x["assay_key"])
           for x in records(AUDIT/"bindingdb_full_kd_task_summary.jsonl")
           if x["unique_canonical_compounds"]>=15}
    train=[];dev=[]
    for x in records(AUDIT/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        component=mapping[x["protein_seq_sha256"]]
        if component not in tr and component not in dv:
            continue  # reserved test excluded before consulting labels
        if (x["protein_seq_sha256"],x["assay_key"]) not in tasks or x["range_pKd"]>.1:
            continue
        (train if component in tr else dev).append(x)
    assert len(train)==manifest["train_pairs"]==27118
    assert len(dev)==manifest["inner_dev_pairs"]==7387
    needed={x["protein_seq_sha256"] for x in train+dev}
    seqs={}
    for item in records(AUDIT/"bindingdb_full_kd_candidates.jsonl"):
        seq=item["sequence"];key=hashlib.sha256(seq.encode()).hexdigest()
        if key in needed:
            seqs[key]=seq
            if len(seqs)==len(needed):
                break
    assert len(seqs)==len(needed)
    cache={}
    pcache={key:protein_features(seq) for key,seq in seqs.items()}
    def matrix(rows):
        c=np.empty((len(rows),21),dtype=np.float32)
        p=np.empty((len(rows),21),dtype=np.float32)
        for i,x in enumerate(rows):
            smi=x["smiles_isomeric"]
            if smi not in cache: cache[smi]=molecule_features(smi)
            c[i]=cache[smi]
            p[i]=pcache[x["protein_seq_sha256"]]
        assert np.isfinite(c).all() and np.isfinite(p).all()
        return c,np.column_stack((c,p))
    ctr,jtr=matrix(train)
    cdv,jdv=matrix(dev)
    ytr=np.array([x["median_pKd"] for x in train],dtype=np.float64)
    ydv=np.array([x["median_pKd"] for x in dev],dtype=np.float64)
    gtr=np.array([mapping[x["protein_seq_sha256"]] for x in train])
    gdv=np.array([mapping[x["protein_seq_sha256"]] for x in dev])
    assert set(gtr).isdisjoint(set(gdv))
    return ctr,jtr,cdv,jdv,ytr,ydv,gtr,gdv

def fit_predict(c_tr,j_tr,y_tr,c_va,j_va):
    model_c=HistGradientBoostingRegressor(**PARAMS).fit(c_tr,y_tr)
    model_j=HistGradientBoostingRegressor(**PARAMS).fit(j_tr,y_tr)
    return model_c.predict(c_va),model_j.predict(j_va)

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    t0=time.time()
    ctr,jtr,cdv,jdv,ytr,ydv,gtr,gdv=gather()
    split=GroupShuffleSplit(n_splits=3,test_size=.2,random_state=SEED)
    fold_cache=[]
    print("Loaded frozen train/dev",len(ytr),len(ydv),flush=True)
    for idx,(train_ix,val_ix) in enumerate(split.split(ctr,ytr,gtr),start=1):
        assert set(gtr[train_ix]).isdisjoint(set(gtr[val_ix]))
        c_pred,j_pred=fit_predict(ctr[train_ix],jtr[train_ix],ytr[train_ix],
                                  ctr[val_ix],jtr[val_ix])
        fold_cache.append((float(ytr[train_ix].mean()),c_pred,j_pred,ytr[val_ix],gtr[val_ix]))
        print(json.dumps({"fold":idx,"fit_pairs":len(train_ix),"heldout_pairs":len(val_ix),
                          "heldout_components":len(set(gtr[val_ix]))}),flush=True)
    candidates=[]
    for shrink in GRID_SCALE:
        for protein_weight in GRID_PROTEIN:
            fold_metrics=[metric(predict_ensemble(mu,c,j,shrink,protein_weight),y,groups)
                          for mu,c,j,y,groups in fold_cache]
            macro=float(np.mean([x["component_macro_rmse"] for x in fold_metrics]))
            pooled=float(np.mean([x["pooled_rmse"] for x in fold_metrics]))
            candidates.append({"shrink":shrink,"protein_weight":protein_weight,
                               "mean_fold_pooled_rmse":pooled,
                               "mean_fold_macro_rmse":macro,
                               "balanced_score":(macro+pooled)/2.0})
    def best(name):
        return min(candidates,key=lambda x:(x[name],x["shrink"],x["protein_weight"]))
    selected=best("balanced_score")
    selected_pooled=best("mean_fold_pooled_rmse")
    selected_macro=best("mean_fold_macro_rmse")
    print("TRAIN-ONLY selection",json.dumps({"balanced":selected,"pooled":selected_pooled,
                                            "macro":selected_macro}),flush=True)
    # No development metric is computed until the entire selection rule is frozen above.
    final_chem,final_joint=fit_predict(ctr,jtr,ytr,cdv,jdv)
    mean=float(ytr.mean())
    controls={
      "train_mean":metric(np.full(len(ydv),mean),ydv,gdv),
      "unshrunk_chemistry":metric(final_chem,ydv,gdv),
      "unshrunk_joint":metric(final_joint,ydv,gdv)
    }
    pred=predict_ensemble(mean,final_chem,final_joint,
                          selected["shrink"],selected["protein_weight"])
    validation=metric(pred,ydv,gdv)
    result={
       "status":"EXPLORATORY_TRAIN_ONLY_CV_SELECTED_FROZEN_INNER_DEV_NO_TEST",
       "sample_counts":{"train":len(ytr),"inner_dev":len(ydv)},
       "components":{"train":len(set(gtr)),"inner_dev":len(set(gdv))},
       "model":"two HistGradientBoostingRegressor fits (chemical 21 and joint 42 features); mean-centered ensemble",
       "fixed_model_params":PARAMS,
       "train_only_cv":{"folds":3,"splitter":"GroupShuffleSplit grouped by protein component",
                        "seed":SEED,"scale_grid":GRID_SCALE,"protein_weight_grid":GRID_PROTEIN,
                        "candidates":candidates,"selected_balanced":selected,
                        "selected_pooled":selected_pooled,"selected_macro":selected_macro},
       "frozen_inner_dev":{"selected_balanced":validation,"controls":controls},
       "elapsed_seconds":round(time.time()-t0,2),
       "limitations":[
          "Training-only CV selection, but the external inner-dev split has been inspected repeatedly in prior experiments",
          "Overlapping group shuffle folds have widely different pair counts and component size imbalance",
          "Assay and construct QC not independently approved; no prospective test metric computed",
          "Model may not exceed train-only mean on pooled RMSE"]
    }
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("COMPLETE",json.dumps({"selected":selected,"frozen_dev":validation,
                                "baseline":controls["train_mean"]}),flush=True)

if __name__=="__main__":
    main()
