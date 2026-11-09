"""Finish an interrupted *completed training* run from six frozen checkpoints.

Selection is performed solely using validation labels. Saves the chosen family
before any test metric computation; test is scored once, after freeze.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from train_full_davis_v5 import (HERE,OUT,PREP,SPLIT,RUN,SEEDS,CONFIGS,
    LowRankDynaCPI,normalize_protein,metrics,grid_prediction,sha,write_json,
    load_local_checkpoint)

def main():
    if (RUN/"RESULTS.json").exists():
        raise RuntimeError("Refusing to overwrite completed results")
    progress=json.loads((RUN/"PROGRESS.json").read_text(encoding="utf-8"))
    manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
    if len(progress["per_run"])!=len(SEEDS)*len(CONFIGS):
        raise RuntimeError("Cannot finalize incomplete model training")
    if progress["prepared_features_sha256"]!=sha(PREP):
        raise RuntimeError("Feature provenance mismatch")
    if progress["manifest_sha256"]!=sha(SPLIT):
        raise RuntimeError("Frozen split provenance mismatch")
    for item in progress["per_run"]:
        path=RUN/"checkpoints"/item["checkpoint"]
        if not path.exists() or sha(path)!=item["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed: "+str(path))
    with np.load(PREP,allow_pickle=False) as z:
        pfeat=z["pfeat"];bits=z["bits"];y=z["y"];floor=z["censored"]
        train=z["train"].astype(int);val=z["val"].astype(int);test=z["test"].astype(int)
    scaled,mu,sd=normalize_protein(pfeat,train)
    prior_values=y[train].mean(0).astype(np.float64)
    label_std=float(y[train].std())
    device="cuda" if torch.cuda.is_available() else "cpu"
    chemo=torch.tensor(bits,device=device,dtype=torch.float32)
    prior=torch.tensor(prior_values,device=device,dtype=torch.float32)
    feats={k:torch.tensor(x,device=device,dtype=torch.float32) for k,x in scaled.items()}
    def predict(arm,seed,pi):
        cpath=RUN/"checkpoints"/f"{arm}_seed{seed}.pt"
        ck=load_local_checkpoint(cpath)
        if ck["variant"]!=arm or ck["seed"]!=seed or ck["prepared_features_sha256"]!=sha(PREP):
            raise RuntimeError("Checkpoint metadata mismatch")
        if not np.allclose(ck["chemical_train_mean"],prior_values,atol=1e-6):
            raise RuntimeError("Saved chemical prior mismatch")
        model=LowRankDynaCPI().to(device)
        model.load_state_dict(ck["state_dict"],strict=True)
        mode=next(m for name,m,_ in CONFIGS if name==arm)
        result=grid_prediction(model,pi,feats[mode],chemo,prior,label_std,device)
        del model
        return result
    val_scores={}
    for arm,_,_ in CONFIGS:
        ensemble=np.mean([predict(arm,seed,val) for seed in SEEDS],axis=0)
        val_scores[arm]=metrics(ensemble,y[val])
        print(json.dumps({"arm":arm,"validation":val_scores[arm]}),flush=True)
    baseline_val=metrics(np.broadcast_to(prior_values,(len(val),len(prior_values))),y[val])
    winner=min(val_scores,key=lambda name:(val_scores[name]["macro_rmse"],name))
    if (RUN/"SELECTION_FROZEN.json").exists():
        saved=json.loads((RUN/"SELECTION_FROZEN.json").read_text())
        if saved.get("selected_family")!=winner:
            raise RuntimeError("Previously frozen selection differs from current validation")
    else:
        write_json(RUN/"SELECTION_FROZEN.json",{
          "frozen_on_validation":True,
          "chosen_variant":winner,
          "selected_family":winner,
          "validation_scores":val_scores,
          "chemical_prior_val":baseline_val,
          "source_progress_sha256":sha(RUN/"PROGRESS.json"),
          "test_not_yet_evaluated":True})
    # TEST labels consulted below, once, after the chosen family has been
    # serialized to a stable selection file.
    ypred=np.mean([predict(winner,seed,test) for seed in SEEDS],axis=0)
    selected_test=metrics(ypred,y[test])
    test_baseline=metrics(np.broadcast_to(prior_values,(len(test),len(prior_values))),y[test])
    progress["status"]="COMPLETE_FULL_DAVIS_V5_PROTEIN_GROUPED_DEMO_NOT_MATCHED_PERCEIVERCPI"
    progress["selected_variant"]=winner
    progress["validation_all_configurations"]=val_scores
    progress["chemical_prior_validation"]=baseline_val
    progress["selected_test"]=selected_test
    progress["baseline_test"]=test_baseline
    progress["test_evaluated_once_after_frozen_validation_selection"]=True
    progress["selection_frozen_sha256"]=sha(RUN/"SELECTION_FROZEN.json")
    write_json(RUN/"RESULTS.json",progress)
    print(json.dumps({"status":progress["status"],"selected":winner,
        "chemical_val":baseline_val,"selected_val":val_scores[winner],
        "chemical_test":test_baseline,"selected_test":selected_test},indent=2),flush=True)

if __name__=="__main__":
    main()
