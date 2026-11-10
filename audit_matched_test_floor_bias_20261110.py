"""Post-hoc error audit for the existing one-split DAVIS result (NOT a new test).
Compares official PerceiverCPI both as-is and with same physically meaningful
pKd>=5 output clamp used by DynaCPI v5. Reports weak-binding floor and
non-floor strata separately and paired protein-group descriptive intervals.
No retraining, no model selection, no mutation of existing test results.
"""
import ast
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from train_full_davis_v5 import (
    PREP,SPLIT,RUN,SEEDS,CONFIGS,LowRankDynaCPI,
    normalize_protein,metrics,grid_prediction,sha,load_local_checkpoint)

HERE=Path(__file__).resolve().parent
B=HERE/"matched_perceivercpi_20261110"/"official_matched_v3_HOLDOUT_ONCE"
OUTPUT=B/"POSTHOC_FAIR_FLOOR_AUDIT_20261110.json"

def read_perceiver(expected_smiles):
    with (B/"test_exact_once.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    with (B/"predictions_official.csv").open(newline="",encoding="utf-8") as f:
        predrows=list(csv.DictReader(f))
    if len(rows)!=5508 or len(predrows)!=5508:
        raise RuntimeError("Original official prediction count not 5508")
    seq_lens=[]
    for i,(truth,pr) in enumerate(zip(rows,predrows)):
        compound=pr["smiles"]
        try:
            c=ast.literal_eval(compound)
            if isinstance(c,list) and len(c)==1:compound=c[0]
        except (ValueError,SyntaxError):
            pass
        want=expected_smiles[i%68]
        if compound!=truth["smiles"] or want!=truth["smiles"]:
            raise RuntimeError(f"SMILES alignment error at row {i}")
        seq_lens.append(len(truth["sequence"]))
    return (np.array([float(r["label"]) for r in rows],dtype=np.float64).reshape(81,68),
            np.array([float(r["label"]) for r in predrows],dtype=np.float64).reshape(81,68),
            seq_lens)

def loaded_v5_predictions(pfeat,bits,y,train,test):
    selection=json.loads((RUN/"SELECTION_FROZEN.json").read_text())
    assert selection["frozen_on_validation"]
    arm=selection["selected_family"]
    mode=next(m for name,m,_ in CONFIGS if name==arm)
    scaled,_,_=normalize_protein(pfeat,train)
    device="cuda" if torch.cuda.is_available() else "cpu"
    chem=torch.as_tensor(bits,dtype=torch.float32,device=device)
    prot=torch.as_tensor(scaled[mode],dtype=torch.float32,device=device)
    prior=y[train].mean(axis=0)
    ct=torch.as_tensor(prior,dtype=torch.float32,device=device)
    sd=float(y[train].std())
    preds=[]
    for seed in SEEDS:
        path=RUN/"checkpoints"/f"{arm}_seed{seed}.pt"
        ck=load_local_checkpoint(path)
        if ck["prepared_features_sha256"]!=sha(PREP):raise RuntimeError("V5 feature checksum mismatch")
        assert ck["seed"]==seed and ck["variant"]==arm
        if not np.allclose(ck["chemical_train_mean"],prior,atol=1e-6):
            raise RuntimeError("Chemical prior mismatch")
        net=LowRankDynaCPI().to(device)
        net.load_state_dict(ck["state_dict"],strict=True)
        preds.append(grid_prediction(net,test,prot,chem,ct,sd,device))
    combined=np.mean(preds,axis=0)
    old=json.loads((RUN/"RESULTS.json").read_text())
    actual=metrics(combined,y[test])
    for k in ["macro_rmse","pooled_rmse","centered_mse"]:
        if abs(old["selected_test"][k]-actual[k])>1e-6:
            raise RuntimeError(f"Reconstructed v5 test metrics differ: {k}, computed={actual[k]}, locked={old['selected_test'][k]}, delta={actual[k]-old['selected_test'][k]}")
    return combined

def metrics_for_mask(y,models,mask):
    result={"n_pairs":int(mask.sum()),"floor_fraction":float(mask.mean())}
    for name,p in models.items():
        resid=(p-y)[mask]
        result[name]={"rmse":float(np.sqrt(np.mean(resid*resid))),
                      "mae":float(np.mean(np.abs(resid))),
                      "mse":float(np.mean(resid*resid))}
    return result

def group_bootstrap(y,dyna,perceiver,groups,n=12000):
    """Group bootstrap on the EXACT same already-scored split; posthoc descriptive only."""
    gid=np.unique(groups)
    members=[np.flatnonzero(groups==g) for g in gid]
    rng=np.random.default_rng(20261110)
    allrmse=np.empty((n,2),dtype=np.float64)
    # Each group gets sampled once per draw, with replacement; each entry has 68 compounds.
    for i in range(n):
        choice=rng.integers(0,len(gid),size=len(gid))
        row=np.concatenate([members[k] for k in choice])
        a=np.square(dyna[row]-y[row])
        b=np.square(perceiver[row]-y[row])
        allrmse[i,0]=np.mean(np.sqrt(a.mean(axis=1)))-np.mean(np.sqrt(b.mean(axis=1)))
        allrmse[i,1]=np.sqrt(a.mean())-np.sqrt(b.mean())
    return {"n_3mer_sequence_groups":int(len(gid)),
            "sampled_groups_with_replacement":True,
            "n_bootstrap":int(n),
            "delta_sign":"DynaCPI RMSE minus PerceiverCPI RMSE; negative favors DynaCPI",
            "macro_rmse_delta_95pct_descriptive_CI":list(map(float,np.quantile(allrmse[:,0],[.025,.975]))),
            "pooled_rmse_delta_95pct_descriptive_CI":list(map(float,np.quantile(allrmse[:,1],[.025,.975]))),
            "bootstrap_fraction_dyna_better_macro":float((allrmse[:,0]<0).mean()),
            "bootstrap_fraction_dyna_better_pooled":float((allrmse[:,1]<0).mean())}

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if OUTPUT.exists():raise RuntimeError("Do not overwrite a prior posthoc audit")
    manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
    if sha(PREP)!=manifest["prepared_features_sha256"]:
        raise RuntimeError("Frozen v5 data has changed")
    with np.load(PREP,allow_pickle=False) as z:
        train=z["train"].astype(int);test=z["test"].astype(int)
        y=z["y"].astype(np.float64);pfeat=z["pfeat"]
        bits=z["bits"];floor=z["censored"].astype(bool)
        groups=z["groups"].astype(int);smiles=z["smiles"].astype(str)
    if len(train)!=302 or len(test)!=81 or len(smiles)!=68:
        raise RuntimeError("Split or compounds invalid")
    ytest=y[test]
    ycsv,official,seq_lens=read_perceiver(smiles)
    if not np.allclose(ycsv,ytest,atol=1e-8,rtol=0):
        raise RuntimeError("Official prediction labels do not align v5 test labels")
    dyna=loaded_v5_predictions(pfeat,bits,y,train,test)
    official_clamp=np.maximum(official,5.0)
    models={"DynaCPI_v5":dyna,"PerceiverCPI_original":official,
            "PerceiverCPI_floor_clamped_5":official_clamp}
    mask_floor=floor[test]
    if not np.array_equal(mask_floor,ytest==5.):
        raise RuntimeError("DAVIS floor label boolean mismatch")
    result={
      "status":"POSTHOC_SENSITIVITY_AUDIT_EXISTING_EXPOSED_TEST",
      "not_new_confirmatory_test":True,
      "n_protein_entries":81,"n_pairs":5508,
      "fraction_labels_at_pKd5":float(mask_floor.mean()),
      "n_pKd5":int(mask_floor.sum()),"n_nonfloor":int((~mask_floor).sum()),
      "original_prediction_sha256":sha(B/"predictions_official.csv"),
      "frozen_data_manifest_sha256":sha(SPLIT),
      "clamped_perceiver_pred_fraction":float((official<5.0).mean()),
      "perceiver_min_pred":float(official.min()),
      "perceiver_max_pred":float(official.max()),
      "overall_metrics":{name:metrics(p,ytest) for name,p in models.items()},
      "strata":{
        "floor_pKd_equals_5":metrics_for_mask(ytest,models,mask_floor),
        "non_floor_pKd_above_5":metrics_for_mask(ytest,models,~mask_floor)
      },
      "per_protein_better_than_official":{
        "vs_original":int((np.sqrt(np.mean((dyna-ytest)**2,axis=1))<
                           np.sqrt(np.mean((official-ytest)**2,axis=1))).sum()),
        "vs_clamped":int((np.sqrt(np.mean((dyna-ytest)**2,axis=1))<
                           np.sqrt(np.mean((official_clamp-ytest)**2,axis=1))).sum())
      },
      "bootstraps":{
        "against_original":group_bootstrap(ytest,dyna,official,groups[test]),
        "against_floor_clamped":group_bootstrap(ytest,dyna,official_clamp,groups[test])
      },
      "interpretation":"Post hoc sensitivity audit on the previously exposed same 81-protein test. Treat as diagnostics only, not a new blind hypothesis test. No independent family-disjoint verification; 2 v5 seeds vs 1 Perceiver seed."
    }
    OUTPUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    summary={k:result[k] for k in ["status","n_protein_entries","n_pairs","fraction_labels_at_pKd5","clamped_perceiver_pred_fraction",
                 "overall_metrics","strata","per_protein_better_than_official","bootstraps"]}
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
