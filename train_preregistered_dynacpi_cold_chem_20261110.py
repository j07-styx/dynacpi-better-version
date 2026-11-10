"""Train the frozen DynaCPI v5 architecture on predeclared cold-chemistry DAVIS splits.

NO TEST LABEL ACCESS. Chemically novel compound anchors use ONLY mean pKd
of training compounds on training proteins, extrapolated with fixed predeclared
radius-2 Morgan Tanimoto ridge-0.1 kernel. Validation labels are only used for
checkpoint selection. Original LowRankDynaCPI network, censor loss,
seeds, optimizer, batch size and early-stopping rule are unchanged.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from train_full_davis_v5 import (PREP, LowRankDynaCPI, normalize_protein,
    metrics, loss_values, grid_prediction, sha, write_json, load_local_checkpoint,
    profile_kernel_coef, tanimoto, LR, WD, BATCH, MAX_EPOCH, PATIENCE, MIN_EPOCH)
from prepare_preregistered_davis_generalization_20261110 import SOURCE, OUT

ARMS={"compound_scaffold_20261115":"compound","double_cold_20261116":"double"}
NAME="centered_esm_censored"
SEEDS=(2931,2932)
RIDGE=0.1

def build_chemical_prior(fingerprints,chem_train,training_mean):
    """Exact training-chemical mean anchor + frozen Morgan kernel for novel ligands."""
    fp=fingerprints[chem_train]
    intercept,coef=profile_kernel_coef(fp,training_mean)
    estimate=intercept + tanimoto(fingerprints,fp) @ coef
    estimate=np.asarray(estimate,dtype=np.float64)
    estimate[chem_train]=training_mean
    if not np.isfinite(estimate).all():raise RuntimeError("Invalid chemical extrapolation prior")
    # Train priors must exactly match each training-only compound mean.
    if not np.array_equal(estimate[chem_train],training_mean):
        raise RuntimeError("Training chemical prior changed")
    return estimate,intercept,coef

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenario",required=True,choices=sorted(ARMS))
    args=parser.parse_args()
    name=args.scenario
    scenario_dir=OUT/name
    dest=scenario_dir/"dynacpi_cold_chem_fixed_train"
    if dest.exists():raise RuntimeError("Refusing to overwrite an existing run")
    summary=json.loads((OUT/"SUITE_LOCKED.json").read_text(encoding="utf-8"))
    spec_path=scenario_dir/"SPLIT_LOCKED.json"
    spec=json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["scenario"]!=name or spec["category"]!=ARMS[name]:
        raise RuntimeError("Wrong split category")
    assert spec["fixed_v5_architecture"]==NAME
    assert spec["training_seeds"]==list(SEEDS)
    assert spec["selection_locked_before_test"] and spec["test_labels_not_exported"]
    matched=[x for x in summary["scenarios"] if x["scenario"]==name]
    if len(matched)!=1 or matched[0]["split_manifest_sha256"]!=sha(spec_path):
        raise RuntimeError("Local preregistration hash changed")
    if sha(PREP)!=spec["source_features_sha256"]:
        raise RuntimeError("DAVIS source cached features changed")
    if sha(SOURCE/"source"/"Y")!=spec["source_Y_sha256"]:
        raise RuntimeError("Public affinity matrix changed")
    for fn,digest in spec["input_csv_sha256"].items():
        if sha(scenario_dir/fn)!=digest:
            raise RuntimeError("Training/validation CSV was modified: "+fn)
    idx=spec["indices"]
    ptr=np.array(idx["protein_train"],dtype=int)
    pv=np.array(idx["protein_val"],dtype=int)
    ct=np.array(idx["compound_train"],dtype=int)
    cv=np.array(idx["compound_val"],dtype=int)
    if set(ct)&set(cv):raise RuntimeError("Novel validation ligands must be excluded from train")
    if name.startswith("double") and set(ptr)&set(pv):
        raise RuntimeError("Double cold requires disjoint proteins")
    if len(ptr)*len(ct)!=spec["train_pairs"] or len(pv)*len(cv)!=spec["validation_pairs"]:
        raise RuntimeError("Frozen partition cardinality invalid")
    # Avoid materializing test y into in-memory training arrays.
    with np.load(PREP,allow_pickle=False) as f:
        pfeat=f["pfeat"].astype(np.float32)
        bits=f["bits"].astype(np.float32)
        ytrain=f["y"][np.ix_(ptr,ct)].astype(np.float32)
        yval=f["y"][np.ix_(pv,cv)].astype(np.float32)
        floortrain=f["censored"][np.ix_(ptr,ct)].astype(bool)
    assert pfeat.shape==(442,1280) and bits.shape==(68,2048)
    assert np.isfinite(ytrain).all() and np.isfinite(yval).all()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("A CUDA GPU is required")
    norm,mu,sd=normalize_protein(pfeat,ptr)
    mean_train=ytrain.mean(axis=0,dtype=np.float64)
    prior_all,intercept,coef=build_chemical_prior(bits,ct,mean_train)
    label_sd=float(ytrain.std())
    if label_sd<=0:raise RuntimeError("Training label SD collapsed")
    device="cuda"
    prot=torch.as_tensor(norm["centered"],device=device,dtype=torch.float32)
    chem=torch.as_tensor(bits,device=device,dtype=torch.float32)
    prior=torch.as_tensor(prior_all,device=device,dtype=torch.float32)
    tr_truth=torch.as_tensor(ytrain,device=device,dtype=torch.float32)
    tr_floor=torch.as_tensor(floortrain,device=device,dtype=torch.bool)
    tr_local_p=torch.repeat_interleave(torch.arange(len(ptr),device=device),len(ct))
    tr_local_c=torch.arange(len(ct),device=device).repeat(len(ptr))
    original_p=torch.as_tensor(ptr,device=device,dtype=torch.long)
    original_c=torch.as_tensor(ct,device=device,dtype=torch.long)
    num_pairs=len(ptr)*len(ct)
    assert num_pairs==len(tr_local_p)==len(tr_local_c)
    dest.mkdir(exist_ok=False,parents=True)
    (dest/"checkpoints").mkdir()
    proto={
        "status":"RUNNING_TRAINING_AND_VALIDATION_ONLY",
        "scenario":name,"split_sha256":sha(spec_path),
        "prepared_features_sha256":sha(PREP),
        "original_source_Y_sha256":spec["source_Y_sha256"],
        "architecture":"LowRankDynaCPI centered_esm_censored, original fixed network",
        "fingerprint":"RDKit radius2 chiral 2048 bit Morgan, source cached",
        "prior_train_compounds":"exact training protein mean affinity on each training compound",
        "prior_unseen_compounds":"training-compound-only fixed ridge0.1 Tanimoto extrapolation",
        "prior_ridge_fixed_not_tuned":RIDGE,
        "n_train_compounds":len(ct),"n_val_compounds":len(cv),
        "n_train_proteins":len(ptr),"n_val_proteins":len(pv),
        "training_seeds":list(SEEDS),"max_epochs":MAX_EPOCH,
        "min_epochs":MIN_EPOCH,"patience":PATIENCE,"lr":LR,
        "weight_decay":WD,"batch_size":BATCH,
        "validation_selection":"minimum target-macro RMSE per seed",
        "test_labels_never_loaded":True,
        "test_compound_labels_not_used_for_chemical_prior":True,
        "no_model_selection_on_test":True
    }
    write_json(dest/"TRAINING_PROTOCOL.json",proto)
    records=[]
    started=time.monotonic()
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net=LowRankDynaCPI().to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD)
        best=float("inf");bestepoch=None;bad=0;hist=[]
        out=dest/"checkpoints"/f"{NAME}_seed{seed}.pt"
        for epoch in range(1,MAX_EPOCH+1):
            net.train()
            perm=np.random.default_rng(seed*1000+epoch).permutation(num_pairs)
            train_losses=[]
            for b in range(0,num_pairs,BATCH):
                s=torch.as_tensor(perm[b:b+BATCH],device=device,dtype=torch.long)
                p=tr_local_p[s];c=tr_local_c[s]
                resid=net(prot[original_p[p]],chem[original_c[c]])
                raw=prior[original_c[c]]+resid*label_sd
                loss=loss_values(raw,tr_truth[p,c],tr_floor[p,c],label_sd,True)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),2.0,error_if_nonfinite=True)
                opt.step()
                train_losses.append(float(loss.detach().cpu()))
            # Predict ALL chemical representations, but score only validation
            # pairs whose protein / compound IDs were excluded from TRAIN.
            allchem=grid_prediction(net,pv,prot,chem,prior,label_sd,device)
            vm=metrics(allchem[:,cv],yval)
            improved=vm["macro_rmse"]<best-0.0002
            if improved:
                best=vm["macro_rmse"];bestepoch=epoch;bad=0
                checkpoint={
                    "format":"dynacpi_v5_fixed_cold_chem_preregistered_20261110",
                    "seed":seed,"epoch":epoch,"variant":NAME,
                    "state_dict":{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                    "split_sha256":sha(spec_path),"prepared_features_sha256":sha(PREP),
                    "trained_compound_ids":ct.tolist(),
                    "chemical_train_mean":mean_train.astype(np.float32),
                    "chemical_all_prior":prior_all.astype(np.float32),
                    "protein_train_mu":mu.astype(np.float32),
                    "protein_train_scale":sd.astype(np.float32),
                    "label_sd":label_sd,
                    "tanimoto_kernel_intercept":float(intercept),
                    "tanimoto_kernel_coef":np.asarray(coef,dtype=np.float32),
                    "no_test_label_access":True
                }
                tmp=out.with_suffix(".tmp")
                torch.save(checkpoint,tmp)
                os.replace(tmp,out)
            else:bad+=1
            hist.append({"epoch":epoch,"train_loss":float(np.mean(train_losses)),
                         "validation_macro_rmse":vm["macro_rmse"],
                         "validation_pooled_rmse":vm["pooled_rmse"]})
            if epoch in (1,5,10) or improved or epoch%10==0:
                print(json.dumps({"scenario":name,"seed":seed,"epoch":epoch,
                                  "train_loss":round(float(np.mean(train_losses)),6),
                                  "val_macro_rmse":round(vm["macro_rmse"],6),
                                  "best_epoch":bestepoch}),flush=True)
            if epoch>=MIN_EPOCH and bad>=PATIENCE:break
        records.append({"seed":seed,"best_epoch":bestepoch,
                        "best_validation_macro_rmse":best,"history":hist,
                        "checkpoint_sha256":sha(out)})
        write_json(dest/"TRAIN_PROGRESS.json",{"completed_seeds":[r["seed"] for r in records],
                   "model_selection_validation_only":True})
        del net,opt
        torch.cuda.empty_cache()
    arr=[]
    for seed in SEEDS:
        path=dest/"checkpoints"/f"{NAME}_seed{seed}.pt"
        # Backward-compatible safe load for the first run's int32 NumPy ID array.
        with torch.serialization.safe_globals([type(np.dtype(np.int32))]):
            ck=load_local_checkpoint(path)
        if ck["split_sha256"]!=sha(spec_path):raise RuntimeError("Checkpoint split hash changed")
        net=LowRankDynaCPI().to(device)
        net.load_state_dict(ck["state_dict"],strict=True)
        arr.append(grid_prediction(net,pv,prot,chem,prior,label_sd,device)[:,cv])
        del net
    ensemble=np.mean(arr,axis=0)
    performance=metrics(ensemble,yval)
    baseline=np.broadcast_to(prior_all[cv],ensemble.shape)
    report=dict(proto,
        status="VALIDATION_CHECKPOINTS_FROZEN_NO_TEST",
        completed_seeds=records,
        final_validation=performance,
        anchor_only_validation=metrics(baseline,yval),
        elapsed_seconds=float(time.monotonic()-started),
        test_predictions_not_generated=True)
    write_json(dest/"VALIDATION_FROZEN.json",report)
    print(json.dumps({"status":report["status"],"scenario":name,
        "n_val_pairs":int(yval.size),
        "validation_ensemble":performance,
        "validation_training_only_chemical_anchor":report["anchor_only_validation"]},indent=2),flush=True)

if __name__=="__main__":main()
