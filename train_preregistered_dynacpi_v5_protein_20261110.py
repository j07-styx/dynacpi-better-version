"""Strict-protein DAVIS split: FIXED predeclared DynaCPI v5 architecture training.

No test label evaluations or test predictions in this script. Only the already
frozen centered_esm_censored variant, seeds (2931,2932), and original training
settings; checkpoint selection uses held-out VALIDATION proteins only.
Comparisons must remain unreported until official PerceiverCPI is trained on
the SAME preregistered split and both validation choices are frozen.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from train_full_davis_v5 import (
  PREP, LowRankDynaCPI, normalize_protein, metrics, grid_prediction,
  loss_values, sha, write_json, load_local_checkpoint,
  LR,WD,BATCH,MAX_EPOCH,PATIENCE,MIN_EPOCH)
from prepare_preregistered_davis_generalization_20261110 import SOURCE,OUT

SPEC_NAME="protein_strict_20261113"
ARM="centered_esm_censored"
SEEDS=(2931,2932)
FOLDER=OUT/SPEC_NAME
SPEC=FOLDER/"SPLIT_LOCKED.json"
DEST=FOLDER/"dynacpi_v5_fixed_train"

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("GPU required")
    if DEST.exists():raise RuntimeError("Refusing to overwrite a preregistered training run")
    plan=json.loads(SPEC.read_text(encoding="utf-8"))
    assert plan["scenario"]==SPEC_NAME and plan["category"]=="protein"
    assert plan["fixed_v5_architecture"]==ARM and plan["training_seeds"]==list(SEEDS)
    assert plan["test_labels_not_exported"] and plan["selection_locked_before_test"]
    if sha(PREP)!=plan["source_features_sha256"]:
        raise RuntimeError("Original frozen preprocessed feature source changed")
    if sha(SOURCE/"source"/"Y")!=plan["source_Y_sha256"]:
        raise RuntimeError("Original raw DAVIS assay changed")
    if sha(SPEC)!=json.loads((OUT/"SUITE_LOCKED.json").read_text())["scenarios"][0]["split_manifest_sha256"]:
        raise RuntimeError("Preregistered split manifest changed")
    for fn,expected in plan["input_csv_sha256"].items():
        if sha(FOLDER/fn)!=expected:raise RuntimeError("CSV export changed "+fn)
    if plan["input_csv_sha256"]["validation.csv"]!=plan["input_csv_sha256"]["validation_proxy_test.csv"]:
        raise RuntimeError("Upstream validation proxy mismatch")
    ids=plan["indices"]
    tr=np.asarray(ids["protein_train"],dtype=np.int64)
    val=np.asarray(ids["protein_val"],dtype=np.int64)
    test=np.asarray(ids["protein_test"],dtype=np.int64)
    for k in ("compound_train","compound_val","compound_test"):
        assert ids[k]==list(range(68))
    assert len(set(tr)&set(val))==len(set(tr)&set(test))==len(set(val)&set(test))==0
    assert len(tr)+len(val)+len(test)==442
    # Pull only TRAIN + VALIDATION labels into the training process. The original
    # NPZ includes other labels but no y[test] operation is performed or saved.
    with np.load(PREP,allow_pickle=False) as z:
        pfeat=z["pfeat"].astype(np.float32)
        chem_bits=z["bits"].astype(np.float32)
        ytr=z["y"][tr].astype(np.float32)
        yval=z["y"][val].astype(np.float32)
        ctr=z["censored"][tr].astype(np.bool_)
        prepared_proteins=z["pfeat"].shape[0]
    assert pfeat.shape==(442,1280) and chem_bits.shape==(68,2048)
    assert ytr.shape==(len(tr),68) and yval.shape==(len(val),68)
    assert len(tr)*68==plan["train_pairs"]
    assert len(val)*68==plan["validation_pairs"]
    scaled,mu,scale=normalize_protein(pfeat,tr)
    xprot=torch.as_tensor(scaled["centered"],dtype=torch.float32,device="cuda")
    xchem=torch.as_tensor(chem_bits,dtype=torch.float32,device="cuda")
    prior_values=ytr.mean(axis=0).astype(np.float64)
    prior=torch.as_tensor(prior_values,dtype=torch.float32,device="cuda")
    label_sd=float(ytr.std())
    if label_sd<=0:raise RuntimeError("Training labels collapsed")
    truth=torch.as_tensor(ytr,dtype=torch.float32,device="cuda")
    cens=torch.as_tensor(ctr,dtype=torch.bool,device="cuda")
    # Local training protein indexes, never source-dataset test indexes.
    train_local_p=np.repeat(np.arange(len(tr),dtype=np.int64),68)
    train_local_c=np.tile(np.arange(68,dtype=np.int64),len(tr))
    li=torch.as_tensor(train_local_p,dtype=torch.long,device="cuda")
    ci=torch.as_tensor(train_local_c,dtype=torch.long,device="cuda")
    pi=torch.as_tensor(tr,dtype=torch.long,device="cuda")
    DEST.mkdir(parents=True,exist_ok=False)
    (DEST/"checkpoints").mkdir()
    lock={
       "status":"FIXED_PROTOCOL_NO_TEST_CONSULTED",
       "split":SPEC_NAME,"split_manifest_sha256":sha(SPEC),"prepared_features_sha256":sha(PREP),
       "arm":ARM,"seeds":list(SEEDS),"max_epochs":MAX_EPOCH,"early_stopping_patience":PATIENCE,
       "min_epochs":MIN_EPOCH,"optimizer":"AdamW","lr":LR,"weight_decay":WD,"batch_size":BATCH,
       "best_epoch_selection":"minimum validation target-macro RMSE",
       "model_family_selection":"NONE, frozen centered_esm_censored",
       "test_labels_not_used":True,
       "n_training_proteins":len(tr),"n_validation_proteins":len(val),"n_test_proteins_hidden":len(test),
       "n_training_compounds":68}
    write_json(DEST/"TRAINING_PROTOCOL.json",lock)
    results=[]
    starttime=time.monotonic()
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model=LowRankDynaCPI().to("cuda")
        optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        cp=DEST/"checkpoints"/f"{ARM}_seed{seed}.pt"
        best_rmse=float("inf");best_epoch=0;bad=0;history=[]
        for epoch in range(1,MAX_EPOCH+1):
            model.train()
            perms=np.random.default_rng(seed*1000+epoch).permutation(len(li))
            ls=[]
            for s in range(0,len(perms),BATCH):
                idx=torch.as_tensor(perms[s:s+BATCH],dtype=torch.long,device="cuda")
                lp=li[idx];lc=ci[idx]
                pred=prior[lc]+model(xprot[pi[lp]],xchem[lc])*label_sd
                loss=loss_values(pred,truth[lp,lc],cens[lp,lc],label_sd,True)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.0,error_if_nonfinite=True)
                optimizer.step()
                ls.append(float(loss.detach().cpu()))
            val_pred=grid_prediction(model,val,xprot,xchem,prior,label_sd,"cuda")
            m=metrics(val_pred,yval)
            improved=m["macro_rmse"]<best_rmse-0.0002
            if improved:
                best_rmse=m["macro_rmse"];best_epoch=epoch;bad=0
                checkpoint={
                   "format":"dynacpi_v5_fixed_independent_strict_protein_20261110",
                   "state_dict":{k:v.cpu().clone() for k,v in model.state_dict().items()},
                   "seed":seed,"epoch":epoch,"variant":ARM,
                   "split_sha256":sha(SPEC),"prepared_features_sha256":sha(PREP),
                   "chemical_train_mean":prior_values.astype(np.float32),
                   "train_label_sd":label_sd,
                   "pfeat_mu":mu.astype(np.float32),"pfeat_scale":scale.astype(np.float32)}
                tmp=cp.with_suffix(".tmp")
                torch.save(checkpoint,tmp)
                os.replace(tmp,cp)
            else:bad+=1
            history.append({"epoch":epoch,"train_loss":float(np.mean(ls)),
                            "val_macro_rmse":m["macro_rmse"],
                            "val_pooled_rmse":m["pooled_rmse"]})
            if epoch==1 or epoch%5==0 or improved:
                print(json.dumps({"seed":seed,"epoch":epoch,"train_loss":round(float(np.mean(ls)),5),
                       "val_macro_rmse":round(m["macro_rmse"],6),
                       "best_epoch":best_epoch,"best_val_macro_rmse":round(best_rmse,6)}),flush=True)
            if epoch>=MIN_EPOCH and bad>=PATIENCE:break
        results.append({"seed":seed,"best_epoch":best_epoch,
                        "validation_macro_rmse":best_rmse,
                        "checkpoint_sha256":sha(cp),"checkpoint_name":cp.name,
                        "training_epochs":len(history),"history":history})
        write_json(DEST/"TRAIN_PROGRESS.json",{"completed_seeds":[v["seed"] for v in results],
                    "checkpoints_selected_only_on_validation":True})
        del model,optimizer
        torch.cuda.empty_cache()
    # Evaluate VALIDATION ONLY to summarize the frozen 2-seed ensemble; no test.
    chosen=[]
    for seed in SEEDS:
        cp=DEST/"checkpoints"/f"{ARM}_seed{seed}.pt"
        ck=load_local_checkpoint(cp)
        model=LowRankDynaCPI().to("cuda")
        model.load_state_dict(ck["state_dict"],strict=True)
        chosen.append(grid_prediction(model,val,xprot,xchem,prior,label_sd,"cuda"))
        del model
    vpred=np.mean(chosen,axis=0)
    summary=dict(lock,status="VALIDATION_SELECTED_LOCKED_NO_TEST",
                 train_records=results,final_validation=metrics(vpred,yval),
                 elapsed_seconds=float(time.monotonic()-starttime))
    write_json(DEST/"VALIDATION_FROZEN.json",summary)
    print(json.dumps({"status":summary["status"],"split":SPEC_NAME,
                      "n_train_proteins":len(tr),"n_val_proteins":len(val),
                      "frozen_validation_2seed":summary["final_validation"],
                      "test_has_never_been_read":True},indent=2),flush=True)

if __name__=="__main__":main()
