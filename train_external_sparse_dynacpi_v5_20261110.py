"""Frozen DynaCPI external KIBA/Metz SPARSE affinity benchmark training.

- Fixed 1280D ESM2-650M frozen protein embeddings + 2048D Morgan.
- Original LowRankDynaCPI 32D 2-tower ARCHITECTURE, now trained with
  uncensored squared-error for each dataset's own numeric endpoint.
- Missing drug/protein observations are NEVER filled as negative examples.
- Training-compound-only chemical means and fixed top16 Tanimoto prior
  extrapolation for unseen compounds; NO validation/test labels in prior.
- Checkpoints selected only on validation per-target macro RMSE.
- Two fixed seeds 2931/2932, original v5 AdamW/batch/early stop settings.
- NEVER access the sealed test pair labels or choose models on test scores.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from train_full_davis_v5 import (
    LowRankDynaCPI,normalize_protein,sha,write_json,load_local_checkpoint,
    LR,WD,BATCH,MAX_EPOCH,PATIENCE,MIN_EPOCH)
from prepare_external_kiba_metz_frozen_benchmarks_20261110 import OUT,TRAINING_SEEDS

ARM="lowrank_v5_external_sparse_uncensored"
PRIOR_NEIGHBORS=16
PRIOR_SIM_POWER=2
DEVICE="cuda"

def frozen_training_only_prior(bits,train_c,train_y):
    nc=bits.shape[0]
    ct=np.bincount(train_c,minlength=nc)
    sm=np.bincount(train_c,weights=train_y,minlength=nc)
    observed=np.flatnonzero(ct>0)
    if len(observed)<10:raise RuntimeError("Not enough training-only prior compounds")
    means=sm[observed]/ct[observed]
    globalmean=float(train_y.mean())
    fp=torch.as_tensor(bits,device=DEVICE,dtype=torch.float32)
    present=fp[observed]
    overlap=fp@present.T
    den=fp.sum(1,keepdim=True)+present.sum(1).unsqueeze(0)-overlap
    sim=torch.where(den>0,overlap/den,torch.zeros_like(den))
    top,indices=sim.topk(min(PRIOR_NEIGHBORS,len(observed)),dim=1,largest=True)
    weights=top.clamp_min(0).pow(PRIOR_SIM_POWER)
    chemmeans=torch.as_tensor(means,device=DEVICE,dtype=torch.float32)
    est=(weights*chemmeans[indices]).sum(1)/weights.sum(1).clamp_min(1e-8)
    est=torch.where(weights.sum(1)>1e-6,est,est.new_tensor(globalmean))
    prior=est.detach().cpu().numpy().astype(np.float32)
    prior[observed]=means.astype(np.float32)
    if not np.isfinite(prior).all() or not np.allclose(prior[observed],means,atol=1e-5):
        raise RuntimeError("Invalid training-only chemical anchor")
    return prior,observed,means

def metrics(pred,truth,pids):
    pred=np.asarray(pred,dtype=np.float64)
    truth=np.asarray(truth,dtype=np.float64)
    pids=np.asarray(pids,dtype=int)
    if pred.shape!=truth.shape or pred.ndim!=1 or len(pred)==0 or not np.isfinite(pred).all():
        raise RuntimeError("Bad sparse observed pair prediction")
    err2=(pred-truth)**2
    counts=np.bincount(pids,minlength=int(pids.max())+1)
    groups=np.flatnonzero(counts>0)
    means=np.bincount(pids,weights=err2,minlength=len(counts))[groups]/counts[groups]
    return {"macro_rmse":float(np.sqrt(means).mean()),
            "pooled_rmse":float(np.sqrt(err2.mean())),
            "pooled_mse":float(err2.mean()),
            "n_proteins_scored":int(len(groups)),
            "n_compound_protein_pairs":int(len(pred))}

def infer_observed(net,indices,pids,cids,prot,chem,prior,label_sd,batch=2048):
    result=[]
    net.eval()
    with torch.inference_mode():
        for start in range(0,len(indices),batch):
            batch_ids=indices[start:start+batch]
            pi=torch.as_tensor(pids[batch_ids],device=DEVICE,dtype=torch.long)
            ci=torch.as_tensor(cids[batch_ids],device=DEVICE,dtype=torch.long)
            raw=prior[ci]+label_sd*net(prot[pi],chem[ci])
            result.append(raw.float().cpu().numpy())
    if not result:raise RuntimeError("No observed pairs")
    return np.concatenate(result).astype(np.float64)

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--dataset",required=True,choices=["kiba","metz"])
    cli.add_argument("--scenario",required=True,choices=["protein_cold","compound_cold","double_cold"])
    a=cli.parse_args()
    dataset=a.dataset;scenario=a.scenario
    root=OUT/dataset
    specpath=root/scenario/"SPLIT_LOCKED.json"
    s=json.loads(specpath.read_text(encoding="utf-8"))
    audit=json.loads((root/"SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    index=next(i for i in audit["splits"] if i["name"]==scenario)
    if sha(specpath)!=index["manifest_sha256"]:
        raise RuntimeError("FROZEN split hash mismatch")
    if s["dataset"]!=dataset or s["scenario"]!=scenario or not s["test_labels_not_exported"]:
        raise RuntimeError("External heldout not frozen")
    if s["training_seeds"]!=list(TRAINING_SEEDS):
        raise RuntimeError("Unauthorized seed change")
    if s["source_features_sha256"]!=audit["features_sha256"]:
        raise RuntimeError("Source feature metadata changed")
    source=root/"source_features.npz"
    if sha(source)!=audit["features_sha256"]:
        raise RuntimeError("Source NUMPY feature file hash mismatch")
    for fn,digest in s["csv_sha256"].items():
        if sha(root/scenario/fn)!=digest:
            raise RuntimeError("Train/validation CSV hash mismatch")
    train=np.asarray(s["pair_train_row_indices"],dtype=np.int64)
    val=np.asarray(s["pair_val_row_indices"],dtype=np.int64)
    tst=np.asarray(s["pair_test_row_indices"],dtype=np.int64)
    if (set(train)&set(val)) or (set(train)&set(tst)) or (set(val)&set(tst)):
        raise RuntimeError("Train/validation/test overlap")
    dest=root/scenario/"dynacpi_fixed_sparse_train"
    if dest.exists():raise RuntimeError("Do not overwrite completed/partial run")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("Need CUDA for fixed GPU training")
    with np.load(source,allow_pickle=False) as z:
        pfeat=z["protein_features"].astype(np.float32)
        bits=z["compound_morgan_bits"].astype(np.float32)
        pids=z["protein_index"].astype(np.int32)
        cids=z["compound_index"].astype(np.int32)
        # Full data file is immutable and private; ONLY training and
        # validation outcome rows are indexed, not test outcomes.
        truth_tr=z["observed_score"][train].astype(np.float32)
        truth_val=z["observed_score"][val].astype(np.float32)
    if len(pids)!=audit["n_pairs_observed"] or len(train)!=s["n_pairs"]["train"]:
        raise RuntimeError("External sparse label index mismatch")
    if np.any(cids[train]>=bits.shape[0]) or np.any(pids[train]>=pfeat.shape[0]):
        raise RuntimeError("External feature/pair IDs mismatch")
    prot_index=np.unique(pids[train])
    centered,protein_mean,protein_std=normalize_protein(pfeat,prot_index)
    prior_np,known_compounds,known_means=frozen_training_only_prior(bits,cids[train],truth_tr)
    prior=torch.as_tensor(prior_np,device=DEVICE,dtype=torch.float32)
    prot=torch.as_tensor(centered["centered"],device=DEVICE,dtype=torch.float32)
    chem=torch.as_tensor(bits,device=DEVICE,dtype=torch.float32)
    label_sd=float(np.std(truth_tr))
    if label_sd<1e-5:raise RuntimeError("Training endpoint variance too small")
    p_tr=torch.as_tensor(pids[train],dtype=torch.long,device=DEVICE)
    c_tr=torch.as_tensor(cids[train],dtype=torch.long,device=DEVICE)
    y_tr=torch.as_tensor(truth_tr,dtype=torch.float32,device=DEVICE)
    # No test labels on GPU or in inference during training.
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"checkpoints").mkdir()
    protocol={
      "status":"RUNNING_VALIDATION_ONLY",
      "dataset":dataset,"scenario":scenario,"label_units":audit["label_definition"],
      "source_features_sha256":sha(source),
      "frozen_split_sha256":sha(specpath),
      "source_committed_sha":audit["source_repo_sha"],
      "training_seeds":list(TRAINING_SEEDS),"model_name":ARM,
      "architecture":"Original LowRankDynaCPI layer sizes (1280/2048 ->32 x2 -> head)",
      "loss":"standard uncensored normalized squared error, no DAVIS pKd 5 clamp",
      "prior":"training-observed compound exact means, fixed top16 Morgan Tanimoto squared kernel for genuinely unseen ligands",
      "n_known_training_compounds":int(len(known_compounds)),
      "n_train_pairs":int(len(train)),"n_validation_pairs":int(len(val)),
      "sealed_test_pairs_never_read":int(len(tst)),
      "checkpoint_selection":"validation macro-RMSE only",
      "max_epochs":MAX_EPOCH,"min_epochs":MIN_EPOCH,"patience":PATIENCE,
      "batch_size":BATCH,"lr":LR,"weight_decay":WD,
      "missing_affinity_values_never_imputed":True,
      "test_label_access":False,
      "original_perceiver_not_yet_compared":True
    }
    write_json(dest/"TRAINING_PROTOCOL.json",protocol)
    records=[];ts=time.monotonic()
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed);np.random.seed(seed)
        model=LowRankDynaCPI().to(DEVICE)
        optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        checkpoint=dest/"checkpoints"/f"{ARM}_seed{seed}.pt"
        best=float("inf");bestepoch=0;bad=0;history=[]
        for epoch in range(1,MAX_EPOCH+1):
            model.train();losses=[]
            rng=np.random.default_rng(seed*1000+epoch)
            order=rng.permutation(len(train))
            for pos in range(0,len(train),BATCH):
                batch=torch.as_tensor(order[pos:pos+BATCH],device=DEVICE,dtype=torch.long)
                prediction=prior[c_tr[batch]]+label_sd*model(prot[p_tr[batch]],chem[c_tr[batch]])
                l=((prediction-y_tr[batch])/label_sd).square().mean()
                optimizer.zero_grad(set_to_none=True)
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(l.detach().cpu()))
            val_pred=infer_observed(model,val,pids,cids,prot,chem,prior,label_sd)
            m=metrics(val_pred,truth_val,pids[val])
            improved=m["macro_rmse"]<best-0.0002
            if improved:
                best=m["macro_rmse"];bestepoch=epoch;bad=0
                state={
                 "format":"external_uncensored_lowrank_v5",
                 "seed":seed,"epoch":epoch,"variant":ARM,
                 "model_state":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                 "frozen_split_sha256":sha(specpath),
                 "source_features_sha256":sha(source),
                 "label_sd":label_sd,
                 "training_compound_prior":prior_np,
                 "n_training_known_compounds":len(known_compounds),
                 "no_test_access":True
                }
                tmp=checkpoint.with_suffix(".tmp")
                torch.save(state,tmp)
                os.replace(tmp,checkpoint)
            else:bad+=1
            history.append({"epoch":epoch,"loss_train":float(np.mean(losses)),
                "validation_macro_rmse":m["macro_rmse"],
                "validation_pooled_rmse":m["pooled_rmse"]})
            if epoch in (1,5,10) or improved or epoch%10==0:
                print(json.dumps({"dataset":dataset,"scenario":scenario,
                       "seed":seed,"epoch":epoch,"train_loss":round(float(np.mean(losses)),5),
                       "val_macro_rmse":round(m["macro_rmse"],6),
                       "best_epoch":bestepoch}),flush=True)
            if epoch>=MIN_EPOCH and bad>=PATIENCE:break
        records.append({"seed":seed,"best_epoch":bestepoch,
           "best_validation_macro_rmse":best,
           "checkpoint_sha256":sha(checkpoint),"history":history})
        write_json(dest/"TRAIN_PROGRESS.json",
          {"finished_seeds":[x["seed"] for x in records],
           "select_by_validation_only":True,"no_test_access":True})
        del model,optimizer
        torch.cuda.empty_cache()
    val_pred=[]
    for seed in TRAINING_SEEDS:
        path=dest/"checkpoints"/f"{ARM}_seed{seed}.pt"
        ck=load_local_checkpoint(path)
        if ck["frozen_split_sha256"]!=sha(specpath):raise RuntimeError("Checkpoint split corruption")
        net=LowRankDynaCPI().to(DEVICE)
        net.load_state_dict(ck["model_state"],strict=True)
        val_pred.append(infer_observed(net,val,pids,cids,prot,chem,prior,label_sd))
        del net
    ens=np.mean(val_pred,axis=0)
    scores=metrics(ens,truth_val,pids[val])
    prior_scores=metrics(prior_np[cids[val]],truth_val,pids[val])
    report={
      **protocol,
      "status":"VALIDATION_LOCKED_TWO_SEEDS_NO_TEST",
      "seed_records":records,
      "validation_two_seed":scores,
      "train_only_chemical_prior_validation":prior_scores,
      "seconds_elapsed":float(time.monotonic()-ts),
      "no_sealed_test_scores":True
    }
    write_json(dest/"VALIDATION_FROZEN.json",report)
    print(json.dumps({"status":report["status"],
       "dataset":dataset,"scenario":scenario,
       "validation_two_seed":scores,
       "validation_prior_only":prior_scores,
       "sealed_test_evaluation":"NOT_RUN"},indent=2),flush=True)

if __name__=="__main__":main()
