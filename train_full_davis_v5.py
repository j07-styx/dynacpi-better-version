"""Train the low-rank DynaCPI v5 demo on all public DeepDTA DAVIS entries.

This is a SEPARATE reduced-feature two-tower neural architecture, NOT a
reproduction of published PerceiverCPI and NOT a claim about new chemistry.
Frozen ESM2 features and Morgan fingerprints; no original source-level review.
Protein identity/3-mer-neighbor split; all chemical identities are warm.
Only validation scores select model / epoch, test used once on selected arm.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
OUT=HERE/"full_davis_demo_20261109"
PREP=OUT/"prepared_features.npz"
SPLIT=OUT/"split_manifest.json"
RUN=OUT/"train_v5"
SEEDS=(2931,2932)
CONFIGS=(
    ("raw_esm_mse","raw",False),
    ("centered_esm_mse","centered",False),
    ("centered_esm_censored","centered",True),
)
MAX_EPOCH=45
PATIENCE=10
MIN_EPOCH=8
LR=0.001
WD=0.03
BATCH=1024
HIDDEN_PROT=96
HIDDEN_CHEM=128
BOTTLENECK=32
DROPOUT=.15

class LowRankDynaCPI(nn.Module):
    def __init__(self):
        super().__init__()
        self.protein=nn.Sequential(
            nn.Linear(1280,HIDDEN_PROT),
            nn.LayerNorm(HIDDEN_PROT),nn.GELU(),nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_PROT,BOTTLENECK),nn.LayerNorm(BOTTLENECK))
        self.chemical=nn.Sequential(
            nn.Linear(2048,HIDDEN_CHEM),
            nn.LayerNorm(HIDDEN_CHEM),nn.GELU(),nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_CHEM,BOTTLENECK),nn.LayerNorm(BOTTLENECK))
        self.head=nn.Sequential(
            nn.Linear(BOTTLENECK*4,64),
            nn.GELU(),nn.Dropout(DROPOUT),
            nn.Linear(64,1))
        # Residual starts at zero => chemical baseline at initialization.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,p,c):
        pe=self.protein(p)
        ce=self.chemical(c)
        both=torch.cat((ce,pe,ce*pe,torch.abs(ce-pe)),dim=1)
        return self.head(both).squeeze(1)

def load_local_checkpoint(path):
    """Restricted weights-only load for our own NumPy-containing checkpoints."""
    from numpy.core.multiarray import _reconstruct
    allow=[_reconstruct,np.ndarray,np.dtype,type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(allow):
        return torch.load(path,map_location="cpu",weights_only=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)

def normalize_protein(pfeat,train):
    mu=pfeat[train].mean(0)
    sd=pfeat[train].std(0)
    normstd=np.maximum(sd,.025)
    centered=np.clip((pfeat-mu)/normstd,-6.,6.)
    length=np.maximum(np.linalg.norm(pfeat,axis=1,keepdims=True),1e-8)
    raw=pfeat/length*np.sqrt(pfeat.shape[1])
    return {"raw":raw.astype(np.float32),
            "centered":centered.astype(np.float32)},mu,normstd

def tanimoto(fp1,fp2):
    a=np.asarray(fp1,dtype=np.float64)
    b=np.asarray(fp2,dtype=np.float64)
    dot=a@b.T
    den=a.sum(1)[:,None]+b.sum(1)[None,:]-dot
    return np.divide(dot,den,out=np.zeros_like(dot),where=den>0)

def profile_kernel_coef(bits,train_compound_mean):
    x=np.asarray(train_compound_mean,dtype=np.float64)
    mid=float(x.mean())
    kc=tanimoto(bits,bits)
    ridge=.1 # predeclared fixed prior, not tuned on validation/test
    coef=np.linalg.solve(kc+ridge*np.eye(len(kc)),x-mid)
    return mid,coef

def loss_values(pred,truth,is_floor,label_sd,censored):
    err=pred-truth
    if censored:
        err=torch.where(is_floor,torch.relu(pred-5.0),err)
    return (err/label_sd).square().mean()

def metrics(pred,true):
    p=np.asarray(pred,dtype=np.float64)
    y=np.asarray(true,dtype=np.float64)
    if p.shape!=y.shape or p.ndim!=2:
        raise ValueError(f"metric must be 2d target x compound: {p.shape}")
    error=p-y
    return {
      "macro_rmse":float(np.sqrt(np.mean(error**2,axis=1)).mean()),
      "pooled_rmse":float(np.sqrt(np.mean(error**2))),
      "centered_mse":float(np.mean((error-error.mean(axis=1,keepdims=True))**2)),
      "protein_bias_mse":float(np.mean(error.mean(axis=1)**2)),
      "mean_prediction_std_over_compounds":float(p.std(axis=1).mean()),
      "n_proteins":int(len(y)),"n_pairs":int(y.size)
    }

def grid_prediction(model,pi,px,chem_tensor,prior,sd,device,batch=2048):
    model.eval()
    pids=np.repeat(pi,len(prior))
    dids=np.tile(np.arange(len(prior),dtype=np.int64),len(pi))
    results=[]
    with torch.inference_mode():
        for start in range(0,len(pids),batch):
            p=torch.tensor(pids[start:start+batch],device=device,dtype=torch.long)
            c=torch.tensor(dids[start:start+batch],device=device,dtype=torch.long)
            resid=model(px[p],chem_tensor[c])*sd
            ypred=resid+prior[c]
            # benchmark reports pKd floor; safe to apply same floor to all
            results.append(torch.maximum(ypred,ypred.new_tensor(5.)).cpu().numpy())
    return np.concatenate(results).reshape((len(pi),len(prior))).astype(np.float64)

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for matched v5 GPU training")
    if RUN.exists():
        raise RuntimeError("Completed or partial training directory exists; cannot overwrite")
    manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
    if sha(PREP)!=manifest["prepared_features_sha256"]:
        raise RuntimeError("Prepared features do not match frozen data manifest")
    with np.load(PREP,allow_pickle=False) as z:
        pfeat=z["pfeat"];bits=z["bits"];y=z["y"];floor=z["censored"]
        train=z["train"].astype(int)
        val=z["val"].astype(int);test=z["test"].astype(int)
        names=z["names"].astype(str);smiles=z["smiles"].astype(str)
    assert pfeat.shape==(442,1280)
    assert bits.shape==(68,2048)
    assert y.shape==floor.shape==(442,68)
    assert len(train)==manifest["n_proteins"]["train"]
    assert len(val)==manifest["n_proteins"]["val"]
    assert len(test)==manifest["n_proteins"]["test"]
    assert not set(train)&set(val) and not set(train)&set(test)

    scaled,mu,normstd=normalize_protein(pfeat,train)
    prior_values=y[train].mean(0).astype(np.float64)
    kernel_intercept,kernel_coef=profile_kernel_coef(bits,prior_values)
    label_sd=float(y[train].std())
    assert label_sd>0
    chemical_baselines={"val":metrics(
        np.broadcast_to(prior_values,(len(val),len(prior_values))),y[val])}

    device="cuda"
    chemo=torch.tensor(bits,device=device,dtype=torch.float32)
    prior=torch.tensor(prior_values,device=device,dtype=torch.float32)
    truth=torch.tensor(y,device=device,dtype=torch.float32)
    cens=torch.tensor(floor,device=device,dtype=torch.bool)
    run_protein={key:torch.tensor(arr,device=device,dtype=torch.float32)
                 for key,arr in scaled.items()}
    trp=np.repeat(train,len(prior_values)).astype(np.int64)
    trc=np.tile(np.arange(len(prior_values)),len(train)).astype(np.int64)
    trp=torch.from_numpy(trp).to(device)
    trc=torch.from_numpy(trc).to(device)
    initial={
      "status":"RUNNING_FULL_DAVIS_V5_EXPLORATORY_DEMO",
      "dataset":"public DeepDTA DAVIS, 442 target entries x 68 compounds, pKd labels",
      "prepared_features_sha256":sha(PREP),
      "manifest_sha256":sha(SPLIT),
      "train_proteins":int(len(train)),"val_proteins":int(len(val)),
      "test_proteins":int(len(test)),
      "label_std_train":label_sd,
      "model":"LowRankDynaCPI frozen ESM2 mean 1280 + Morgan 2048 -> low-rank 32 x2 -> interaction MLP",
      "model_comment":"New lightweight research demonstration; NOT exact PerceiverCPI, nor v4.3 repaired sequence-attention model",
      "heads":"chemical TRAIN compound mean anchor + learned interaction residual",
      "feature_norm":"raw normalized or train-only centered/standardized, chosen only using validation",
      "censoring":"Heuristic pKd=5 treated as pKd <=5 for censored arm; nominal pKd RMSE still used for validation",
      "configurations":[{"name":k,"protein_features":mode,"censored_loss":closs} for k,mode,closs in CONFIGS],
      "seeds":list(SEEDS),"max_epochs":MAX_EPOCH,"early_stop_patience":PATIENCE,
      "lr":LR,"weight_decay":WD,"batch_size_pairs":BATCH,
      "validation":"cold protein proxy grouped; all 68 compounds warm, test never used for training selection",
      "seed_selection":"best checkpoint epoch by VALIDATION macro RMSE each seed; choose model family by 2-seed ensemble VALIDATION macro RMSE",
      "test_evaluation":"selected model family evaluated once after validation selection is frozen",
      "important_caveats":[
        "No matched published PerceiverCPI comparison on this split",
        "Proxy 3-mer protein-neighbor group is NOT a curated kinase-family split",
        "All 68 chemical identities are warm: not a double-cold benchmark",
        "Public DAVIS pKd=5 censoring is treated heuristically; not independently verified assay censor flags",
        "External safety, clinical use and molecular rank-order predictions are unsupported"
      ],
      "chemical_prior_validation":chemical_baselines["val"],
      "per_run":[]}
    RUN.mkdir(parents=True)
    (RUN/"checkpoints").mkdir()
    write_json(RUN/"PROTOCOL.json",initial)
    started=time.monotonic()
    for config_name,mode,use_cens in CONFIGS:
        run_prots=run_protein[mode]
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model=LowRankDynaCPI().to(device)
            optim=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
            count=sum(p.numel() for p in model.parameters())
            history=[];best_val=float("inf");best_epoch=None;nonimprove=0
            ckpath=RUN/"checkpoints"/f"{config_name}_seed{seed}.pt"
            for epoch in range(1,MAX_EPOCH+1):
                model.train()
                batches=np.random.default_rng(seed*1000+epoch).permutation(len(trp))
                running=[]
                for start in range(0,len(batches),BATCH):
                    ids=torch.tensor(batches[start:start+BATCH],device=device,dtype=torch.long)
                    pi=trp[ids];ci=trc[ids]
                    prediction=prior[ci]+model(run_prots[pi],chemo[ci])*label_sd
                    loss=loss_values(prediction,truth[pi,ci],cens[pi,ci],label_sd,use_cens)
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),2.0,error_if_nonfinite=True)
                    optim.step()
                    running.append(float(loss.detach().cpu()))
                vpred=grid_prediction(model,val,run_prots,chemo,prior,label_sd,device)
                vm=metrics(vpred,y[val])
                # Conservative early-stop checkpoint selected only from validation.
                better=vm["macro_rmse"]<best_val-0.0002
                if better:
                    best_val=vm["macro_rmse"];best_epoch=epoch;nonimprove=0
                    check={
                      "format":"dynacpi_v5_full_davis_20261109","architecture":"LowRankDynaCPI",
                      "state_dict":{k:v.cpu().clone() for k,v in model.state_dict().items()},
                      "seed":seed,"epoch":epoch,"variant":config_name,
                      "protein_norm":mode,
                      "pfeat_mu":mu.astype(np.float32),
                      "pfeat_scale":normstd.astype(np.float32),
                      "chemical_train_mean":prior_values.astype(np.float32),
                      "chemical_kernel_intercept":kernel_intercept,
                      "chemical_kernel_coef":kernel_coef.astype(np.float32),
                      "training_label_sd":label_sd,
                      "training_data_sha256":manifest["raw_Y_sha256"],
                      "prepared_features_sha256":sha(PREP),
                      "split_manifest_sha256":sha(SPLIT),
                      "model_parameters":count}
                    temp=ckpath.with_suffix(".tmp")
                    torch.save(check,temp)
                    os.replace(temp,ckpath)
                else:
                    nonimprove+=1
                history.append({
                   "epoch":epoch,
                   "standardized_training_objective":float(np.mean(running)),
                   "validation_macro_rmse":vm["macro_rmse"],
                   "validation_pooled_rmse":vm["pooled_rmse"],
                   "validation_centered_mse":vm["centered_mse"],
                   "validation_protein_bias_mse":vm["protein_bias_mse"]})
                if epoch==1 or epoch%5==0 or epoch==best_epoch:
                    print(json.dumps({"variant":config_name,"seed":seed,
                       "epoch":epoch,"train_loss":round(history[-1]["standardized_training_objective"],5),
                       "val_rmse":round(vm["macro_rmse"],6),"best_rmse":round(best_val,6),
                       "best_epoch":best_epoch}),flush=True)
                if epoch>=MIN_EPOCH and nonimprove>=PATIENCE:
                    break
            record={"variant":config_name,"seed":seed,
                    "best_epoch":best_epoch,"best_validation_macro_rmse":best_val,
                    "checkpoint":ckpath.name,"checkpoint_sha256":sha(ckpath),
                    "n_parameters":count,"history":history}
            initial["per_run"].append(record)
            write_json(RUN/"PROGRESS.json",initial)
            del model,optim
            torch.cuda.empty_cache()
    # Validation only: ensemble the two independently best seeds of each
    # config, then choose ONE config; do not query TEST labels before selection.
    scores={}
    for config_name,mode,_ in CONFIGS:
        arr=[]
        for seed in SEEDS:
            cp=RUN/"checkpoints"/f"{config_name}_seed{seed}.pt"
            ck=load_local_checkpoint(cp)
            model=LowRankDynaCPI().to(device);model.load_state_dict(ck["state_dict"])
            arr.append(grid_prediction(model,val,run_protein[mode],chemo,prior,label_sd,device))
            del model
        vals=metrics(np.mean(arr,axis=0),y[val])
        scores[config_name]=vals
    winner=min(scores,key=lambda k:(scores[k]["macro_rmse"],k))
    # Freeze the choice before any reference TEST label access.
    selection={"selected_family":winner,"validation_scores":scores,
               "chemical_prior_val":chemical_baselines["val"],
               "frozen_on_validation":True}
    write_json(RUN/"SELECTION_FROZEN.json",selection)
    mode=next(m for k,m,c in CONFIGS if k==winner)
    test_predictions=[]
    for seed in SEEDS:
        cp=RUN/"checkpoints"/f"{winner}_seed{seed}.pt"
        ck=load_local_checkpoint(cp)
        model=LowRankDynaCPI().to(device);model.load_state_dict(ck["state_dict"])
        test_predictions.append(grid_prediction(model,test,run_protein[mode],chemo,prior,label_sd,device))
        del model
    chosen_test=metrics(np.mean(test_predictions,axis=0),y[test])
    baseline_test=metrics(np.broadcast_to(prior_values,(len(test),len(prior_values))),y[test])
    initial["status"]="COMPLETE_FULL_DAVIS_V5_PROTEIN_GROUPED_DEMO_NOT_PERCEIVERCPI_COMPARISON"
    initial["elapsed_seconds"]=round(time.monotonic()-started,2)
    initial["selected_variant"]=winner
    initial["validation_all_configurations"]=scores
    initial["baseline_test"]=baseline_test
    initial["selected_test"]=chosen_test
    initial["full_dataset_floor_fraction"]=float(floor.mean())
    initial["test_evaluated_once_after_freezing_selection"]=True
    write_json(RUN/"RESULTS.json",initial)
    print(json.dumps({"status":initial["status"],
        "selected":winner,"validation":scores[winner],
        "chemical_test":baseline_test,"selected_test":chosen_test,
        "training_time_seconds":initial["elapsed_seconds"]}),flush=True)

if __name__=="__main__":
    main()
