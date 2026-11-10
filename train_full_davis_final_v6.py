"""DynaCPI v6 final refit: all 442 DAVIS entries × 68 ligands.

The v5 evaluation's model family and each seed's epoch were already frozen on
the v5 validation split before this code. This creates a separate deployable
full-data model; its TRAIN numbers must never be called held-out scores.
The 650M ESM encoder and 2048-bit Morgan descriptors remain FROZEN.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ.setdefault("PYTHONHASHSEED","0")
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from train_full_davis_v5 import (OUT,PREP,SPLIT,RUN,SEEDS,CONFIGS,
    LowRankDynaCPI,normalize_protein,metrics,grid_prediction,sha,write_json,
    loss_values,BATCH,LR,WD)

FINAL=OUT/"final_all442_v6"
NAME="centered_esm_censored"
N_PROTEINS=442
N_CHEMICALS=68

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    if FINAL.exists():raise RuntimeError("Refusing to overwrite final model run")
    previous=json.loads((RUN/"RESULTS.json").read_text(encoding="utf-8"))
    selected=json.loads((RUN/"SELECTION_FROZEN.json").read_text(encoding="utf-8"))
    manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
    if selected["selected_family"]!=NAME or previous["selected_variant"]!=NAME:
        raise RuntimeError("Final chosen v5 family differs: do not tune after test")
    if not selected["frozen_on_validation"]:
        raise RuntimeError("Validation choice not frozen")
    if sha(PREP)!=manifest["prepared_features_sha256"]:
        raise RuntimeError("Dataset hash changed")
    with np.load(PREP,allow_pickle=False) as z:
        pfeat=z["pfeat"];bits=z["bits"];y=z["y"];floor=z["censored"]
        assert z["pfeat"].shape==(N_PROTEINS,1280) and z["bits"].shape==(N_CHEMICALS,2048)
        assert y.shape==(N_PROTEINS,N_CHEMICALS) and floor.shape==y.shape
        original_train=z["train"];original_val=z["val"];original_test=z["test"]
    assert len(set(original_train)|set(original_val)|set(original_test))==N_PROTEINS
    for idxs in (original_train,original_val,original_test):
        assert len(idxs)>0
    all_pi=np.arange(N_PROTEINS,dtype=np.int64)
    norm,mu,sd=normalize_protein(pfeat,all_pi)
    full_prior=y.mean(axis=0).astype(np.float32)
    label_sd=float(y.std())
    if label_sd<=0:raise RuntimeError("Zero label std")
    records=[r for r in previous["per_run"] if r["variant"]==NAME]
    if sorted(r["seed"] for r in records)!=list(SEEDS):
        raise RuntimeError("Missing all selected v5 seeds")
    epochs={int(r["seed"]):int(r["best_epoch"]) for r in records}
    if epochs!={2931:29,2932:24}:
        raise RuntimeError("Unexpected historical validation-selected epochs: "+str(epochs))
    device="cuda"
    px=torch.from_numpy(norm["centered"]).float().to(device)
    cx=torch.from_numpy(bits).float().to(device)
    yy=torch.from_numpy(y).float().to(device)
    ff=torch.from_numpy(floor).bool().to(device)
    cp=torch.from_numpy(full_prior).float().to(device)
    pi=torch.repeat_interleave(torch.arange(N_PROTEINS,device=device),N_CHEMICALS)
    ci=torch.arange(N_CHEMICALS,device=device).repeat(N_PROTEINS)
    assert pi.numel()==ci.numel()==30056
    proto={
      "status":"RUNNING_FINAL_REFIT_ALL_DAVIS_TRAINING_ONLY",
      "source":"canonical public DeepDTA DAVIS (Y, proteins.txt, ligands_can.txt)",
      "raw_y_sha256":manifest["raw_Y_sha256"],
      "prepared_features_sha256":sha(PREP),
      "previous_evaluation_results_sha256":sha(RUN/"RESULTS.json"),
      "frozen_validation_selection_sha256":sha(RUN/"SELECTION_FROZEN.json"),
      "architecture":"LowRankDynaCPI, frozen ESM2 650M mean embeddings + Morgan fp + 32D two tower residual",
      "variant":NAME,
      "seed_epochs":epochs,
      "training_protein_reference_entries":N_PROTEINS,
      "n_unique_protein_sequences":manifest["n_unique_sequences"],
      "n_ligands":N_CHEMICALS,
      "n_training_pairs":N_PROTEINS*N_CHEMICALS,
      "previous_v5_protein_grouped_test_also_used_in_refit":True,
      "not_independently_evaluated":True,
      "frozen_esm_and_rdkit_morgan":True,
      "lr":LR,"weight_decay":WD,"batch_size":BATCH,
      "censored_loss_floor_pKd":5.0,
      "full_data_prior":"Mean pKd per ligand, fit over all 442 protein entries",
      "prediction_clamp_lower_pKd":5.0,
      "software":{"torch":torch.__version__,"cuda":torch.version.cuda,
                  "device":torch.cuda.get_device_name(0),
                  "deterministic_algorithms":torch.are_deterministic_algorithms_enabled()},
      "model_selection":"Previously frozen v5 heldout-validation only; no refit hyperparameter search",
      "limitations":[
        "The final refit has seen ALL original DAVIS labels and has no untouched DAVIS holdout.",
        "The old v5 heldout scores describe the OLDER validation-selected models, not these full-data refits.",
        "All 68 DAVIS chemicals have been seen, arbitrary new chemicals are unsupported experimentally.",
        "Arbitrary unseen protein generalization depends on ESM embeddings and is not independently validated.",
        "Not original published PerceiverCPI or evidence of superiority over it.",
        "The pKd=5 censor marker is a heuristic on benchmark labels, not independently verified."
      ],
      "runs":[]
    }
    FINAL.mkdir(parents=True)
    (FINAL/"checkpoints").mkdir()
    write_json(FINAL/"PROTOCOL.json",proto)
    start_time=time.monotonic()
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model=LowRankDynaCPI().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        history=[]
        for epoch in range(1,epochs[seed]+1):
            model.train()
            perm=np.random.default_rng(seed*1000+epoch).permutation(len(pi))
            loss_vals=[]
            for start in range(0,len(pi),BATCH):
                ids=torch.as_tensor(perm[start:start+BATCH],device=device,dtype=torch.long)
                p=pi[ids];c=ci[ids]
                pred=cp[c]+model(px[p],cx[c])*label_sd
                loss=loss_values(pred,yy[p,c],ff[p,c],label_sd,True)
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.0,error_if_nonfinite=True)
                opt.step()
                loss_vals.append((float(loss.detach()),len(ids)))
            mean_loss=float(sum(x*n for x,n in loss_vals)/len(pi))
            history.append({"epoch":epoch,"standardized_training_objective":mean_loss})
            if epoch==1 or epoch%5==0 or epoch==epochs[seed]:
                print(json.dumps({"seed":seed,"epoch":epoch,"of":epochs[seed],
                    "train_loss":round(mean_loss,6)}),flush=True)
        model.eval()
        pred=grid_prediction(model,all_pi,px,cx,cp,label_sd,device)
        fit_metrics=metrics(pred,y)
        ck={
          "format":"dynacpi_v6_full_442_train_refit",
          "state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
          "seed":seed,"epoch":epochs[seed],"variant":NAME,
          "pfeat_mu":mu.astype(np.float32),"pfeat_scale":sd.astype(np.float32),
          "chemical_prior":full_prior,"label_std":label_sd,
          "training_data_sha256":manifest["raw_Y_sha256"],
          "prepared_features_sha256":sha(PREP),
          "n_train_proteins":N_PROTEINS,
          "n_train_compounds":N_CHEMICALS,
          "n_train_pairs":N_PROTEINS*N_CHEMICALS
        }
        dest=FINAL/"checkpoints"/("final_"+NAME+"_seed"+str(seed)+".pt")
        tmp=dest.with_suffix(".tmp")
        torch.save(ck,tmp);os.replace(tmp,dest)
        proto["runs"].append({"seed":seed,"epochs":epochs[seed],
             "checkpoint":dest.name,"checkpoint_sha256":sha(dest),
             "training_objective_final":history[-1]["standardized_training_objective"],
             "training_fit_macro_rmse":fit_metrics["macro_rmse"],
             "training_fit_centered_mse":fit_metrics["centered_mse"],
             "history":history})
        write_json(FINAL/"PROGRESS.json",proto)
        del model,opt
        torch.cuda.empty_cache()
    proto["status"]="COMPLETE_FINAL_ALL_442_DAVIS_REFIT_TRAINING_ONLY"
    proto["elapsed_seconds"]=round(time.monotonic()-start_time,2)
    write_json(FINAL/"RESULTS.json",proto)
    print(json.dumps({"status":proto["status"],"n_train_pairs":proto["n_training_pairs"],
        "seed_final_training_fit_rmse":[r["training_fit_macro_rmse"] for r in proto["runs"]],
        "elapsed_s":proto["elapsed_seconds"]}),flush=True)

if __name__=="__main__":main()
