"""Train candidate DynaCPI v4.2 from scratch, DAVIS engineering TRAIN only.

Reuses existing 24-protein/1296-pair engineering TRAIN cohort, cached ESM-2
and Morgan fingerprints. The final epoch budget=5 is prespecified based on
prior 18/6 training-only split (best epochs 10/5/5; median 5).
All 24 TRAIN proteins are used here; formerly held 6 are NOT a new validation.
The 8 previously inspected DEV proteins are scored once after all checkpoints
are saved, never used for model selection or training.
BindingDB unreviewed labels and prospective TEST labels are NEVER read.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
OLD=Path(r"C:\Users\J0124\dynacpi_v4")
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(OLD/"scripts"))
from gpu_lease import gpu_lease
from repaired_model import RepairedSequenceFoundation

DATA=OLD/"data/engineering_pilot/rows.json"
STAGING=OLD/"data/staging"
ESM=OLD/"data/esm2_t33_650M_layer33_w1000_o128_fp16"
OUT=HERE/"checkpoints"
SEEDS=(2831,2832,2833)
EPOCHS=5
LR=3e-4
WEIGHT_DECAY=1e-4
CHECKS=HERE/"causal_diagnosis_20261109/holdout_results.json"
PROTOCOL="repaired_two_bypasses_v1_24train_5epochs_prespecified"

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)

def load_data():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the inherited MPN implementation.")
    hist=json.loads(CHECKS.read_text(encoding="utf-8"))
    if sorted(hist["seeds"])!=list(SEEDS) or hist["epochs"]!=20:
        raise RuntimeError("Prior internal selection protocol mismatch.")
    selected=[hist["arms"]["both_bypasses_seed"+str(seed)]["selected_epoch"] for seed in SEEDS]
    if sorted(selected)!=[5,5,10]:
        raise RuntimeError("Observed internal selections changed; final budget cannot be inferred.")
    rows=json.loads(DATA.read_text(encoding="utf-8"))
    tr=[r for r in rows if r["split"]=="train"]
    dev=[r for r in rows if r["split"]=="val"]
    train_ids=sorted({r["sequence"] for r in tr})
    val_ids=sorted({r["sequence"] for r in dev})
    if len(tr)!=1296 or len(dev)!=432 or len(train_ids)!=24 or len(val_ids)!=8 or set(train_ids)&set(val_ids):
        raise RuntimeError("Unexpected frozen engineering data split; abort.")
    cids=sorted({r["compound"] for r in rows})
    if len(cids)!=54:raise RuntimeError("Expected 54 known compounds.")
    smi=json.loads((STAGING/"compounds.json").read_text(encoding="utf-8"))
    sm=[smi[c] for c in cids]
    fpz=np.load(STAGING/"morgan_radius2_2048_chiral.npz")
    fp=dict(zip(fpz["ids"].astype(str),fpz["bits"]))
    bits=torch.as_tensor(np.stack([fp[c] for c in cids]),device="cuda",dtype=torch.float32)
    def targets_for(splitids,role):
        scores={}
        for p in splitids:
            rr={r["compound"]:float(r["y"]) for r in rows if r["split"]==role and r["sequence"]==p}
            if set(rr)!=set(cids):raise RuntimeError("Protein lacks full 54-compound panel.")
            scores[p]=np.array([rr[c] for c in cids],dtype=np.float64)
        return scores
    train=targets_for(train_ids,"train");val=targets_for(val_ids,"val")
    mean=float(np.mean(np.stack(list(train.values()))))
    std=float(np.std(np.stack(list(train.values()))))
    if std<=0 or not math.isfinite(std):raise RuntimeError("Invalid train label scale.")
    all_ids=sorted(set(train_ids)|set(val_ids))
    embs={}
    for p in all_ids:
        obj=torch.load(ESM/(p+".pt"),map_location="cpu",weights_only=True)
        if obj.get("sequence_hash")!=p or obj["embeddings"].ndim!=2 or obj["embeddings"].shape[1]!=1280:
            raise RuntimeError("ESM provenance check failed.")
        embs[p]=obj["embeddings"].float().cuda()
    train_y={p:torch.as_tensor((v-mean)/std,device="cuda",dtype=torch.float32) for p,v in train.items()}
    return {
      "train":train,"dev":val,"train_y":train_y,"embs":embs,"sm":sm,"bits":bits,
      "train_ids":train_ids,"dev_ids":val_ids,"mean":mean,"std":std,
      "row_sha256":sha_file(DATA),
      "fingerprint_sha256":sha_file(STAGING/"morgan_radius2_2048_chiral.npz"),
      "repair_source_sha256":sha_file(HERE/"repaired_model.py"),
      "selected_epoch_prior":selected
    }

def forward(model,c,emb):
    token=model.sequence(emb.float())
    mask=torch.ones((len(c),len(token)),device="cuda",dtype=torch.bool)
    return model.predict_tokens(c,token[None].expand(len(c),-1,-1),mask)

def score_dev(model,info):
    """Previously inspected engineering DEV, diagnostic ONLY; never selecting epochs."""
    model.eval()
    report={}
    with torch.no_grad():
        c=model.compound(info["sm"],info["bits"])
        for p in info["dev_ids"]:
            prediction=(forward(model,c,info["embs"][p])*info["std"]+info["mean"]).cpu().numpy()
            truth=info["dev"][p]
            report[p]={
             "rmse":float(np.sqrt(np.mean((prediction-truth)**2))),
             "pred_sd":float(np.std(prediction)),
             "true_sd":float(np.std(truth)),
             "n":54
            }
    return {"protein_count":8,"target_macro_rmse":float(np.mean([r["rmse"] for r in report.values()])),
        "mean_pred_sd":float(np.mean([r["pred_sd"] for r in report.values()])),
        "per_protein":report,
        "status":"INSPECTED_DAVIS_DEV_DIAGNOSTIC_ONLY"}

def main():
    start=time.monotonic()
    torch.set_num_threads(min(2,torch.get_num_threads()))
    info=load_data()
    OUT.mkdir(parents=True,exist_ok=True)
    meta={"status":"ENGINEERING_FULL_TRAIN_ONLY_NO_INDEPENDENT_SCIENCE","protocol":PROTOCOL,
       "seeds":list(SEEDS),"epochs":EPOCHS,"lr":LR,"weight_decay":WEIGHT_DECAY,
       "training_pairs":1296,"train_proteins":24,"previously_inspected_dev_pairs":432,
       "previously_inspected_dev_proteins":8,"known_compounds":54,
       "label_mean":info["mean"],"label_std":info["std"],
       "row_sha256":info["row_sha256"],"fingerprint_sha256":info["fingerprint_sha256"],
       "repair_source_sha256":info["repair_source_sha256"],
       "epoch_choice":"median previously observed training-only 18/6 best epochs (5,5,10)",
       "frozen_esm":True,"model":"RepairedSequenceFoundation with two direct feature bypasses",
       "limitations":[
         "Prior internal validation reused historical 24 TRAIN proteins and the same 54 compounds.",
         "The 8 DEV proteins have been inspected in previous experiments: not an independent test.",
         "Unreviewed BindingDB and any reserved prospective test are excluded.",
         "No ligand-pose geometry is used in this repaired SEQUENCE-only model.",
         "Two-path repair adds parameters and is not a capacity-matched baseline."
       ]}
    dump(OUT/"PROTOCOL.json",meta)
    saved=[]
    for seed in SEEDS:
        path=OUT/("seed_"+str(seed))
        path.mkdir(parents=True,exist_ok=True)
        if (path/"checkpoint.pt").exists() or (path/"COMPLETE.json").exists():
            raise RuntimeError("Refusing to overwrite an existing completed/incomplete checkpoint: "+str(path))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model=RepairedSequenceFoundation().cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        hist=[]
        for epoch in range(1,EPOCHS+1):
            model.train()
            ids=np.random.default_rng(seed*100+epoch).permutation(len(info["train_ids"]))
            total=0.
            for index in ids:
                p=info["train_ids"][index]
                optimizer.zero_grad(set_to_none=True)
                c=model.compound(info["sm"],info["bits"])
                pred=forward(model,c,info["embs"][p])
                loss=(pred-info["train_y"][p]).square().mean()
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                total+=float(loss.detach())
            rec={"epoch":epoch,"train_standardized_mse":total/24,
                 "train_elapsed_seconds":round(time.monotonic()-start,2)}
            hist.append(rec)
            dump(path/"history.json",hist)
            print(json.dumps({"seed":seed,**rec}),flush=True)
        model.eval()
        with torch.no_grad():
            c=model.compound(info["sm"],info["bits"])
            normed=c/(c.norm(dim=-1,keepdim=True)+1e-9)
            cos=(normed@normed.T)
            chemical_cos=float((cos.sum()-cos.diagonal().sum())/(54*53))
        checkpoint={
         "kind":"repaired_sequence_v4_2",
         "status":"ENGINEERING_TRAIN_ONLY_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED",
         "seed":seed,"epoch":EPOCHS,
         "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
         "label_mean":info["mean"],"label_std":info["std"],
         "row_sha256":info["row_sha256"],"fingerprint_sha256":info["fingerprint_sha256"],
         "repair_source_sha256":info["repair_source_sha256"],
         "architecture":"two_path_v1",
         "protocol":PROTOCOL
        }
        tmp=path/"checkpoint.pt.tmp"
        torch.save(checkpoint,tmp)
        os.replace(tmp,path/"checkpoint.pt")
        dev=score_dev(model,info)
        result={"status":checkpoint["status"],"seed":seed,"epoch":EPOCHS,
          "checkpoint_sha256":sha_file(path/"checkpoint.pt"),
          "training_loss":hist,"chemical_embedding_mean_cosine":chemical_cos,
          "historical_dev_diagnostic":dev,
          "limits":meta["limitations"]}
        dump(path/"COMPLETE.json",result)
        print(json.dumps({"seed":seed,"status":"CHECKPOINT_SAVED",
            "dev_macro_rmse_diagnostic_only":dev["target_macro_rmse"],
            "chemical_mean_cosine":chemical_cos}),flush=True)
        saved.append({"seed":seed,"checkpoint_sha256":result["checkpoint_sha256"],
           "historical_dev_macro_rmse":dev["target_macro_rmse"],
           "chemical_embedding_mean_cosine":chemical_cos})
        del model,optimizer
        torch.cuda.empty_cache()
    meta["completed"]=saved
    meta["elapsed_seconds"]=round(time.monotonic()-start,2)
    dump(OUT/"SUMMARY.json",meta)
    print(json.dumps({"status":"ALL_TRAINING_COMPLETE","candidate_count":len(saved),
        "elapsed_seconds":meta["elapsed_seconds"],"candidates":saved}),flush=True)

if __name__=="__main__":
    with gpu_lease(OLD):
        main()
