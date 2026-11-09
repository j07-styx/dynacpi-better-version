"""Matched multi-seed DeepDTA-inspired CNN on frozen BALM TRAIN/inner DEV.

Same model, training data, hyperparameters and 100/1000 length conventions
as the prior BALM DeepDTA 10-epoch single-seed implementation. Per-row DEV
predictions are saved by epoch for fully paired group comparisons. No test.
"""
import argparse,hashlib,json,random,time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

from dynacpi.deepdta_reference_dev import DeepDTA
from dynacpi.balm_public_data_dev import ROOT,get_train_dev
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import collapse_training
from scripts.gpu_lease import gpu_lease
from scripts.compare_balm_perceiver_vs_ensemble_dev import row_hash

OUTPUT_DIR=ROOT/"research/balm_deepdta_multiseed_dev_20261108"
MAX_DRUG=100
MAX_PROTEIN=1000
BATCH=128
EVAL_BATCH=128
LR=.0003
WEIGHT_DECAY=.0001
GRAD_NORM=1.

def tokenize(rows,drug_vocab,protein_vocab,dev_groups):
    d=np.zeros((len(rows),MAX_DRUG),dtype=np.int64)
    p=np.zeros((len(rows),MAX_PROTEIN),dtype=np.int64)
    y=np.asarray([r["label"] for r in rows],dtype=np.float32)
    gr=np.asarray([dev_groups.get(r["group"],-1) for r in rows],dtype=np.int64)
    for i,r in enumerate(rows):
        for j,c in enumerate(r["smiles"][:MAX_DRUG]):
            d[i,j]=drug_vocab.get(c,0)
        for j,c in enumerate(r["protein"][:MAX_PROTEIN]):
            p[i,j]=protein_vocab.get(c,0)
    return TensorDataset(
        torch.from_numpy(d),torch.from_numpy(p),torch.from_numpy(y),
        torch.from_numpy(gr))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--epochs",type=int,choices=(10,30),required=True)
    arg=ap.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True,parents=True)
    outfile=OUTPUT_DIR/f"deepdta_seed{arg.seed}_ep{arg.epochs}.json"
    if outfile.exists():raise FileExistsError(outfile)
    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    random.seed(arg.seed)
    data=get_train_dev()
    tr=collapse_training(data["train"])
    dev=data["inner_dev"]
    assert len(tr)==12315 and len(dev)==3891
    assert not set(x["group"] for x in tr)&set(x["group"] for x in dev)
    dv={ch:i+1 for i,ch in enumerate(sorted(set("".join(x["smiles"] for x in tr))))}
    pv={ch:i+1 for i,ch in enumerate(sorted(set("".join(x["protein"] for x in tr))))}
    assert len(dv)<128 and len(pv)<32
    dev_groups={g:i for i,g in enumerate(sorted({x["group"] for x in dev}))}
    assert len(dev_groups)==173
    tr_tensors=tokenize(tr,dv,pv,dev_groups)
    dev_tensors=tokenize(dev,dv,pv,dev_groups)
    train_mean=float(np.mean([r["label"] for r in tr]))
    train_std=float(np.std([r["label"] for r in tr]))
    assert train_std>0
    ydev=np.asarray([r["label"] for r in dev],dtype=np.float64)
    gidx=np.asarray([dev_groups[r["group"]] for r in dev],dtype=np.int64)
    counts=np.bincount(gidx,minlength=len(dev_groups))
    history=[];epoch_preds=[]
    with gpu_lease(ROOT):
        device="cuda" if torch.cuda.is_available() else "cpu"
        assert device=="cuda", "Expected RTX4060 CUDA"
        model=DeepDTA().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        loader=DataLoader(tr_tensors,batch_size=BATCH,shuffle=True,pin_memory=True)
        validation=DataLoader(dev_tensors,batch_size=EVAL_BATCH,pin_memory=True)
        for epoch in range(1,arg.epochs+1):
            t0=time.perf_counter()
            model.train();batchloss=[]
            for drug,protein,y,_ in loader:
                drug=drug.to(device);protein=protein.to(device);y=y.to(device)
                opt.zero_grad(set_to_none=True)
                prediction=model(drug,protein)
                loss=(prediction-(y-train_mean)/train_std).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_NORM)
                opt.step()
                batchloss.append(float(loss.item()))
            model.eval();results=[]
            with torch.inference_mode():
                for drug,protein,_,_ in validation:
                    pred=model(drug.to(device),protein.to(device))
                    results.extend((pred.cpu().numpy()*train_std+train_mean).tolist())
            preds=np.asarray(results,dtype=np.float64)
            assert preds.shape==(3891,) and np.isfinite(preds).all()
            err=np.square(preds-ydev)
            sse=np.bincount(gidx,weights=err,minlength=len(dev_groups))
            scores={
                "epoch":epoch,
                "train_mse_normalized":float(np.mean(batchloss)),
                "pooled_dev_rmse":float(np.sqrt(np.mean(err))),
                "macro_group_dev_rmse":float(np.sqrt(sse/counts).mean()),
                "seconds":round(time.perf_counter()-t0,2),
            }
            history.append(scores)
            epoch_preds.append([round(float(v),5) for v in preds])
            print(json.dumps(scores),flush=True)
    out={
        "status":"BALM_FROZEN_INNER_DEV_MATCHED_DEEPDTA_MULTISEED_NO_TEST",
        "model":"DeepDTA-inspired 1D dual CNN (same implementation as existing BALM DeepDTA reference)",
        "seed":arg.seed,"epochs":arg.epochs,
        "train_rows":len(tr),
        "dev_rows":len(dev),
        "dev_protein_groups":len(dev_groups),
        "dev_row_sha256":row_hash(dev),
        "hyperparameters":{
            "max_smiles_tokens":MAX_DRUG,
            "max_protein_tokens":MAX_PROTEIN,
            "drug_vocab_training_only":len(dv),
            "protein_vocab_training_only":len(pv),
            "batch_size":BATCH,
            "learning_rate":LR,
            "optimizer":"AdamW",
            "weight_decay":WEIGHT_DECAY,
            "clip_grad_norm":GRAD_NORM,
            "train_target_standardization":True,
        },
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "history":history,
        "dev_predictions_by_epoch":epoch_preds,
        "warnings":[
            "Frozen original TRAIN/inner DEV only. No reserved prospective test samples or labels.",
            "DEV already inspected repeatedly. Best DEV epoch is optimistic, not independent selection.",
            "DeepDTA-inspired model is not a full paper reproduction with published hyperparameter optimization.",
            "1000 protein tokens vs PerceiverCPI 500 tokens, different model/compute budgets.",
            "Protein groups use original non-validated 3-mer proxy and not strict scaffold-cold holdout.",
            "BindingDB source assay provenance and contradictory same-input labels unresolved.",
        ],
    }
    outfile.write_text(json.dumps(out)+"\n",encoding="utf8")
    print("SAVED",str(outfile),flush=True)

if __name__=="__main__":main()
