"""Exact same 24-protein, 5-epoch, 3-seed budget for original architecture.

The original model has the same shared-weight initialization before the
repaired model's extra layers. Only previously inspected engineering DAVIS DEV
is scored after fitting. No model is promoted on this diagnostic.
"""
import json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_repaired_full import (
    HERE, OLD, OUT, EPOCHS, LR, WEIGHT_DECAY, SEEDS, load_data, forward,
    score_dev, sha_file, dump,
)
from dynacpi.sequence_model import SequenceFoundation
from gpu_lease import gpu_lease

def main():
    start=time.monotonic()
    torch.set_num_threads(min(2,torch.get_num_threads()))
    data=load_data()
    results=[]
    for seed in SEEDS:
        folder=OUT/("original_seed_"+str(seed))
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/"COMPLETE.json").exists() or (folder/"checkpoint.pt").exists():
            raise RuntimeError("Refusing to overwrite an existing matched-original result")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        model=SequenceFoundation().cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        history=[]
        for epoch in range(1,EPOCHS+1):
            model.train()
            ids=np.random.default_rng(seed*100+epoch).permutation(len(data["train_ids"]))
            loss_sum=0.
            for ix in ids:
                protein_id=data["train_ids"][ix]
                optimizer.zero_grad(set_to_none=True)
                c=model.compound(data["sm"],data["bits"])
                pred=forward(model,c,data["embs"][protein_id])
                loss=(pred-data["train_y"][protein_id]).square().mean()
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite baseline training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                loss_sum+=float(loss.detach())
            row={"epoch":epoch,"train_standardized_mse":loss_sum/24}
            history.append(row)
            dump(folder/"history.json",history)
            print(json.dumps({"arm":"original","seed":seed,**row}),flush=True)
        model.eval()
        with torch.no_grad():
            c=model.compound(data["sm"],data["bits"])
            unit=c/(c.norm(dim=-1,keepdim=True)+1e-9)
            score=unit@unit.T
            cosine=float((score.sum()-score.diagonal().sum())/(54*53))
        checkpoint={"kind":"original_sequence_matched_v4_2_control",
            "status":"ENGINEERING_TRAIN_ONLY_NOT_INDEPENDENTLY_VALIDATED",
            "seed":seed,"epoch":EPOCHS,
            "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
            "label_mean":data["mean"],"label_std":data["std"],
            "row_sha256":data["row_sha256"],"protocol":"same_24train_5epochs_as_repaired_v4_2"}
        temp=folder/"checkpoint.pt.tmp"
        torch.save(checkpoint,temp)
        os.replace(temp,folder/"checkpoint.pt")
        diagnosis=score_dev(model,data)
        item={"seed":seed,"status":"COMPLETE_DEV_ONLY","epoch":EPOCHS,
             "checkpoint_sha256":sha_file(folder/"checkpoint.pt"),
             "train_loss":history,"chemical_embedding_mean_cosine":cosine,
             "historical_dev_diagnostic":diagnosis}
        dump(folder/"COMPLETE.json",item)
        print(json.dumps({"arm":"original","seed":seed,
            "checkpoint_saved":True,"dev_macro_rmse_diagnostic_only":diagnosis["target_macro_rmse"],
            "chemical_mean_cosine":cosine}),flush=True)
        results.append(item)
        del model,optimizer
        torch.cuda.empty_cache()
    matched=[]
    for r in results:
        seed=r["seed"]
        repaired=json.loads((OUT/("seed_"+str(seed))/"COMPLETE.json").read_text(encoding="utf-8"))
        old=r["historical_dev_diagnostic"]["target_macro_rmse"]
        new=repaired["historical_dev_diagnostic"]["target_macro_rmse"]
        matched.append({"seed":seed,"original_rmse":old,"repaired_rmse":new,
                        "delta_repaired_minus_original":new-old,
                        "relative_change_pct":100*(new-old)/old,
                        "original_chemical_cosine":r["chemical_embedding_mean_cosine"],
                        "repaired_chemical_cosine":repaired["chemical_embedding_mean_cosine"]})
    mean=lambda field:float(np.mean([x[field] for x in matched]))
    final={"status":"COMPLETE_MATCHED_TRAINING_BUDGET_REUSED_DEVELOPMENT_PANEL_NOT_SCIENTIFIC",
       "protocol":{"seeds":list(SEEDS),"epochs":EPOCHS,
                  "train_proteins":24,"train_pairs":1296,
                  "validation_proteins":8,"validation_pairs":432,
                  "selection":"fixed 5 epochs prespecified from earlier internal TRAIN split",
                  "initialization":"same seed, identical shared foundation initial parameters, repaired adds bypass parameters",
                  "different_parameter_count":"Repaired has additional layers; not capacity-matched",
                  "data_caveat":"The 8-protein DAVIS development panel has been inspected repeatedly",
                  "no_bindingdb":True,"no_prospective_test":True},
       "per_seed":matched,
       "original_mean_rmse":mean("original_rmse"),
       "repaired_mean_rmse":mean("repaired_rmse"),
       "mean_paired_delta":mean("delta_repaired_minus_original"),
       "relative_mean_rmse_change_pct":100*(mean("repaired_rmse")-mean("original_rmse"))/mean("original_rmse"),
       "elapsed_seconds":round(time.monotonic()-start,2)}
    dump(OUT/"MATCHED_COMPARISON.json",final)
    print(json.dumps({"status":"ALL_MATCHED_CONTROLS_COMPLETE",
        "original_mean":final["original_mean_rmse"],"repaired_mean":final["repaired_mean_rmse"],
        "relative_change_pct":final["relative_mean_rmse_change_pct"],"per_seed":matched}),flush=True)

if __name__=="__main__":
    with gpu_lease(OLD):
        main()
