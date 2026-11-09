"""Read-only verification of selected full DAVIS v5 checkpoint performance.

No model selection: selection is frozen on validation in SELECTION_FROZEN.json.
Reports descriptive protein-group bootstrap uncertainty, not family-independent CI.
"""
import json
import numpy as np
import torch
from train_full_davis_v5 import (OUT,PREP,SPLIT,RUN,SEEDS,CONFIGS,
   LowRankDynaCPI,normalize_protein,metrics,grid_prediction,sha,
   write_json,load_local_checkpoint)

def main():
    if (RUN/"VERIFICATION.json").exists():
        raise RuntimeError("Will not overwrite verification")
    result=json.loads((RUN/"RESULTS.json").read_text(encoding="utf-8"))
    selection=json.loads((RUN/"SELECTION_FROZEN.json").read_text(encoding="utf-8"))
    assert selection["frozen_on_validation"] and selection["selected_family"]==result["selected_variant"]
    assert result["test_evaluated_once_after_frozen_validation_selection"]
    with np.load(PREP,allow_pickle=False) as z:
        pfeat=z["pfeat"];bits=z["bits"];y=z["y"]
        train=z["train"].astype(int);val=z["val"].astype(int);test=z["test"].astype(int);groups=z["groups"].astype(int)
    scaled,_,_=normalize_protein(pfeat,train)
    mode=next(m for name,m,_ in CONFIGS if name==selection["selected_family"])
    device="cuda" if torch.cuda.is_available() else "cpu"
    feats=torch.tensor(scaled[mode],device=device,dtype=torch.float32)
    chem=torch.tensor(bits,device=device,dtype=torch.float32)
    prior_values=y[train].mean(0)
    prior=torch.tensor(prior_values,device=device,dtype=torch.float32)
    label_sd=float(y[train].std())
    pr=[]
    for seed in SEEDS:
        path=RUN/"checkpoints"/f"{result['selected_variant']}_seed{seed}.pt"
        ck=load_local_checkpoint(path)
        net=LowRankDynaCPI().to(device)
        net.load_state_dict(ck["state_dict"],strict=True)
        pr.append(grid_prediction(net,test,feats,chem,prior,label_sd,device))
        del net
    preds=np.mean(pr,axis=0)
    baseline=np.broadcast_to(prior_values,(len(test),68))
    neural_metrics=metrics(preds,y[test])
    baseline_metrics=metrics(baseline,y[test])
    for name,computed in (("selected_test",neural_metrics),("baseline_test",baseline_metrics)):
        saved=result[name]
        for key in ("macro_rmse","pooled_rmse","centered_mse","protein_bias_mse"):
            assert abs(computed[key]-saved[key])<1e-9,(name,key,computed[key],saved[key])
    b=np.sqrt(np.mean((baseline-y[test])**2,axis=1))
    m=np.sqrt(np.mean((preds-y[test])**2,axis=1))
    deltas=m-b
    percent=(baseline_metrics["macro_rmse"]-neural_metrics["macro_rmse"])/baseline_metrics["macro_rmse"]*100
    rng=np.random.default_rng(260911)
    # Multiple reference target entries can share an identical sequence/group.
    ids=np.unique(groups[test])
    group_deltas=np.array([deltas[groups[test]==g].mean() for g in ids])
    samples=group_deltas[rng.integers(0,len(ids),size=(10000,len(ids)))].mean(1)
    ci=np.quantile(samples,[.025,.975])
    verification={
      "status":"VERIFIED_NO_RETRAINING",
      "selected_variant":result["selected_variant"],
      "selected_test":neural_metrics,"chemical_prior_test":baseline_metrics,
      "test_macro_rmse_relative_reduction_percent":float(percent),
      "heldout_test_proteins":int(len(test)),
      "heldout_test_sequence_neighbor_groups":int(len(ids)),
      "n_heldout_targets_better_than_baseline":int(np.sum(deltas<0)),
      "n_heldout_targets_worse_than_baseline":int(np.sum(deltas>0)),
      "descriptive_group_bootstrap_95pct_mean_rmse_delta":[float(v) for v in ci],
      "note":"Group bootstrap is a descriptive check on one fixed, model-selected DAVIS split, not independent homology-family verification or a PerceiverCPI superiority claim.",
      "frozen_selection_sha256":sha(RUN/"SELECTION_FROZEN.json"),
      "train_results_sha256":sha(RUN/"RESULTS.json"),
      "checkpoint_sha256":[sha(RUN/"checkpoints"/f"{result['selected_variant']}_seed{seed}.pt") for seed in SEEDS]
    }
    write_json(RUN/"VERIFICATION.json",verification)
    print(json.dumps(verification,indent=2),flush=True)

if __name__=="__main__":
    main()
