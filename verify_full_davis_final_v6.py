"""Independent v6 final checkpoint verification on ALL TRAINING labels.

This is not a validation or test benchmark. Never compare its fit numbers to
v5 heldout numbers as if both were generalization metrics.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from train_full_davis_v5 import PREP,SPLIT,RUN,SEEDS,LowRankDynaCPI,load_local_checkpoint,grid_prediction,metrics,sha,write_json
from train_full_davis_final_v6 import FINAL

def main():
    final=json.loads((FINAL/"RESULTS.json").read_text(encoding="utf-8"))
    legacy=json.loads((RUN/"RESULTS.json").read_text(encoding="utf-8"))
    manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
    assert final["status"]=="COMPLETE_FINAL_ALL_442_DAVIS_REFIT_TRAINING_ONLY"
    assert final["raw_y_sha256"]==manifest["raw_Y_sha256"]
    assert sha(PREP)==manifest["prepared_features_sha256"]
    with np.load(PREP,allow_pickle=False) as z:
        pf=z["pfeat"].astype(np.float32)
        bits=z["bits"].astype(np.float32)
        y=z["y"].astype(np.float32)
    assert pf.shape==(442,1280) and bits.shape==(68,2048) and y.shape==(442,68)
    seqs=[]
    per_seed=[]
    expected_epochs={2931:29,2932:24}
    for seed in SEEDS:
        rec=next(r for r in final["runs"] if r["seed"]==seed)
        path=FINAL/"checkpoints"/rec["checkpoint"]
        assert sha(path)==rec["checkpoint_sha256"]
        ck=load_local_checkpoint(path)
        assert ck["format"]=="dynacpi_v6_full_442_train_refit"
        assert ck["seed"]==seed and ck["epoch"]==expected_epochs[seed]
        assert ck["n_train_pairs"]==30056
        assert ck["training_data_sha256"]==manifest["raw_Y_sha256"]
        assert ck["prepared_features_sha256"]==sha(PREP)
        model=LowRankDynaCPI().eval()
        model.load_state_dict(ck["state_dict"],strict=True)
        centered=np.clip((pf-ck["pfeat_mu"])/ck["pfeat_scale"],-6,6).astype(np.float32)
        cx=torch.from_numpy(bits)
        px=torch.from_numpy(centered)
        cp=torch.tensor(ck["chemical_prior"])
        p=grid_prediction(model,np.arange(442),px,cx,cp,ck["label_std"],"cpu")
        recorded=rec["training_fit_macro_rmse"]
        measured=metrics(p,y)["macro_rmse"]
        if abs(recorded-measured)>1e-5:
            raise AssertionError(f"Seed fit mismatch {seed}: {recorded} vs {measured}")
        per_seed.append({"seed":seed,"epoch":ck["epoch"],
             "checkpoint_sha256":sha(path),
             "trained_fit_metrics":metrics(p,y)})
        seqs.append(p)
        print(f"PASS seed {seed} reconstructed train RMSE {measured:.6f}",flush=True)
    ensemble=np.stack(seqs).mean(axis=0)
    base=np.broadcast_to(y.mean(0,keepdims=True),(442,68))
    ans={
        "status":"VERIFIED_ALL_DATA_TRAINING_FIT_NOT_HELDOUT",
        "model":"LowRankDynaCPI v6 final all-DAVIS refit",
        "pairs_train":30056,"proteins_train_entries":442,"unique_protein_sequences":manifest["n_unique_sequences"],"compounds_train":68,
        "n_model_seeds":2,
        "training_data_sha256":manifest["raw_Y_sha256"],
        "all_data_fit_baseline_chemical_mean":metrics(base,y),
        "all_data_fit_v6_ensemble":metrics(ensemble,y),
        "all_data_fit_seed_metrics":per_seed,
        "v5_separate_heldout_evaluation":{
            "tested_earlier_model_not_v6_final_refit":True,
            "training_proteins":legacy["train_proteins"],
            "validation_proteins":legacy["val_proteins"],
            "heldout_proteins":legacy["test_proteins"],
            "macro_rmse_v5_heldout":legacy["selected_test"]["macro_rmse"],
            "macro_rmse_chemical_baseline_v5_heldout":legacy["baseline_test"]["macro_rmse"]
        },
        "strict_warning":"ALL original DAVIS pairs were fitted by the v6 model. These RMSE/MSE values are in-sample ONLY. The v5 test metrics belong to DIFFERENT earlier checkpoints. No independent v6 heldout estimate, matched PerceiverCPI, cold compound or external validation."
    }
    ans["all_data_fit_improvement_relative_baseline_percent"] = float(
        100*(1-ans["all_data_fit_v6_ensemble"]["macro_rmse"]/ans["all_data_fit_baseline_chemical_mean"]["macro_rmse"]))
    dst=FINAL/"METRICS_TRAIN_FIT.json"
    if dst.exists(): raise RuntimeError("Refusing to overwrite existing verification results")
    write_json(dst,ans)
    print(json.dumps({k:v for k,v in ans.items() if k in (
        "status","pairs_train","all_data_fit_baseline_chemical_mean",
        "all_data_fit_v6_ensemble","all_data_fit_improvement_relative_baseline_percent",
        "v5_separate_heldout_evaluation")},indent=2),flush=True)
    print("PASS_FROZEN_V6_ENSEMBLE_TRAIN_FIT_RECONSTRUCTION",flush=True)

if __name__=="__main__":
    torch.set_num_threads(2)
    main()
