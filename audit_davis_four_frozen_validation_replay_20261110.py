"""Replay ALL four frozen DynaCPI DAVIS VAL metrics without opening test labels.

This is a fail-closed readiness audit BEFORE the queued 4-scenario blinded
two-seed official PerceiverCPI comparison. Not training. No test predictions,
test labels, checkpoint selection or changes to predeclared data splits.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from prepare_preregistered_davis_generalization_20261110 import OUT
from train_full_davis_v5 import (
    PREP,LowRankDynaCPI,normalize_protein,grid_prediction,metrics,sha,
    load_local_checkpoint)
from train_preregistered_dynacpi_cold_chem_20261110 import (
    build_chemical_prior,NAME)
CASES=(
 ("protein_strict_20261113","protein","dynacpi_v5_fixed_train"),
 ("protein_strict_20261114","protein","dynacpi_v5_fixed_train"),
 ("compound_scaffold_20261115","compound","dynacpi_cold_chem_fixed_train"),
 ("double_cold_20261116","double","dynacpi_cold_chem_fixed_train")
)
SEEDS=(2931,2932)
def run_one(name,category,branch,device):
    root=OUT/name
    lock=root/"SPLIT_LOCKED.json"
    spec=json.loads(lock.read_text(encoding="utf-8"))
    suite=json.loads((OUT/"SUITE_LOCKED.json").read_text(encoding="utf-8"))
    declared=next(r for r in suite["scenarios"] if r["scenario"]==name)
    if spec["category"]!=category or sha(lock)!=declared["split_manifest_sha256"]:
        raise RuntimeError("Locked scenario manifest changed")
    if sha(PREP)!=spec["source_features_sha256"]:
        raise RuntimeError("Features hash changed")
    ids=spec["indices"]
    pt=np.asarray(ids["protein_train"],dtype=int)
    pv=np.asarray(ids["protein_val"],dtype=int)
    ct=np.asarray(ids["compound_train"],dtype=int)
    cv=np.asarray(ids["compound_val"],dtype=int)
    # Only train and validation labels indexed. Test labels are never touched.
    with np.load(PREP,allow_pickle=False) as z:
        pfeat=z["pfeat"].astype(np.float32)
        bits=z["bits"].astype(np.float32)
        train_y=z["y"][np.ix_(pt,ct)].astype(np.float32)
        validation_y=z["y"][np.ix_(pv,cv)].astype(np.float32)
    protein=normalize_protein(pfeat,pt)[0]["centered"]
    prot=torch.as_tensor(protein,dtype=torch.float32,device=device)
    chem=torch.as_tensor(bits,dtype=torch.float32,device=device)
    prior_np=(
        train_y.mean(axis=0).astype(np.float64)
        if category=="protein" else
        build_chemical_prior(bits,ct,train_y.mean(axis=0,dtype=np.float64))[0]
    )
    if category=="protein":
        # For protein only training used ALL 68 chemical identities.
        if ct.tolist()!=list(range(68)) or cv.tolist()!=list(range(68)):
            raise RuntimeError("Protein strict train/validation compound order drifted")
    else:
        if set(ct)&set(cv):raise RuntimeError("Chemical prior leakage: train/val compound overlap")
    prior=torch.as_tensor(prior_np,dtype=torch.float32,device=device)
    sd=float(train_y.std())
    dy=root/branch
    frozen=json.loads((dy/"VALIDATION_FROZEN.json").read_text(encoding="utf-8"))
    if category=="protein":
        if frozen["status"]!="VALIDATION_SELECTED_LOCKED_NO_TEST" or not frozen["test_labels_not_used"]:
            raise RuntimeError("Protein validation was not frozen")
        if [a["seed"] for a in frozen["train_records"]]!=list(SEEDS):
            raise RuntimeError("Protein selected seed set diverged")
    else:
        if frozen["status"] not in ("VALIDATION_CHECKPOINTS_FROZEN_NO_TEST",
                                  "VALIDATION_CHECKPOINTS_RECOVERED_AND_FROZEN_NO_TEST"):
            raise RuntimeError("Chemistry validation was not frozen")
        if not frozen["test_labels_never_loaded"]:
            raise RuntimeError("Chemistry checkpoint reports test labels read")
    if sha(lock)!=frozen.get("split_manifest_sha256",frozen.get("split_sha256")):
        raise RuntimeError("DynaCPI validation is on a different split")
    preds=[]
    for seed in SEEDS:
        cp=dy/"checkpoints"/f"{NAME}_seed{seed}.pt"
        if category=="protein":
            rec=next(r for r in frozen["train_records"] if r["seed"]==seed)
            if sha(cp)!=rec["checkpoint_sha256"]:
                raise RuntimeError("Selected protein checkpoint hash drifted")
        elif "completed_seeds" in frozen:
            for r in frozen["completed_seeds"]:
                if r["seed"]==seed and r.get("checkpoint_sha256") and sha(cp)!=r["checkpoint_sha256"]:
                    raise RuntimeError("Chemical checkpoint hash drifted")
        with torch.serialization.safe_globals([type(np.dtype(np.int32))]):
            ck=load_local_checkpoint(cp)
        if ck["split_sha256"]!=sha(lock) or ck["prepared_features_sha256"]!=sha(PREP):
            raise RuntimeError("Checkpoint source or split SHA changed")
        if ck["seed"]!=seed or ck["variant"]!=NAME:
            raise RuntimeError("Checkpoint seed or architecture was changed")
        if category=="protein":
            if not np.allclose(ck["chemical_train_mean"],prior_np,atol=1e-6):
                raise RuntimeError("Protein checkpoint train-only anchor differs")
            if abs(ck["train_label_sd"]-sd)>1e-6:
                raise RuntimeError("Protein checkpoint training label scale changed")
        else:
            if not np.allclose(ck["chemical_all_prior"],prior_np,atol=1e-6):
                raise RuntimeError("Cold checkpoint train-only extrapolation anchor differs")
            if not np.array_equal(np.asarray(ck["trained_compound_ids"],dtype=int),ct):
                raise RuntimeError("Chemical checkpoint contains a nontraining ligand")
        model=LowRankDynaCPI().to(device)
        model.load_state_dict(ck["state_dict"],strict=True)
        preds.append(grid_prediction(model,pv,prot,chem,prior,sd,device)[:,cv])
        del model
    ensemble=np.mean(preds,axis=0)
    replay=metrics(ensemble,validation_y)
    for k in ("macro_rmse","pooled_rmse"):
        if abs(replay[k]-frozen["final_validation"][k])>1e-6:
            raise RuntimeError(f"{name} {k} disagrees with original frozen validation")
    return dict(scenario=name,dynacpi_two_seed_val_macro_rmse=replay["macro_rmse"],
                pooled_rmse=replay["pooled_rmse"],n_val_pairs=int(validation_y.size),
                selected_checkpoint_validation_replay="PASS",
                test_labels_read=False,source_hash_verified=True)
def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device="cuda" if torch.cuda.is_available() else "cpu"
    out=[run_one(*item,device) for item in CASES]
    print(json.dumps({"status":"PASS_FOUR_DAVIS_FROZEN_VAL_REPLAYS_BEFORE_TEST",
                      "device":device,"no_sealed_test_labels_read":True,"results":out},indent=2),flush=True)
if __name__=="__main__":main()
