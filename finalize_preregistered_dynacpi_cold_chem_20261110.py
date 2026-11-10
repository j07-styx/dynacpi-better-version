"""Complete an interrupted cold-compound run from BOTH previously saved checkpoints.

No test-label access, no fitting, no new epoch selection. Only validation
prediction with existing per-seed validation-selected checkpoints and the
unchanged training-compound-only chemical prior.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from prepare_preregistered_davis_generalization_20261110 import OUT
from train_preregistered_dynacpi_cold_chem_20261110 import (
    PREP, SOURCE, ARMS, SEEDS, NAME, build_chemical_prior,
    LowRankDynaCPI, normalize_protein, grid_prediction, metrics, sha,
    write_json, load_local_checkpoint, RIDGE)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scenario",required=True,choices=sorted(ARMS))
    args=p.parse_args()
    dest=OUT/args.scenario/"dynacpi_cold_chem_fixed_train"
    frozen=dest/"VALIDATION_FROZEN.json"
    if frozen.exists():raise RuntimeError("Prior completed run exists; never overwrite")
    progress=json.loads((dest/"TRAIN_PROGRESS.json").read_text(encoding="utf-8"))
    if progress["completed_seeds"]!=list(SEEDS):
        raise RuntimeError("Both training seeds must finish before completing validation")
    lock=json.loads((dest/"TRAINING_PROTOCOL.json").read_text(encoding="utf-8"))
    assert lock["status"]=="RUNNING_TRAINING_AND_VALIDATION_ONLY"
    assert lock["training_seeds"]==list(SEEDS) and lock["scenario"]==args.scenario
    spec_path=OUT/args.scenario/"SPLIT_LOCKED.json"
    spec=json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["scenario"]==args.scenario and spec["test_labels_not_exported"]
    index=json.loads((OUT/"SUITE_LOCKED.json").read_text(encoding="utf-8"))
    ref=next(v for v in index["scenarios"] if v["scenario"]==args.scenario)
    if ref["split_manifest_sha256"]!=sha(spec_path):
        raise RuntimeError("Preregistered split hash mismatch")
    if lock["split_sha256"]!=sha(spec_path) or lock["prepared_features_sha256"]!=sha(PREP):
        raise RuntimeError("Training protocol provenance mismatch")
    ids=spec["indices"]
    train_p=np.asarray(ids["protein_train"],dtype=int)
    train_c=np.asarray(ids["compound_train"],dtype=int)
    val_p=np.asarray(ids["protein_val"],dtype=int)
    val_c=np.asarray(ids["compound_val"],dtype=int)
    assert not set(train_c)&set(val_c)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with np.load(PREP,allow_pickle=False) as z:
        pfeat=z["pfeat"].astype(np.float32)
        bits=z["bits"].astype(np.float32)
        ytrain=z["y"][np.ix_(train_p,train_c)].astype(np.float32)
        yval=z["y"][np.ix_(val_p,val_c)].astype(np.float32)
    prior_train=ytrain.mean(axis=0,dtype=np.float64)
    all_prior,_,_=build_chemical_prior(bits,train_c,prior_train)
    norm,_,_=normalize_protein(pfeat,train_p)
    label_sd=float(ytrain.std())
    device="cuda" if torch.cuda.is_available() else "cpu"
    proteins=torch.as_tensor(norm["centered"],device=device,dtype=torch.float32)
    chemicals=torch.as_tensor(bits,device=device,dtype=torch.float32)
    prior=torch.as_tensor(all_prior,device=device,dtype=torch.float32)
    predictions=[]
    epochs=[]
    cps=[]
    for seed in SEEDS:
        path=dest/"checkpoints"/f"{NAME}_seed{seed}.pt"
        if not path.is_file():raise RuntimeError("Checkpoint missing: "+str(path))
        # Safe torch weights-only local reconstruction, including older NumPy int32 IDs.
        with torch.serialization.safe_globals([type(np.dtype(np.int32))]):
            ck=load_local_checkpoint(path)
        if ck["seed"]!=seed or ck["split_sha256"]!=sha(spec_path):
            raise RuntimeError("Seed or split checkpoint mismatch")
        if ck["prepared_features_sha256"]!=sha(PREP):
            raise RuntimeError("Features checkpoint mismatch")
        if not np.allclose(ck["chemical_all_prior"],all_prior,atol=1e-6):
            raise RuntimeError("Chemical prior differs from saved checkpoint")
        if not np.allclose(ck["chemical_train_mean"],prior_train,atol=1e-6):
            raise RuntimeError("Training chemical means differ from checkpoint")
        if abs(ck["label_sd"]-label_sd)>1e-6:
            raise RuntimeError("Training SD changed")
        model=LowRankDynaCPI().to(device)
        model.load_state_dict(ck["state_dict"],strict=True)
        predictions.append(grid_prediction(model,val_p,proteins,chemicals,prior,label_sd,device)[:,val_c])
        epochs.append({"seed":seed,"best_epoch":int(ck["epoch"]),"checkpoint_sha256":sha(path)})
        cps.append(str(path))
        del model
    average=np.mean(predictions,axis=0)
    anchor=np.broadcast_to(all_prior[val_c],average.shape)
    result={
        **lock,
        "status":"VALIDATION_CHECKPOINTS_RECOVERED_AND_FROZEN_NO_TEST",
        "completed_seeds":epochs,
        "final_validation":metrics(average,yval),
        "anchor_only_validation":metrics(anchor,yval),
        "all_original_checkpoint_sha256s_verified":True,
        "test_labels_never_read":True,
        "frozen_validation_n_pairs":int(yval.size),
        "recovered_instead_of_retraining":True,
        "safe_weights_only_checkpoint_loading":True
    }
    write_json(frozen,result)
    print(json.dumps({"status":result["status"],"scenario":args.scenario,
        "seed_epochs":epochs,"ensemble":result["final_validation"],
        "anchor":result["anchor_only_validation"],
        "no_test_labels_accessed":True},indent=2),flush=True)
if __name__=="__main__":main()
