"""Validation-only molecular feature reliance control across six external DynaCPI models.

Retain each validation observation's TRUE label, target protein representation
and training-only chemical-mean anchor. Permute only the chemical Morgan bits
fed into the *learned neural residual*, across validation ligand identities.
This isolates learned chemical structure usage above the chemically derived
anchor. Does not alter predictions/checkpoints, split definitions or test data.
Five fixed permutations, no model selection.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from evaluate_external_kiba_metz_all_six_once_20261110 import selected_predictions
from prepare_external_kiba_metz_frozen_benchmarks_20261110 import OUT,file_sha
from train_external_sparse_dynacpi_v5_20261110 import metrics,infer_observed
DATASETS=("metz","kiba")
SCENARIOS=("protein_cold","compound_cold","double_cold")
SEEDS=(32001,32002,32003,32004,32005)
def main_one(dataset,scenario):
    dir=OUT/dataset/scenario
    resultfile=dir/"DYNACPI_CHEMICAL_FEATURE_SHUFFLE_VALIDATION_DIAGNOSTIC.json"
    if resultfile.exists():raise RuntimeError("Refusing overwrite of previous molecular diagnostic")
    specfile=dir/"SPLIT_LOCKED.json"
    spec=json.loads(specfile.read_text(encoding="utf-8"))
    if not spec["test_labels_not_exported"]:
        raise RuntimeError("Frozen chemical model scenario altered")
    val=np.asarray(spec["pair_val_row_indices"],dtype=int)
    tr=np.asarray(spec["pair_train_row_indices"],dtype=int)
    test=np.asarray(spec["pair_test_row_indices"],dtype=int)
    if set(tr)&set(val) or set(val)&set(test) or set(tr)&set(test):
        raise RuntimeError("Frozen observed pair partitions overlap")
    npz=OUT/dataset/"source_features.npz"
    with np.load(npz,allow_pickle=False) as f:
        data={n:f[n] for n in ("protein_features","compound_morgan_bits",
            "protein_index","compound_index","observed_score","chemical_smiles")}
    sel=selected_predictions(data,spec,dir/"dynacpi_fixed_sparse_train")
    pids=data["protein_index"]
    cids=data["compound_index"]
    truth=data["observed_score"][val].astype(np.float64)
    chemicals=np.unique(cids[val])
    if len(chemicals)<5:raise RuntimeError("Too few unique validation molecules for permutation control")
    perms=[]
    for seed in SEEDS:
        newids=np.random.default_rng(seed).permutation(chemicals)
        if np.array_equal(newids,chemicals):newids=np.roll(newids,1)
        modified_chem=sel["chem"].clone()
        modified_chem[chemicals]=sel["chem"][newids]
        ens=np.mean([infer_observed(m,val,pids,cids,sel["prot"],
                    modified_chem,sel["prior"],sel["sd"]) for m in sel["models"]],axis=0)
        met=metrics(ens,truth,pids[val])
        base=sel["validation_replay"]
        perms.append({"seed":seed,"macro_rmse":met["macro_rmse"],
                      "pooled_rmse":met["pooled_rmse"],
                      "delta_macro_rmse":met["macro_rmse"]-base["macro_rmse"],
                      "delta_pooled_rmse":met["pooled_rmse"]-base["pooled_rmse"]})
    out={
        "status":"VALIDATION_ONLY_MOLECULAR_NEURAL_RESIDUAL_FEATURE_PERMUTATION",
        "dataset":dataset,"scenario":scenario,
        "split_sha256":file_sha(specfile),
        "features_sha256":file_sha(npz),
        "validation_two_seed_correct":sel["validation_replay"],
        "n_observed_validation_pairs":int(len(val)),
        "n_unique_val_chemicals_shuffled":int(len(chemicals)),
        "n_fixed_permutations":len(SEEDS),
        "chemical_prior_kept_correct_for_true_compound":True,
        "protein_embeddings_not_modified":True,
        "model_weights_not_retrained":True,
        "test_labels_never_indexed":True,
        "permutations":perms,
        "mean_delta_macro_rmse":float(np.mean([x["delta_macro_rmse"] for x in perms])),
        "mean_delta_pooled_rmse":float(np.mean([x["delta_pooled_rmse"] for x in perms])),
        "interpretation":"Validation-only indication of learned chemical-branch signal beyond the training-derived chemical anchor. Small deltas or better shuffled performance limit mechanistic claims; NOT a new test or proof of broad superiority."
    }
    resultfile.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({"dataset":dataset,"scenario":scenario,
       "correct_val_macro":sel["validation_replay"]["macro_rmse"],
       "mean_chemical_shuffle_macro_delta":out["mean_delta_macro_rmse"],
       "mean_chemical_shuffle_pooled_delta":out["mean_delta_pooled_rmse"],
       "no_test_access":True}),flush=True)
def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("CUDA needed")
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            main_one(dataset,scenario)
    print("COMPLETE_ALL_SIX_EXTERNAL_CHEMICAL_NEURAL_SIGNAL_NEGATIVE_CONTROLS_NO_TEST",flush=True)
if __name__=="__main__":main()
