"""Validation-only feature reliance control for six frozen external DynaCPI models.

Measures whether DynaCPI predictions depend on protein features:
- keeps validation observed pair IDs/affinities and training-only chemical priors fixed
- permutes normalized protein-embedding rows among distinct validation proteins
- repeats five independent SHA-documented fixed-seed permutations
- reuses BOTH existing validation-selected checkpoint weights (NO RETRAINING)
- NEVER indexes test labels or changes any frozen train/validation/test split
Descriptive model sanity check, NOT a confirmatory test or comparable original
PerceiverCPI evaluation. Do not choose another model using these outcomes.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,torch
from prepare_external_kiba_metz_frozen_benchmarks_20261110 import OUT,file_sha
from evaluate_external_kiba_metz_all_six_once_20261110 import selected_predictions
from train_external_sparse_dynacpi_v5_20261110 import metrics,infer_observed
DATASETS=("metz","kiba")
SCENARIOS=("protein_cold","compound_cold","double_cold")
PERMUTATION_SEEDS=(31001,31002,31003,31004,31005)
def one(dataset,scenario):
    folder=OUT/dataset/scenario
    report_file=folder/"DYNACPI_PROTEIN_SHUFFLE_VALIDATION_DIAGNOSTIC.json"
    if report_file.exists():raise RuntimeError("Do not overwrite previous diagnostic")
    s=json.loads((folder/"SPLIT_LOCKED.json").read_text(encoding="utf-8"))
    if s["dataset"]!=dataset or s["scenario"]!=scenario or not s["test_labels_not_exported"]:
        raise RuntimeError("Frozen split protocol drift")
    train=np.asarray(s["pair_train_row_indices"],dtype=int)
    val=np.asarray(s["pair_val_row_indices"],dtype=int)
    tst=np.asarray(s["pair_test_row_indices"],dtype=int)
    if (set(train)&set(val)) or (set(train)&set(tst)) or (set(val)&set(tst)):
        raise RuntimeError("Train/val/test overlap")
    npz=OUT/dataset/"source_features.npz"
    with np.load(npz,allow_pickle=False) as z:
        data={key:z[key] for key in ("protein_features","compound_morgan_bits",
                  "protein_index","compound_index","observed_score","chemical_smiles")}
    selected=selected_predictions(data,s,folder/"dynacpi_fixed_sparse_train")
    # Checkpoint-based validation replay was verified inside selected_predictions
    # WITHOUT any test-affinity references.
    yval=data["observed_score"][val].astype(np.float64)
    pid=data["protein_index"];cid=data["compound_index"]
    val_proteins=np.unique(pid[val])
    if len(val_proteins)<3:raise RuntimeError("Too few validation proteins to shuffle")
    device=selected["prot"].device
    results=[]
    for seed in PERMUTATION_SEEDS:
        rng=np.random.default_rng(seed)
        perm=rng.permutation(val_proteins)
        if np.array_equal(perm,val_proteins):
            perm=np.roll(perm,1)
        protein_perm=selected["prot"].clone()
        protein_perm[val_proteins]=selected["prot"][perm]
        each=[infer_observed(m,val,pid,cid,protein_perm,selected["chem"],
               selected["prior"],selected["sd"]) for m in selected["models"]]
        out=np.mean(each,axis=0)
        scr=metrics(out,yval,pid[val])
        results.append({"permutation_seed":seed,"macro_rmse":scr["macro_rmse"],
            "pooled_rmse":scr["pooled_rmse"],
            "delta_macro_vs_correct":float(scr["macro_rmse"]-selected["validation_replay"]["macro_rmse"]),
            "delta_pooled_vs_correct":float(scr["pooled_rmse"]-selected["validation_replay"]["pooled_rmse"])})
    orig=selected["validation_replay"]
    obj={
     "status":"VALIDATION_PROTEIN_REPRESENTATION_PERMUTATION_NEGATIVE_CONTROL",
     "dataset":dataset,"scenario":scenario,
     "frozen_manifest_sha256":file_sha(folder/"SPLIT_LOCKED.json"),
     "frozen_source_features_sha256":file_sha(npz),
     "test_labels_never_indexed":True,
     "no_training_or_checkpoint_modification":True,
     "n_val_observations":int(len(val)),
     "n_val_proteins":len(val_proteins),
     "correct_validation_two_seed":{"macro_rmse":orig["macro_rmse"],"pooled_rmse":orig["pooled_rmse"]},
     "protein_features_shuffled_within_val_proteins_only":True,
     "chemical_features_and_training_only_anchor_unchanged":True,
     "permutations":results,
     "mean_delta_macro_rmse":float(np.mean([x["delta_macro_vs_correct"] for x in results])),
     "mean_delta_pooled_rmse":float(np.mean([x["delta_pooled_vs_correct"] for x in results])),
     "limitations":"Validation-only retrospective diagnostic, not an independent held-out experiment. Low reliance on shuffled features suggests chemical prior dominance and limits claimed interaction novelty."
    }
    report_file.write_text(json.dumps(obj,indent=2),encoding="utf-8")
    print(json.dumps({"dataset":dataset,"scenario":scenario,
      "original_macro":orig["macro_rmse"],
      "shuffle_mean_macro_delta":obj["mean_delta_macro_rmse"],
      "shuffle_mean_pooled_delta":obj["mean_delta_pooled_rmse"],
      "n_proteins_shuffled":len(val_proteins),
      "test_labels_not_read":True}),flush=True)
def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for checkpoint replay")
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            one(dataset,scenario)
    print("COMPLETE_SIX_VALIDATION_PROTEIN_FEATURE_NEGATIVE_CONTROLS_NO_TEST",flush=True)
if __name__=="__main__":main()
