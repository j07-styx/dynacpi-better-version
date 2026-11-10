"""Compare frozen DynaCPI and original PerceiverCPI on VALIDATION only.

Never opens the source test indices or creates a held-out test file.
For a fully trained scenario, predicts original PerceiverCPI on the exact
pre-existing validation_proxy_test.csv used by original model training.
Two official seeds are averaged in original score units. All models are
already selected on validation, so these scores are descriptive/in-sample
model-selection results, NOT independent generalization evidence.
"""
from __future__ import annotations
import argparse,csv,json,os,subprocess
from pathlib import Path
import numpy as np
from evaluate_external_kiba_metz_all_six_once_20261110 import (
    OUT,OFFICIAL,PYTHON,TRAINING_SEEDS,preflight,read_official_prediction,metrics,sha
)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--dataset",choices=("kiba","metz"),required=True)
    a.add_argument("--scenario",choices=("protein_cold","compound_cold","double_cold"),required=True)
    args=a.parse_args()
    dataset,scenario=args.dataset,args.scenario
    folder=OUT/dataset/scenario
    resultdir=folder/"ORIGINAL_VS_DYNA_VALIDATION_ONLY_20261111"
    if resultdir.exists():
        raise RuntimeError("Refuse overwrite of previous original validation comparison")
    statuses=preflight()  # scans all frozen inputs; never indexes test labels
    target=next(x for x in statuses if x["dataset"]==dataset and x["scenario"]==scenario)
    if not target["both_baselines_ready"]:
        raise RuntimeError("Both official 30-epoch validation-selected checkpoints required")
    split=json.loads((folder/"SPLIT_LOCKED.json").read_text(encoding="utf-8"))
    if split["csv_sha256"]["validation.csv"] != split["csv_sha256"]["validation_proxy_test.csv"]:
        raise RuntimeError("Original train proxy must be an unchanged validation copy")
    validpath=folder/"validation.csv"
    with validpath.open("r",newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows or set(rows[0])!={"smiles","sequence","label"}:
        raise RuntimeError("Unexpected validation CSV fields")
    if len(rows)!=len(split["pair_val_row_indices"]):
        raise RuntimeError("Manifest and validation CSV row counts disagree")
    y=np.asarray([float(row["label"]) for row in rows],dtype=np.float64)
    if not np.isfinite(y).all():
        raise RuntimeError("Nonfinite validation truth")
    mapping={}
    pids=[]
    for row in rows:
        sequence=row["sequence"]
        if sequence not in mapping:
            mapping[sequence]=len(mapping)
        pids.append(mapping[sequence])
    pids=np.asarray(pids,dtype=int)
    run=folder/"dynacpi_fixed_sparse_train"
    dyn=json.loads((run/"VALIDATION_FROZEN.json").read_text(encoding="utf-8"))
    if dyn["frozen_split_sha256"]!=sha(folder/"SPLIT_LOCKED.json"):
        raise RuntimeError("Dyna frozen split digest mismatch")
    if dyn["status"]!="VALIDATION_LOCKED_TWO_SEEDS_NO_TEST":
        raise RuntimeError("Dyna frozen selection status invalid")
    resultdir.mkdir(parents=True,exist_ok=False)
    preds=[]
    hashes={}
    for seed in TRAINING_SEEDS:
        checkpoint=folder/f"perceiver_official_seed{seed}_30ep"/"fold_0"/"model_0"/"model.pt"
        if not checkpoint.exists():
            raise RuntimeError("Missing selected official checkpoint")
        out=resultdir/f"official_val_pred_seed{seed}.csv"
        cmd=[str(PYTHON),"-u","predict.py",
             "--test_path",str(folder/"validation_proxy_test.csv"),
             "--preds_path",str(out),"--checkpoint_path",str(checkpoint),
             "--batch_size","128","--num_workers","0"]
        print(f"PREDICT_VALIDATION_ONLY seed={seed}",flush=True)
        (resultdir/f"official_val_predict_seed{seed}_command.json").write_text(
            json.dumps(cmd,indent=2),encoding="utf-8")
        with (resultdir/f"official_val_predict_seed{seed}.log").open("w",encoding="utf-8") as log:
            p=subprocess.run(cmd,cwd=OFFICIAL,
                env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"},
                stdout=log,stderr=subprocess.STDOUT,check=False)
        if p.returncode:
            raise RuntimeError(f"Official validation predict seed{seed} failed: {p.returncode}")
        pred=read_official_prediction(out,rows)
        if not np.isfinite(pred).all():
            raise RuntimeError("Nonfinite original validation prediction")
        preds.append(pred)
        hashes[str(seed)]=sha(checkpoint)
        print(json.dumps({"seed":seed,"validation_metrics":metrics(pred,y,pids)}),flush=True)
    per=metrics(np.mean(preds,axis=0),y,pids)
    dyna=dyn["validation_two_seed"]
    for key in ("macro_rmse","pooled_rmse"):
        if not np.isfinite(dyna[key]):
            raise RuntimeError("Invalid frozen Dyna validation score")
    compare={
        "status":"ORIGINAL_VALIDATION_VS_DYNACPI_VAL_SELECTED_NO_TEST",
        "dataset":dataset,"scenario":scenario,
        "validation_pairs":len(y),"n_proteins":len(mapping),
        "frozen_split_sha256":sha(folder/"SPLIT_LOCKED.json"),
        "official_checkpoint_sha256":hashes,
        "validation_csv_sha256":sha(validpath),
        "original_perceiver_two_seed_validation":per,
        "dynacpi_frozen_two_seed_validation":dyna,
        "delta_macro_dyna_minus_perceiver":dyna["macro_rmse"]-per["macro_rmse"],
        "delta_pooled_dyna_minus_perceiver":dyna["pooled_rmse"]-per["pooled_rmse"],
        "dyna_lower_both_val_rmse":bool(dyna["macro_rmse"]<per["macro_rmse"] and dyna["pooled_rmse"]<per["pooled_rmse"]),
        "limitations":"Both models/checkpoints were validation-selected. NOT sealed test evidence; not used to modify any frozen model or split."
    }
    (resultdir/"COMPLETE_VALIDATION_ONLY.json").write_text(json.dumps(compare,indent=2),encoding="utf-8")
    print(json.dumps(compare,indent=2),flush=True)

if __name__=="__main__":
    main()
