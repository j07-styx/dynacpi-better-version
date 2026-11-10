"""Frozen two-seed PerceiverCPI parity evaluation (DO NOT RUN until 2932 completed).
Reuses the exact, already-exposed original DAVIS 81-protein test; this is a
same-split replication NOT a fresh independent confirmatory cohort. No
checkpoint tuning or selection based on test results.
"""
from __future__ import annotations
import ast
import csv
import json
import os
import subprocess
from pathlib import Path
import numpy as np
from train_full_davis_v5 import metrics, RUN as V5_RUN
from prepare_full_davis_v5 import DATA,SPLIT,SRC,Y_SHA256,sha

HERE=Path(__file__).resolve().parent
B=HERE/"matched_perceivercpi_20261110"
UPSTREAM=B/"upstream"
PRIOR=B/"official_matched_v3_HOLDOUT_ONCE"
RUN1=B/"official_matched_v3_seed2931_30ep"
RUN2=B/"official_matched_v4_seed2932_30ep"
OUT=B/"parity_2seed_heldout"
PYTHON=Path.home()/"dynacpi_v4"/".venv"/"Scripts"/"python.exe"

def confirm_complete(run,seed):
    log=(run/"verbose.log").read_text(encoding="utf-8",errors="replace")
    if "Epoch 29" not in log or "Overall test mse" not in log or "Elapsed time" not in log:
        raise RuntimeError(f"Seed {seed} has not completed its fixed 30 epochs; no test labels will be used")
    args=json.loads((run/"args.json").read_text())
    if int(args["epochs"])!=30 or int(args["pytorch_seed"])!=seed or int(args["ensemble_size"])!=1:
        raise RuntimeError("Changed official baseline training protocol")
    if Path(args["data_path"]).resolve()!=(B/"data_v2"/"train.csv").resolve():
        raise RuntimeError("Training CSV changed")
    if Path(args["separate_val_path"]).resolve()!=(B/"data_v2"/"val.csv").resolve():
        raise RuntimeError("Validation CSV changed")
    if Path(args["separate_test_path"]).resolve()!=(B/"data_v2"/"val_proxy_test.csv").resolve():
        raise RuntimeError("Validation-only proxy changed")
    ck=run/"fold_0"/"model_0"/"model.pt"
    if not ck.is_file():raise RuntimeError("Official model checkpoint missing")
    return ck

def read_prediction(path,tests):
    if not path.exists():raise RuntimeError("Prediction missing: "+str(path))
    with path.open("r",newline="",encoding="utf-8") as fh:
        data=list(csv.DictReader(fh))
    if len(data)!=len(tests) or len(tests)!=5508:
        raise RuntimeError("Wrong prediction count")
    for k,(a,b) in enumerate(zip(data,tests)):
        smi=a["smiles"]
        try:
            parsed=ast.literal_eval(smi)
            if isinstance(parsed,list) and len(parsed)==1:smi=parsed[0]
        except (ValueError,SyntaxError):
            pass
        if smi!=b["smiles"]:
            raise RuntimeError(f"Perceiver prediction row mismatch: {k}")
    return np.array([float(a["label"]) for a in data],dtype=np.float64).reshape(81,68)

def main():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite parity test results")
    ck1=confirm_complete(RUN1,2931)
    ck2=confirm_complete(RUN2,2932)
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=UPSTREAM,text=True).strip()!="755f076ef51099a90332fd1ec8437f94e3750c0b":
        raise RuntimeError("Upstream source version changed")
    if subprocess.check_output(["git","diff","--name-only","--","*.py"],cwd=UPSTREAM,text=True).strip():
        raise RuntimeError("Upstream official Python source was modified")
    prior_metrics=json.loads((PRIOR/"METRICS_MATCHED_TEST.json").read_text(encoding="utf-8"))
    if prior_metrics["official_model_checkpoint_sha256"]!=sha(ck1):
        raise RuntimeError("Original official seed2931 checkpoint changed")
    if prior_metrics["predictions_csv_sha256"]!=sha(PRIOR/"predictions_official.csv"):
        raise RuntimeError("Original first-seed prediction changed")
    if prior_metrics["frozen_split_manifest_sha256"]!=sha(SPLIT) or sha(SRC/"Y")!=Y_SHA256:
        raise RuntimeError("Original DAVIS split or raw Y changed")
    if prior_metrics["heldout_csv_sha256"]!=sha(PRIOR/"test_exact_once.csv"):
        raise RuntimeError("Original once-only test row file changed")
    with (PRIOR/"test_exact_once.csv").open("r",newline="",encoding="utf-8") as fh:
        test=list(csv.DictReader(fh))
    if len(test)!=5508:raise RuntimeError("Original test row count not 5508")
    y=np.array([float(a["label"]) for a in test],dtype=np.float64).reshape(81,68)
    p1=read_prediction(PRIOR/"predictions_official.csv",test)
    if abs(metrics(p1,y)["macro_rmse"]-prior_metrics["perceivercpi_official"]["macro_rmse"])>1e-9:
        raise RuntimeError("Original PerceiverCPI test metrics changed")
    # All test roles are now fixed and independently chosen checkpoints exist.
    OUT.mkdir(parents=True,exist_ok=False)
    output=OUT/"predictions_seed2932.csv"
    cmd=[str(PYTHON),"-u","predict.py",
         "--test_path",str(PRIOR/"test_exact_once.csv"),
         "--preds_path",str(output),
         "--checkpoint_path",str(ck2),
         "--batch_size","128","--num_workers","0"]
    (OUT/"PREDICT_SEED2932_COMMAND.json").write_text(json.dumps(cmd,indent=2))
    env=os.environ.copy();env.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"})
    print("SECOND_SEED_PREDICTION_STARTED",flush=True)
    with (OUT/"prediction.log").open("w",encoding="utf-8") as log:
        p=subprocess.run(cmd,cwd=UPSTREAM,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
    if p.returncode:raise RuntimeError(f"Official predictor error {p.returncode}; inspect log")
    p2=read_prediction(output,test)
    ensemble=(p1+p2)/2
    per1=metrics(p1,y);per2=metrics(p2,y);perm=metrics(ensemble,y)
    v5=json.loads((V5_RUN/"RESULTS.json").read_text(encoding="utf-8"))
    base=v5["selected_test"]
    assert base["n_proteins"]==81 and base["n_pairs"]==5508
    summary={
      "status":"EXISTING_EXPOSED_SINGLE_SPLIT_2SEED_FAIR_ENSEMBLE",
      "not_fresh_test":True,
      "n_heldout_proteins":81,"n_test_pairs":5508,
      "official_source_git_sha":"755f076ef51099a90332fd1ec8437f94e3750c0b",
      "official_seed2931_checkpoint_sha256":sha(ck1),
      "official_seed2932_checkpoint_sha256":sha(ck2),
      "official_seed2931_pred_sha256":sha(PRIOR/"predictions_official.csv"),
      "official_seed2932_pred_sha256":sha(output),
      "official_seed2931_test":per1,"official_seed2932_test":per2,
      "official_2seed_ensemble_test":perm,"dynacpi_v5_2seed_test":base,
      "delta_macro_rmse_dyna_minus_official":float(base["macro_rmse"]-perm["macro_rmse"]),
      "delta_pooled_rmse_dyna_minus_official":float(base["pooled_rmse"]-perm["pooled_rmse"]),
      "note":"This removes the initial 2-seed-vs-1-seed asymmetry on one already-exposed split. Different pretrained encoders and hyperparameters still limit fairness. No new model selection on heldout test."
    }
    (OUT/"PARITY_METRICS.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    print("FAIR_TWO_SEED_PARITY_COMPLETE",flush=True)

if __name__=="__main__":main()
