"""One-time matched held-out evaluation for OFFICIAL PerceiverCPI.

Do not run while training: requires 30 completed upstream epochs and validation-
selected checkpoint, without any held-out labels in its training data. Produces
local-only test CSV/predictions. Does NOT rewrite DynaCPI v5 results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
BENCH=ROOT/"matched_perceivercpi_20261110"
RUN=BENCH/"official_matched_v3_seed2931_30ep"
UPSTREAM=BENCH/"upstream"
FINAL=BENCH/"official_matched_v3_HOLDOUT_ONCE"
PYTHON=Path.home()/"dynacpi_v4"/".venv"/"Scripts"/"python.exe"
sys.path.insert(0,str(ROOT))
from prepare_full_davis_v5 import DATA,SPLIT,SRC,Y_SHA256,sha
from train_full_davis_v5 import metrics, RUN as V5_RUN

def validate_run():
    if FINAL.exists(): raise RuntimeError("One-time test output already exists; refusing to repeat")
    if not (RUN/"fold_0"/"model_0"/"model.pt").exists():
        raise RuntimeError("Official PerceiverCPI checkpoint does not exist")
    log=(RUN/"verbose.log").read_text(encoding="utf-8",errors="replace")
    if ("Epoch 29" not in log or "Overall test mse" not in log or "Elapsed time" not in log):
        raise RuntimeError("Official full 30-epoch run has NOT completed: refusing test access")
    args=json.loads((RUN/"args.json").read_text(encoding="utf-8"))
    if (int(args["epochs"])!=30 or int(args["ensemble_size"])!=1
        or int(args["pytorch_seed"])!=2931):
        raise RuntimeError("Official run setup differs from predeclared seed/epoch budget")
    if not Path(args["data_path"]).resolve()==(BENCH/"data_v2"/"train.csv").resolve():
        raise RuntimeError("Official train data differs from frozen split")
    if not Path(args["separate_val_path"]).resolve()==(BENCH/"data_v2"/"val.csv").resolve():
        raise RuntimeError("Official validation data differs from frozen split")
    if not Path(args["separate_test_path"]).resolve()==(BENCH/"data_v2"/"val_proxy_test.csv").resolve():
        raise RuntimeError("Upstream test alias was not validation clone")
    export=json.loads((BENCH/"data_v2"/"SPLIT_EXPORT.json").read_text(encoding="utf-8"))
    if not export["test_labels_not_exported"] or export["n_training_pairs"]!=20536 or export["n_validation_pairs"]!=4012:
        raise RuntimeError("Frozen train+validation export provenance invalid")
    for fn,expected in export["csv_sha256"].items():
        if sha(BENCH/"data_v2"/fn)!=expected:
            raise RuntimeError("Dataset export hash mismatch "+fn)
    upstream_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=UPSTREAM,text=True).strip()
    if upstream_commit!="755f076ef51099a90332fd1ec8437f94e3750c0b":
        raise RuntimeError("Unexpected official PerceiverCPI source revision")
    py_edits=subprocess.check_output(["git","diff","--name-only","--","*.py"],cwd=UPSTREAM,text=True).strip()
    if py_edits:raise RuntimeError("Official upstream source PY files modified: "+py_edits)
    if sha(DATA)!=export["prepared_features_sha256"]:
        raise RuntimeError("Frozen feature source changed")
    if sha(SRC/"Y")!=export["source_Y_sha256"] or sha(SRC/"Y")!=Y_SHA256:
        raise RuntimeError("DAVIS raw Y changed")
    if sha(SPLIT)!=export["split_manifest_sha256"]:
        raise RuntimeError("DAVIS split manifest changed")
    return {"upstream_commit":upstream_commit,"train_command_args":args,
            "official_checkpoint_sha256":sha(RUN/"fold_0"/"model_0"/"model.pt")}

def write_heldout():
    manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
    with np.load(DATA,allow_pickle=False) as z:
        train=z["train"].astype(int)
        val=z["val"].astype(int)
        test=z["test"].astype(int)
        smiles=z["smiles"].astype(str)
        y=z["y"].astype(np.float64)
    assert len(train)==302 and len(val)==59 and len(test)==81
    assert not set(train)&set(val) and not set(train)&set(test) and not set(val)&set(test)
    assert len(set(train)|set(val)|set(test))==442
    assert len(smiles)==68 and y.shape==(442,68)
    proteins=list(json.loads((SRC/"proteins.txt").read_text(encoding="utf-8")).values())
    assert len(proteins)==442
    rows=[(str(smiles[j]),proteins[int(i)],repr(float(y[i,j])))
          for i in test for j in range(68)]
    assert len(rows)==5508
    FINAL.mkdir(parents=True,exist_ok=False)
    path=FINAL/"test_exact_once.csv"
    with path.open("w",newline="",encoding="utf-8") as fh:
        writer=csv.writer(fh);writer.writerow(["smiles","sequence","label"]);writer.writerows(rows)
    return y[test], path

def score(test_y,pred_path,runinfo,test_csv):
    with pred_path.open("r",newline="",encoding="utf-8") as fh:
        r=csv.DictReader(fh);header=r.fieldnames or []
        if "label" not in header:
            raise RuntimeError("No official prediction column 'label': "+str(header))
        items=list(r)
    if len(items)!=test_y.size:
        raise RuntimeError(f"Count mismatch: predictions={len(items)} expected={test_y.size}")
    pred=np.asarray([float(v["label"]) for v in items],dtype=np.float64).reshape(test_y.shape)
    if not np.isfinite(pred).all():
        raise RuntimeError("Nonfinite predictions")
    perf=metrics(pred,test_y)
    v5=json.loads((V5_RUN/"RESULTS.json").read_text(encoding="utf-8"))
    baseline=v5["baseline_test"]
    dynacpi=v5["selected_test"]
    assert perf["n_proteins"]==baseline["n_proteins"]==dynacpi["n_proteins"]==81
    assert perf["n_pairs"]==5508
    report={
        "evaluation_status":"SINGLE_FROZEN_MATCHED_HELDOUT_TEST",
        "official_model":"dmis-lab/PerceiverCPI InteractionModel, 30 epochs, seed 2931",
        "upstream_commit":runinfo["upstream_commit"],
        "official_model_checkpoint_sha256":runinfo["official_checkpoint_sha256"],
        "data_source_raw_Y_sha256":Y_SHA256,
        "frozen_split_manifest_sha256":sha(SPLIT),
        "heldout_csv_sha256":sha(test_csv),
        "predictions_csv_sha256":sha(pred_path),
        "n_protein_entries":81,"n_compounds":68,"n_heldout_pairs":5508,
        "same_frozen_protein_grouped_split":True,
        "all_68_compounds_seen_in_training":True,
        "evaluation_code":"original official PerceiverCPI predict.py, local Python RDKit+TAPE environment",
        "single_official_seed_caveat":"One 30-epoch official PerceiverCPI checkpoint; DynaCPI v5 has two seeds; further official seeds improve fairness",
        "perceivercpi_official":perf,
        "dynacpi_v5_selected_ensemble":dynacpi,
        "dynacpi_v5_chemical_mean":baseline,
        "perceiver_minus_dynacpi_macro_rmse":float(perf["macro_rmse"]-dynacpi["macro_rmse"]),
        "perceiver_minus_dynacpi_pooled_rmse":float(perf["pooled_rmse"]-dynacpi["pooled_rmse"]),
        "claim_limits":"One predeclared frozen split; no family curated split; one PerceiverCPI seed. Do not infer generalized superiority without more splits/seeds."
    }
    (FINAL/"METRICS_MATCHED_TEST.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k in ("evaluation_status",
       "perceivercpi_official","dynacpi_v5_selected_ensemble",
       "perceiver_minus_dynacpi_macro_rmse","perceiver_minus_dynacpi_pooled_rmse")},indent=2),flush=True)
    print("SINGLE_MATCHED_TEST_COMPLETE",flush=True)

def main():
    state=validate_run()
    test_y,test_csv=write_heldout()
    dest=FINAL/"predictions_official.csv"
    env=os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"})
    cmd=[str(PYTHON),"-u","predict.py",
         "--test_path",str(test_csv),
         "--preds_path",str(dest),
         "--checkpoint_path",str(RUN/"fold_0"/"model_0"/"model.pt"),
         "--batch_size","128","--num_workers","0"]
    (FINAL/"PREDICTION_COMMAND.json").write_text(json.dumps(cmd,indent=2),encoding="utf-8")
    print("RUNNING_OFFICIAL_PREDICT",flush=True)
    p=subprocess.run(cmd,cwd=UPSTREAM,env=env,stdout=FINAL.joinpath("prediction.log").open("w",encoding="utf-8"),
        stderr=subprocess.STDOUT,check=False)
    if p.returncode!=0:
        raise RuntimeError("Original official predict.py failed (see prediction.log): "+str(p.returncode))
    score(test_y,dest,state,test_csv)

if __name__=="__main__":main()
