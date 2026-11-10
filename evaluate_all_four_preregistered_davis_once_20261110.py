"""Fail-closed, ONE-TIME final benchmark gate for 4 predeclared DAVIS splits.

preflight: read ONLY frozen manifest, checkpoint files and logs, NO test labels.
evaluate: allowed only once ALL 8 original PerceiverCPI runs finished 30 epochs
          and ALL FOUR DynaCPI two-seed validation choices were frozen.
          Launch fixed evaluators without changing tuning, record EVERY result
          (positive and negative), create final cross-scenario comparison.

All raw test rows, checkpoints, predictions and metrics remain gitignored and
local. This script NEVER selects architecture, checkpoint or favorable fold.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from prepare_preregistered_davis_generalization_20261110 import OUT
from train_full_davis_v5 import sha,PREP

SPECS=(
    ("protein_strict_20261113","protein","dynacpi_v5_fixed_train",
     "evaluate_preregistered_strict_protein_parity_20261110.py",
     "MATCHED_TWO_SEED_BLIND_TEST_ONCE","FINAL_SAME_SPLIT_METRICS.json"),
    ("protein_strict_20261114","protein","dynacpi_v5_fixed_train",
     "evaluate_preregistered_strict_protein_parity_20261110.py",
     "MATCHED_TWO_SEED_BLIND_TEST_ONCE","FINAL_SAME_SPLIT_METRICS.json"),
    ("compound_scaffold_20261115","compound","dynacpi_cold_chem_fixed_train",
     "evaluate_preregistered_cold_chem_parity_20261110.py",
     "MATCHED_TWO_SEED_CHEM_BLIND_TEST_ONCE","FINAL_LOCKED_TEST_METRICS.json"),
    ("double_cold_20261116","double","dynacpi_cold_chem_fixed_train",
     "evaluate_preregistered_cold_chem_parity_20261110.py",
     "MATCHED_TWO_SEED_CHEM_BLIND_TEST_ONCE","FINAL_LOCKED_TEST_METRICS.json"),
)
SEEDS=(2931,2932)
HERE=Path(__file__).resolve().parent
DEST=OUT/"ONE_TIME_GENERALIZATION_ALL_SCENARIOS"
RUNNER=HERE/"run_remaining_preregistered_perceiver_queue_20261110.ps1"
QUEUE=OUT/"OFFICIAL_TRAINING_QUEUE_20261110.log"
COMMIT="755f076ef51099a90332fd1ec8437f94e3750c0b"

def check():
    import torch
    summary=json.loads((OUT/"SUITE_LOCKED.json").read_text(encoding="utf-8"))
    assert len(summary["scenarios"])==4 and summary["holdout_label_exports"]==0
    statuses=[]
    for name,category,dynfolder,script,subdir,metrics in SPECS:
        splitdir=OUT/name
        manifest=splitdir/"SPLIT_LOCKED.json"
        item=next(v for v in summary["scenarios"] if v["scenario"]==name)
        if sha(manifest)!=item["split_manifest_sha256"]:
            raise RuntimeError("Split SHA mismatch: "+name)
        p=json.loads(manifest.read_text(encoding="utf-8"))
        if p["category"]!=category or p["training_seeds"]!=list(SEEDS) or not p["test_labels_not_exported"]:
            raise RuntimeError("Scenario protocol drift: "+name)
        if sha(PREP)!=p["source_features_sha256"]:
            raise RuntimeError("Frozen source features changed")
        for file,digest in p["input_csv_sha256"].items():
            if sha(splitdir/file)!=digest:
                raise RuntimeError("Changed train/validation CSV: "+name+"/"+file)
        baseline=[]
        for seed in SEEDS:
            run=splitdir/f"perceiver_official_seed{seed}_30ep"
            ck=run/"fold_0"/"model_0"/"model.pt"
            argsfile=run/"args.json"
            log=run/"verbose.log"
            okay=ck.exists() and argsfile.exists() and log.exists()
            if okay:
                info=json.loads(argsfile.read_text(encoding="utf-8"))
                text=log.read_text(encoding="utf-8",errors="replace")
                okay=("Epoch 29" in text and "Overall test mse" in text
                    and "Elapsed time" in text and int(info["epochs"])==30
                    and int(info["pytorch_seed"])==seed and int(info["seed"])==seed
                    and int(info["ensemble_size"])==1)
                for field,fn in (("data_path","train.csv"),
                                 ("separate_val_path","validation.csv"),
                                 ("separate_test_path","validation_proxy_test.csv")):
                    if Path(info[field]).resolve()!=(splitdir/fn).resolve():
                        okay=False
            baseline.append({"seed":seed,"complete":bool(okay),
                "checkpoint_sha256":sha(ck) if okay else None})
        frozen=splitdir/dynfolder/"VALIDATION_FROZEN.json"
        dy_ok=False
        if frozen.exists():
            data=json.loads(frozen.read_text(encoding="utf-8"))
            dy_ok=(data["status"] in (
                "VALIDATION_SELECTED_LOCKED_NO_TEST",
                "VALIDATION_CHECKPOINTS_FROZEN_NO_TEST",
                "VALIDATION_CHECKPOINTS_RECOVERED_AND_FROZEN_NO_TEST")
                and data.get("training_seeds",data.get("seeds"))==list(SEEDS)
                and (data.get("split_manifest_sha256",data.get("split_sha256"))==sha(manifest))
                and bool(data.get("test_labels_not_used",data.get("test_labels_never_loaded",False))))
        target=splitdir/subdir
        statuses.append({"scenario":name,"category":category,"dyna_validation_frozen":dy_ok,
                         "official_baselines":baseline,
                         "test_has_been_scored":target.exists(),
                         "test_output_expected":str(target/metrics),
                         "eligible":bool(dy_ok and all(v["complete"] for v in baseline) and not target.exists())})
    return statuses

def preflight():
    rows=check()
    print(json.dumps({"preflight":"NO_TEST_LABELS_READ","all_eligible":all(r["eligible"] for r in rows),
                      "scenarios":rows},indent=2),flush=True)
    return rows

def eval_all():
    if DEST.exists():
        raise RuntimeError("An all-scenario evaluation already exists; no re-evaluation")
    rows=preflight()
    if not all(r["eligible"] for r in rows):
        raise RuntimeError("Not ALL 8 official checkpoints + 4 Dyna freezes complete; REFUSE opening ANY test")
    # Mandatory all-four DynaCPI VALIDATION replay happens BEFORE ANY TEST data
    # are opened. This catches corrupted checkpoints or mismatched priors across
    # all four splits and prevents partial outcome exposure.
    audit_cmd=[sys.executable,"-u",str(HERE/"audit_davis_four_frozen_validation_replay_20261110.py")]
    audit=subprocess.run(audit_cmd,cwd=HERE,env={**os.environ,
        "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"},
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
    if audit.returncode!=0 or "PASS_FOUR_DAVIS_FROZEN_VAL_REPLAYS_BEFORE_TEST" not in audit.stdout:
        raise RuntimeError("Four-split frozen validation replay did not pass; refuse ALL new tests: "+audit.stdout[-2500:])
    print("ALL_FOUR_DAVIS_VALIDATION_CHECKPOINT_REPLAYS_VERIFIED_BEFORE_TEST",flush=True)
    # Do NOT open test labels here. The original per-split guard scripts each
    # validate their frozen feature and checkpoint provenance before test access.
    original=HERE/"matched_perceivercpi_20261110"/"upstream"
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=original,text=True).strip()!=COMMIT:
        raise RuntimeError("Official PerceiverCPI git source changed")
    if subprocess.check_output(["git","diff","--name-only","--","*.py"],cwd=original,text=True).strip():
        raise RuntimeError("Unmodified official baseline Python source was altered")
    DEST.mkdir(parents=True,exist_ok=False)
    manifest={"status":"EVALUATING_PREREGISTERED_ALL_4_SPLITS_ONCE",
              "all_eight_official_completed_before_first_test":True,
              "no_model_selection_after_new_test_access":True,
              "fixed_seed_set":list(SEEDS),
              "fixed_test_order":[item[0] for item in SPECS],
              "baseline_source_commit":COMMIT,
              "started_wall_time_epoch_s":time.time(),
              "scenarios":[]}
    (DEST/"STATUS.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    for name,category,dynfolder,script,subdir,metricfile in SPECS:
        cmd=[sys.executable,"-u",str(HERE/script),"--scenario",name]
        output=DEST/(name+".log")
        print("BEGIN_FROZEN_TEST "+name,flush=True)
        with output.open("w",encoding="utf-8") as f:
            p=subprocess.run(cmd,cwd=HERE,env={**os.environ,
              "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"},
              stdout=f,stderr=subprocess.STDOUT,check=False)
        record={"scenario":name,"run_exit":p.returncode,"log_file":output.name}
        metrics_path=OUT/name/subdir/metricfile
        if p.returncode==0 and metrics_path.exists():
            data=json.loads(metrics_path.read_text(encoding="utf-8"))
            if category=="protein":
                a,b=data["dynacpi_v5_two_seed"],data["perceiver_official_two_seed"]
            else:
                a,b=data["dynacpi_2seed"],data["official_perceiver_2seed"]
            record.update({
                "status":"TEST_COMPLETED",
                "n_test_pairs":int(data["n_pairs"]),
                "dynacpi_macro_rmse":float(a["macro_rmse"]),
                "perceiver_macro_rmse":float(b["macro_rmse"]),
                "dynacpi_pooled_rmse":float(a["pooled_rmse"]),
                "perceiver_pooled_rmse":float(b["pooled_rmse"]),
                "dynacpi_wins_both_metrics":bool(a["macro_rmse"]<b["macro_rmse"] and a["pooled_rmse"]<b["pooled_rmse"]),
                "dynacpi_wins_macro":bool(a["macro_rmse"]<b["macro_rmse"]),
                "dynacpi_wins_pooled":bool(a["pooled_rmse"]<b["pooled_rmse"]),
                "raw_metric_file_sha256":sha(metrics_path)
            })
        else:
            record.update({"status":"EVALUATION_ERROR_NO_RESULT","metric_file_exists":metrics_path.exists()})
        manifest["scenarios"].append(record)
        (DEST/"STATUS.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        print(json.dumps(record,indent=2),flush=True)
        if record["status"]!="TEST_COMPLETED":
            raise RuntimeError("STOP ON FIRST FAILURE; do not tamper with prior test results: "+name)
    # Resampling is PREDECLARED BEFORE ANY NEW TEST OUTCOMES; run after
    # every paired test has finished. Failures are recorded rather than hidden,
    # and never trigger model retuning or rerunning the test.
    for record in manifest["scenarios"]:
        scenario=record["scenario"]
        bootlog=DEST/(scenario+".paired_bootstrap.log")
        cmd=[sys.executable,"-u",str(HERE/"paired_bootstrap_preregistered_davis_20261110.py"),
             "--scenario",scenario]
        with bootlog.open("w",encoding="utf-8") as f:
            p=subprocess.run(cmd,cwd=HERE,env={**os.environ,
              "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"},
              stdout=f,stderr=subprocess.STDOUT,check=False)
        target=next(item for item in SPECS if item[0]==scenario)
        bs=OUT/scenario/target[4]/"PAIRED_BOOTSTRAP_DESCRIPTIVE.json"
        if p.returncode==0 and bs.exists():
            info=json.loads(bs.read_text(encoding="utf-8"))
            record["bootstrap_status"]="COMPLETED_DESCRIPTIVE_NOT_PROOF"
            record["macro_95pct_delta_interval"]=info["bootstrap"]["macro_rmse_delta_95pct_descriptive_interval"]
            record["pooled_95pct_delta_interval"]=info["bootstrap"]["pooled_rmse_delta_95pct_descriptive_interval"]
            record["bootstrap_n_units"]=info["bootstrap"]["n_resample_independent_units"]
            record["bootstrap_file_sha256"]=sha(bs)
        else:
            record["bootstrap_status"]="FAILED_PRESERVED_RAW_PAIRED_TEST"
            record["bootstrap_exit"]=p.returncode
            record["bootstrap_log"]=bootlog.name
    manifest["status"]="COMPLETE_FOUR_SCENARIO_MATCHED_BLIND_TESTS"
    manifest["completed_wall_time_epoch_s"]=time.time()
    manifest["n_positive_both_metrics"]=sum(r["dynacpi_wins_both_metrics"] for r in manifest["scenarios"])
    manifest["n_negative_or_mixed"]=4-manifest["n_positive_both_metrics"]
    manifest["overall_claim"] = (
       "Consistently lower error on all four prespecified DAVIS splits; not universal superiority"
       if manifest["n_positive_both_metrics"]==4 else
       "NOT consistently superior across all four prespecified DAVIS splits; report every loss"
    )
    (DEST/"ALL_FOUR_BLIND_TEST_RESULTS.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (DEST/"STATUS.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"final_status":manifest["status"],
      "wins":manifest["n_positive_both_metrics"],
      "negative_or_mixed":manifest["n_negative_or_mixed"],
      "claim":manifest["overall_claim"]},indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=("preflight","evaluate"))
    args=p.parse_args()
    if args.mode=="preflight":preflight()
    else:eval_all()
