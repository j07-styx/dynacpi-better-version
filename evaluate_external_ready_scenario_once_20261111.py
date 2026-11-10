"""Once-only early scoring of an already fully frozen KIBA/Metz scenario.

Protocol amendment: all SIX DynaCPI models, splits, and benchmark hyperparameters
were locked before ANY external test. For each ready scenario, TWO original
PerceiverCPI checkpoints must have finished their original validation-based
selection before that scenario's test may be opened ONCE. Remaining scenarios
stay sealed. The final six-scenario aggregator IMPORTS the locked individual
result without reopening/re-scoring its test. Test scores never trigger tuning.
"""
from __future__ import annotations
import argparse,csv,json,os,subprocess
from pathlib import Path
import numpy as np,torch
from evaluate_external_kiba_metz_all_six_once_20261110 import (
    OUT,FINAL,OFFICIAL,PYTHON,TRAINING_SEEDS,EXPECTED_COMMIT,
    preflight,read_records,selected_predictions,read_official_prediction,
    infer_observed,metrics,sha
)

INDIVIDUAL=OUT/"INDIVIDUAL_EXTERNAL_SAME_SPLIT_TESTS_ONCE_20261111"

def readiness(dataset,scenario):
    if FINAL.exists():
        raise RuntimeError("Aggregate sealed test already started; refuse individual test")
    pre=preflight() # checks all six frozen DynaCPI selections and split hashes; no tests read
    entry=next(x for x in pre if x["dataset"]==dataset and x["scenario"]==scenario)
    if not entry["both_baselines_ready"]:
        raise RuntimeError("Two validated official checkpoints missing for "+dataset+"/"+scenario)
    dest=INDIVIDUAL/(dataset+"_"+scenario)
    if dest.exists():
        raise RuntimeError("Already attempted individual test; refuse reopening even if incomplete")
    return entry,dest

def evaluate_once(dataset,scenario):
    ready,dest=readiness(dataset,scenario)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError("Identical CUDA evaluation required")
    root=OUT/dataset
    folder=root/scenario
    split=json.loads((folder/"SPLIT_LOCKED.json").read_text(encoding="utf-8"))
    with np.load(root/"source_features.npz",allow_pickle=False) as z:
        data={n:z[n] for n in ("protein_features","compound_morgan_bits",
               "protein_index","compound_index","observed_score","chemical_smiles")}
    seqs=read_records(dataset)
    # Replay validation checkpoint selection BEFORE ANY held-out row is indexed.
    selected=selected_predictions(data,split,folder/"dynacpi_fixed_sparse_train")
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"STATE.json").write_text(json.dumps({
        "status":"SCORING_ONCE_NO_RERUN_ON_FAILURE",
        "dataset":dataset,"scenario":scenario,
        "official_checkpoints_verified":ready["official_seeds"],
        "all_six_dynacpi_models_frozen_before_first_external_test":True,
        "protocol_amendment":"Independent one-time scenario evaluation, final report imports unchanged"
    },indent=2),encoding="utf-8")
    test=np.asarray(split["pair_test_row_indices"],dtype=int)
    pids=data["protein_index"];cids=data["compound_index"]
    ytest=data["observed_score"][test].astype(np.float64)
    if len(test)==0 or not np.isfinite(ytest).all():
        raise RuntimeError("Invalid held-out outcomes")
    rows=[{"smiles":str(data["chemical_smiles"][cids[k]]),
           "sequence":seqs[pids[k]],"label":repr(float(ytest[i]))}
          for i,k in enumerate(test)]
    test_csv=dest/"sealed_test_exact_once.csv"
    with test_csv.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["smiles","sequence","label"])
        writer.writeheader();writer.writerows(rows)
    official_predictions=[];checkpoints={}
    for seed in TRAINING_SEEDS:
        checkpoint=folder/f"perceiver_official_seed{seed}_30ep"/"fold_0"/"model_0"/"model.pt"
        predpath=dest/f"original_perceiver_seed{seed}.csv"
        cmd=[str(PYTHON),"-u","predict.py","--test_path",str(test_csv),
             "--preds_path",str(predpath),"--checkpoint_path",str(checkpoint),
             "--batch_size","128","--num_workers","0"]
        (dest/f"predict_seed{seed}.json").write_text(json.dumps(cmd,indent=2),encoding="utf-8")
        print(f"ONCE_ONLY_HELDOUT_PREDICTION {dataset}/{scenario} seed{seed}",flush=True)
        with (dest/f"original_predict_seed{seed}.log").open("w",encoding="utf-8") as log:
            result=subprocess.run(cmd,cwd=OFFICIAL,env={**os.environ,
                 "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"},
                 stdout=log,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            raise RuntimeError("Original prediction failure; preserve partial sealed run; do not retry")
        official_predictions.append(read_official_prediction(predpath,rows))
        checkpoints[str(seed)]=sha(checkpoint)
    per=np.mean(official_predictions,axis=0)
    dyn=np.mean([infer_observed(m,test,selected["pids"],selected["cids"],
        selected["prot"],selected["chem"],selected["prior"],selected["sd"])
        for m in selected["models"]],axis=0)
    if per.shape!=ytest.shape or dyn.shape!=ytest.shape:
        raise RuntimeError("Prediction shapes unequal; preserve partial once-test")
    dm=metrics(dyn,ytest,pids[test]);pm=metrics(per,ytest,pids[test])
    raw=dest/"paired_raw_predictions_private.npz"
    np.savez_compressed(raw,labels=ytest,protein_index=pids[test],
            compound_index=cids[test],pred_dynacpi=dyn,pred_original_perceiver=per)
    item={
       "status":"SAME_SPLIT_TWO_SEED_ONCE_TEST_COMPLETE",
       "dataset":dataset,"scenario":scenario,"n_test_pairs":len(test),
       "dataset_label_definition":split["label_definition"],
       "source_features_sha256":sha(root/"source_features.npz"),
       "frozen_split_sha256":sha(folder/"SPLIT_LOCKED.json"),
       "heldout_csv_sha256":sha(test_csv),"paired_raw_pred_sha256":sha(raw),
       "dynacpi_checkpoint_shas":selected["checkpoint_hashes"],
       "perceiver_checkpoint_shas":checkpoints,
       "validation_replay":selected["validation_replay"],
       "dynacpi_2seed":dm,"official_perceiver_2seed":pm,
       "dynacpi_wins_both":bool(dm["macro_rmse"]<pm["macro_rmse"] and
                                dm["pooled_rmse"]<pm["pooled_rmse"]),
       "delta_macro_dyna_minus_perceiver":float(dm["macro_rmse"]-pm["macro_rmse"]),
       "delta_pooled_dyna_minus_perceiver":float(dm["pooled_rmse"]-pm["pooled_rmse"]),
       "individual_once_protocol_amendment":True,
       "limits":"Frozen two-seed same-split test; all DynaCPI models sealed before external testing. Other baseline training incomplete at test time; no tuning is allowed. Not a paper-fold comparison."
    }
    (dest/"TEST_RESULTS.json").write_text(json.dumps(item,indent=2),encoding="utf-8")
    (dest/"STATE.json").write_text(json.dumps({
        "status":"SAME_SPLIT_TWO_SEED_ONCE_TEST_COMPLETE",
        "dataset":dataset,"scenario":scenario,
        "test_results_sha256":sha(dest/"TEST_RESULTS.json")
    },indent=2),encoding="utf-8")
    print(json.dumps(item,indent=2),flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--dataset",required=True,choices=("metz","kiba"))
    cli.add_argument("--scenario",required=True,choices=("protein_cold","compound_cold","double_cold"))
    cli.add_argument("--mode",required=True,choices=("preflight","evaluate"))
    opts=cli.parse_args()
    if opts.mode=="preflight":
        target,_=readiness(opts.dataset,opts.scenario)
        print(json.dumps({"mode":"SAFE_NO_TEST_PREFLIGHT","target":target},indent=2))
    else:
        evaluate_once(opts.dataset,opts.scenario)
