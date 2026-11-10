"""Validate and once-score all six frozen KIBA/Metz external generalization tests.

Two-seed DynaCPI vs unmodified official two-seed PerceiverCPI, SAME observed
pair rows and SAME split for each cohort. Never use test labels for training,
checkpoint selection, hyperparameter tuning or split selection.
Runs only AFTER all 12 original-model training runs have COMPLETED.
KIBA and Metz have different label units; never average their numeric RMSE.
"""
from __future__ import annotations
import argparse,ast,csv,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from train_full_davis_v5 import LowRankDynaCPI,normalize_protein,sha,load_local_checkpoint
from train_external_sparse_dynacpi_v5_20261110 import (
    OUT,ARM,TRAINING_SEEDS,DEVICE,metrics,infer_observed,frozen_training_only_prior)
from prepare_external_kiba_metz_frozen_benchmarks_20261110 import SOURCES
HERE=Path(__file__).resolve().parent
OFFICIAL=HERE/"matched_perceivercpi_20261110"/"upstream"
PYTHON=Path.home()/"dynacpi_v4"/".venv"/"Scripts"/"python.exe"
EXPECTED_COMMIT="755f076ef51099a90332fd1ec8437f94e3750c0b"
SCENARIOS=("protein_cold","compound_cold","double_cold")
DATASETS=("metz","kiba")
FINAL=OUT/"ALL_SIX_EXTERNAL_SAME_SPLIT_TESTS_ONCE"
INDIVIDUAL=OUT/"INDIVIDUAL_EXTERNAL_SAME_SPLIT_TESTS_ONCE_20261111"

def preflight():
    if FINAL.exists():raise RuntimeError("Final result folder already exists; refuse re-evaluation")
    upstream=subprocess.check_output(["git","rev-parse","HEAD"],cwd=OFFICIAL,text=True).strip()
    if upstream!=EXPECTED_COMMIT:raise RuntimeError("Changed official source git SHA")
    diff=subprocess.check_output(["git","diff","--name-only","--","*.py"],cwd=OFFICIAL,text=True).strip()
    if diff:raise RuntimeError("Official PerceiverCPI Python code changed")
    ready=[]
    for dataset in DATASETS:
        audit=json.loads((OUT/dataset/"SOURCE_AUDIT.json").read_text(encoding="utf-8"))
        if sha(OUT/dataset/"source_features.npz")!=audit["features_sha256"]:
            raise RuntimeError("External source features changed: "+dataset)
        for scenario in SCENARIOS:
            folder=OUT/dataset/scenario
            splitpath=folder/"SPLIT_LOCKED.json"
            split=json.loads(splitpath.read_text(encoding="utf-8"))
            item=next(x for x in audit["splits"] if x["name"]==scenario)
            if sha(splitpath)!=item["manifest_sha256"] or not split["test_labels_not_exported"]:
                raise RuntimeError("Frozen external split altered")
            for fn,digest in split["csv_sha256"].items():
                if sha(folder/fn)!=digest:raise RuntimeError("Train/val source CSV changed: "+dataset+"/"+scenario)
            if split["csv_sha256"]["validation.csv"]!=split["csv_sha256"]["validation_proxy_test.csv"]:
                raise RuntimeError("Official train test input not validation proxy")
            locked=folder/"dynacpi_fixed_sparse_train"/"VALIDATION_FROZEN.json"
            if not locked.exists():raise RuntimeError("Missing frozen DynaCPI selection")
            d=json.loads(locked.read_text(encoding="utf-8"))
            if d["status"]!="VALIDATION_LOCKED_TWO_SEEDS_NO_TEST":
                raise RuntimeError("DynaCPI frozen validation status invalid")
            if d["frozen_split_sha256"]!=sha(splitpath) or d["training_seeds"]!=list(TRAINING_SEEDS):
                raise RuntimeError("DynaCPI saved selection does not match frozen split")
            official=[]
            for seed in TRAINING_SEEDS:
                run=folder/f"perceiver_official_seed{seed}_30ep"
                ck=run/"fold_0"/"model_0"/"model.pt"
                args=run/"args.json";log=run/"verbose.log"
                ok=ck.exists() and args.exists() and log.exists()
                if ok:
                    content=log.read_text(encoding="utf-8",errors="replace")
                    cfg=json.loads(args.read_text(encoding="utf-8"))
                    ok=("Epoch 29" in content and "Overall test mse" in content
                        and "Elapsed time" in content
                        and int(cfg["epochs"])==30 and int(cfg["pytorch_seed"])==seed
                        and int(cfg["seed"])==seed and int(cfg["ensemble_size"])==1)
                    for field,fn in [("data_path","train.csv"),("separate_val_path","validation.csv"),
                                     ("separate_test_path","validation_proxy_test.csv")]:
                        if Path(cfg[field]).resolve()!=(folder/fn).resolve():ok=False
                official.append({"seed":seed,"finished":bool(ok),
                                 "checkpoint_sha256":sha(ck) if ok else None})
            ready.append({"dataset":dataset,"scenario":scenario,"dyna_validation_frozen":True,
                          "official_seeds":official,"both_baselines_ready":all(x["finished"] for x in official)})
    return ready

def read_records(dataset):
    if dataset=="kiba":
        path=SOURCES["kiba"]["root"]/"proteins.txt"
        return list(json.loads(path.read_text(encoding="utf-8")).values())
    d=pd.read_csv(SOURCES["metz"]["root"]/"prot_info.csv")
    return d.sort_values("index")["sequence"].astype(str).tolist()

def selected_predictions(data,split,run):
    train=np.asarray(split["pair_train_row_indices"],dtype=int)
    validation=np.asarray(split["pair_val_row_indices"],dtype=int)
    pids=data["protein_index"]
    cids=data["compound_index"]
    y=data["observed_score"]
    pfeat=data["protein_features"]
    bits=data["compound_morgan_bits"]
    seen=np.unique(pids[train])
    norm,_,_=normalize_protein(pfeat,seen)
    prot=torch.as_tensor(norm["centered"],device=DEVICE,dtype=torch.float32)
    chem=torch.as_tensor(bits,device=DEVICE,dtype=torch.float32)
    prior_np,known,_=frozen_training_only_prior(bits,cids[train],y[train])
    prior=torch.as_tensor(prior_np,device=DEVICE,dtype=torch.float32)
    label_sd=float(y[train].std())
    result=[]
    models=[]
    for seed in TRAINING_SEEDS:
        cp=run/"checkpoints"/f"{ARM}_seed{seed}.pt"
        ck=load_local_checkpoint(cp)
        if ck["seed"]!=seed or ck["frozen_split_sha256"]!=sha(run.parent/"SPLIT_LOCKED.json"):
            raise RuntimeError("Wrong DynaCPI checkpoint seed/split")
        if ck["source_features_sha256"]!=sha(run.parent.parent/"source_features.npz"):
            raise RuntimeError("Wrong checkpoint source features")
        if not np.allclose(ck["training_compound_prior"],prior_np,atol=1e-5):
            raise RuntimeError("Train-only prior changed from checkpoint")
        if abs(ck["label_sd"]-label_sd)>1e-5:
            raise RuntimeError("Label SD changed")
        m=LowRankDynaCPI().to(DEVICE)
        m.load_state_dict(ck["model_state"],strict=True)
        result.append(infer_observed(m,validation,pids,cids,prot,chem,prior,label_sd))
        models.append(m)
    ensemble=np.mean(result,axis=0)
    verify=metrics(ensemble,y[validation],pids[validation])
    previous=json.loads((run/"VALIDATION_FROZEN.json").read_text(encoding="utf-8"))
    for k in ["macro_rmse","pooled_rmse"]:
        if abs(verify[k]-previous["validation_two_seed"][k])>1e-5:
            raise RuntimeError("Frozen DynaCPI VAL replay mismatch: "+run.parent.name+" "+k)
    return {"models":models,"pids":pids,"cids":cids,"truth":y,
            "prot":prot,"chem":chem,"prior":prior,"sd":label_sd,
            "validation_replay":verify,
            "checkpoint_hashes":{str(seed):sha(run/"checkpoints"/f"{ARM}_seed{seed}.pt") for seed in TRAINING_SEEDS}}

def read_official_prediction(path,rows):
    with path.open("r",newline="",encoding="utf-8") as fh:generated=list(csv.DictReader(fh))
    if len(generated)!=len(rows):raise RuntimeError("Original Perceiver predictions differ in count")
    for i,(pred,orig) in enumerate(zip(generated,rows)):
        s=pred["smiles"]
        try:
            alt=ast.literal_eval(s)
            if isinstance(alt,list) and len(alt)==1:s=alt[0]
        except (ValueError,SyntaxError):pass
        if s!=orig["smiles"]:
            raise RuntimeError("Original Perceiver prediction row SMILES misaligned "+str(i))
    return np.asarray([float(x["label"]) for x in generated],dtype=np.float64)

def evaluate():
    if FINAL.exists():raise RuntimeError("Once-only result directory already exists")
    ready=preflight()
    if not all(item["both_baselines_ready"] for item in ready):
        raise RuntimeError("Not all twelve original checkpoints completed. REFUSE ALL TEST LABELS.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("Fixed GPU evaluation expected")
    FINAL.mkdir(parents=True,exist_ok=False)
    any_individual=INDIVIDUAL.exists() and bool(list(INDIVIDUAL.glob("*/TEST_RESULTS.json")))
    ledger={"status":"EVALUATING_ONCE","original_perceiver_source_commit":EXPECTED_COMMIT,
            "all_12_checkpoints_val_selected_before_first_test":not any_individual,
            "all_12_checkpoints_val_selected_before_final_aggregate":True,
            "all_six_dynacpi_checkpoints_frozen_before_any_external_test":True,
            "individually_scored_scenarios_imported_without_rescoring":True,
            "all_six_splits_predeclared":True,"scenarios":[]}
    (FINAL/"STATE.json").write_text(json.dumps(ledger,indent=2),encoding="utf-8")
    for dataset in DATASETS:
        source=OUT/dataset/"source_features.npz"
        # ALL official selected checkpoints were already verified before this
        # observed-score NPZ is opened.
        with np.load(source,allow_pickle=False) as z:
            data={n:z[n] for n in ("protein_features","compound_morgan_bits",
                      "protein_index","compound_index","observed_score","chemical_smiles")}
        seqs=read_records(dataset)
        for scenario in SCENARIOS:
            folder=OUT/dataset/scenario
            split=json.loads((folder/"SPLIT_LOCKED.json").read_text(encoding="utf-8"))
            run=folder/"dynacpi_fixed_sparse_train"
            individual=INDIVIDUAL/(dataset+"_"+scenario)
            if individual.exists():
                resultfile=individual/"TEST_RESULTS.json"
                if not resultfile.exists():
                    raise RuntimeError("Partial once-only individual test exists; REFUSE repeat")
                item=json.loads(resultfile.read_text(encoding="utf-8"))
                if (item.get("status")!="SAME_SPLIT_TWO_SEED_ONCE_TEST_COMPLETE"
                    or item["dataset"]!=dataset or item["scenario"]!=scenario
                    or item["n_test_pairs"]!=len(split["pair_test_row_indices"])
                    or item["frozen_split_sha256"]!=sha(folder/"SPLIT_LOCKED.json")
                    or item["source_features_sha256"]!=sha(source)):
                    raise RuntimeError("Individual test provenance differs from frozen source")
                for seed in TRAINING_SEEDS:
                    ck=folder/f"perceiver_official_seed{seed}_30ep"/"fold_0"/"model_0"/"model.pt"
                    dynck=run/"checkpoints"/f"{ARM}_seed{seed}.pt"
                    if (sha(ck)!=item["perceiver_checkpoint_shas"][str(seed)]
                        or sha(dynck)!=item["dynacpi_checkpoint_shas"][str(seed)]):
                        raise RuntimeError("Individual test checkpoint digest mismatch")
                if (sha(individual/"sealed_test_exact_once.csv")!=item["heldout_csv_sha256"]
                    or sha(individual/"paired_raw_predictions_private.npz")!=item["paired_raw_pred_sha256"]):
                    raise RuntimeError("Existing individual test artifacts were altered")
                shutil.copytree(individual,FINAL/(dataset+"_"+scenario))
                ledger["scenarios"].append(item)
                (FINAL/"STATE.json").write_text(json.dumps(ledger,indent=2),encoding="utf-8")
                print(json.dumps({"imported_once_scored_test":dataset+"/"+scenario,
                    "test_reopened":False,"delta_macro":item["delta_macro_dyna_minus_perceiver"]}),flush=True)
                continue
            # Validation replay is performed BEFORE indexing held-out outcomes.
            selected=selected_predictions(data,split,run)
            test=np.asarray(split["pair_test_row_indices"],dtype=int)
            pids=data["protein_index"];cids=data["compound_index"]
            ytest=data["observed_score"][test].astype(np.float64)
            if not np.isfinite(ytest).all():raise RuntimeError("Non-finite test score in source")
            sub=FINAL/(dataset+"_"+scenario)
            sub.mkdir(parents=True,exist_ok=False)
            rows=[{"smiles":str(data["chemical_smiles"][cids[k]]),
                   "sequence":seqs[pids[k]],
                   "label":repr(float(ytest[i]))} for i,k in enumerate(test)]
            test_csv=sub/"sealed_test_exact_once.csv"
            with test_csv.open("w",newline="",encoding="utf-8") as fh:
                writer=csv.DictWriter(fh,fieldnames=["smiles","sequence","label"])
                writer.writeheader();writer.writerows(rows)
            official_predictions=[]
            checkpoint_shas={}
            for seed in TRAINING_SEEDS:
                ck=folder/f"perceiver_official_seed{seed}_30ep"/"fold_0"/"model_0"/"model.pt"
                if not ck.exists():raise RuntimeError("Original baseline missing after preflight")
                prediction=sub/f"original_perceiver_seed{seed}.csv"
                cmd=[str(PYTHON),"-u","predict.py","--test_path",str(test_csv),
                     "--preds_path",str(prediction),"--checkpoint_path",str(ck),
                     "--batch_size","128","--num_workers","0"]
                (sub/f"predict_seed{seed}.json").write_text(json.dumps(cmd,indent=2),encoding="utf-8")
                env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"}
                with (sub/f"original_predict_seed{seed}.log").open("w",encoding="utf-8") as f:
                    output=subprocess.run(cmd,cwd=OFFICIAL,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
                if output.returncode:
                    raise RuntimeError(f"Original Perceiver prediction failed on {dataset}/{scenario} seed{seed}")
                official_predictions.append(read_official_prediction(prediction,rows))
                checkpoint_shas[str(seed)]=sha(ck)
            per=np.mean(official_predictions,axis=0)
            dyn=np.mean([infer_observed(m,test,selected["pids"],selected["cids"],
                    selected["prot"],selected["chem"],selected["prior"],selected["sd"])
                    for m in selected["models"]],axis=0)
            if dyn.shape!=ytest.shape or per.shape!=ytest.shape:
                raise RuntimeError("Paired test shape disagreement")
            dm=metrics(dyn,ytest,pids[test]);pm=metrics(per,ytest,pids[test])
            raw=sub/"paired_raw_predictions_private.npz"
            np.savez_compressed(raw,labels=ytest,protein_index=pids[test],
                      compound_index=cids[test],pred_dynacpi=dyn,pred_original_perceiver=per)
            item={
             "status":"SAME_SPLIT_TWO_SEED_ONCE_TEST_COMPLETE",
             "dataset":dataset,"scenario":scenario,
             "n_test_pairs":int(len(test)),
             "dataset_label_definition":split["label_definition"],
             "source_features_sha256":sha(source),
             "frozen_split_sha256":sha(folder/"SPLIT_LOCKED.json"),
             "heldout_csv_sha256":sha(test_csv),"paired_raw_pred_sha256":sha(raw),
             "dynacpi_checkpoint_shas":selected["checkpoint_hashes"],
             "perceiver_checkpoint_shas":checkpoint_shas,
             "validation_replay":selected["validation_replay"],
             "dynacpi_2seed":dm,"official_perceiver_2seed":pm,
             "dynacpi_wins_both":bool(dm["macro_rmse"]<pm["macro_rmse"] and dm["pooled_rmse"]<pm["pooled_rmse"]),
             "delta_macro_dyna_minus_perceiver":float(dm["macro_rmse"]-pm["macro_rmse"]),
             "delta_pooled_dyna_minus_perceiver":float(dm["pooled_rmse"]-pm["pooled_rmse"]),
             "limits":"One outcome-blind deterministic split; not paper fold or curated family protocol. Report all failures; not universal superiority."
            }
            (sub/"TEST_RESULTS.json").write_text(json.dumps(item,indent=2),encoding="utf-8")
            ledger["scenarios"].append(item)
            (FINAL/"STATE.json").write_text(json.dumps(ledger,indent=2),encoding="utf-8")
            print(json.dumps({k:item[k] for k in ("dataset","scenario","n_test_pairs",
                "delta_macro_dyna_minus_perceiver","delta_pooled_dyna_minus_perceiver",
                "dynacpi_wins_both")}),flush=True)
    # The paired sequence-group / Murcko-scaffold uncertainty protocol was
    # predeclared and synthetically tested BEFORE any sealed external outcomes.
    # Run only once all six paired original-model tests have been recorded.
    # An uncertainty failure is reported, not hidden or used to tune models.
    for item in ledger["scenarios"]:
        dataset=item["dataset"]
        scenario=item["scenario"]
        sub=FINAL/(dataset+"_"+scenario)
        existing=sub/"PAIRED_CLUSTER_BOOTSTRAP.json"
        if existing.exists():
            boot=json.loads(existing.read_text(encoding="utf-8"))
            if (boot["original_test_result_sha256"]!=sha(sub/"TEST_RESULTS.json")
                or boot["original_raw_predictions_sha256"]!=sha(sub/"paired_raw_predictions_private.npz")
                or boot["dataset"]!=dataset or boot["scenario"]!=scenario):
                raise RuntimeError("Imported precomputed paired bootstrap provenance mismatch")
            item["paired_bootstrap_status"]="COMPLETE_DESCRIPTIVE_IMPORTED_UNCHANGED"
            item["macro_95pct_difference_interval"]=boot["bootstrap"]["macro_delta_95pct_interval"]
            item["pooled_95pct_difference_interval"]=boot["bootstrap"]["pooled_delta_95pct_interval"]
            item["bootstrap_file_sha256"]=sha(existing)
            (FINAL/"STATE.json").write_text(json.dumps(ledger,indent=2),encoding="utf-8")
            continue
        log=sub/"paired_bootstrap.log"
        cmd=[str(PYTHON),"-u",str(HERE/"paired_bootstrap_external_kiba_metz_20261110.py"),
             "--dataset",dataset,"--scenario",scenario]
        with log.open("w",encoding="utf-8") as output:
            p=subprocess.run(cmd,cwd=HERE,env={**os.environ,
                "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"},
                stdout=output,stderr=subprocess.STDOUT,check=False)
        bootfile=sub/"PAIRED_CLUSTER_BOOTSTRAP.json"
        if p.returncode==0 and bootfile.exists():
            boot=json.loads(bootfile.read_text(encoding="utf-8"))
            item["paired_bootstrap_status"]="COMPLETE_DESCRIPTIVE"
            item["macro_95pct_difference_interval"]=boot["bootstrap"]["macro_delta_95pct_interval"]
            item["pooled_95pct_difference_interval"]=boot["bootstrap"]["pooled_delta_95pct_interval"]
            item["bootstrap_file_sha256"]=sha(bootfile)
        else:
            item["paired_bootstrap_status"]="FAILED_PRESERVE_UNCHANGED_TEST"
            item["paired_bootstrap_exit_code"]=p.returncode
        (FINAL/"STATE.json").write_text(json.dumps(ledger,indent=2),encoding="utf-8")
    ledger["status"]="ALL_SIX_EXTERNAL_MATCHED_TWO_SEED_TESTS_COMPLETE"
    ledger["n_both_metric_wins"]=sum(x["dynacpi_wins_both"] for x in ledger["scenarios"])
    ledger["n_not_both_metric_wins"]=6-ledger["n_both_metric_wins"]
    ledger["claim"]="Winning on all listed new external splits does not prove universal superiority"
    (FINAL/"COMPLETE_SIX_RESULTS.json").write_text(json.dumps(ledger,indent=2),encoding="utf-8")
    (FINAL/"STATE.json").write_text(json.dumps(ledger,indent=2),encoding="utf-8")
    print(json.dumps({"finished":ledger["status"],
          "n_both_metric_wins":ledger["n_both_metric_wins"],
          "n_losses_or_mixed":ledger["n_not_both_metric_wins"]}),flush=True)

if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--mode",required=True,choices=["preflight","validate-only","evaluate"])
    args=arg.parse_args()
    if args.mode=="preflight":
        r=preflight()
        print(json.dumps({"preflight":"NO_TEST_LABELS_ACCESSED",
          "all_12_ready":all(x["both_baselines_ready"] for x in r),
          "status":r},indent=2),flush=True)
    elif args.mode=="validate-only":
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        for dataset in DATASETS:
            with np.load(OUT/dataset/"source_features.npz",allow_pickle=False) as z:
                data={n:z[n] for n in ("protein_features","compound_morgan_bits",
                    "protein_index","compound_index","observed_score","chemical_smiles")}
            for scenario in SCENARIOS:
                root=OUT/dataset/scenario
                locked=json.loads((root/"SPLIT_LOCKED.json").read_text(encoding="utf-8"))
                result=selected_predictions(data,locked,root/"dynacpi_fixed_sparse_train")
                for model in result["models"]:del model
                print(json.dumps({"dataset":dataset,"scenario":scenario,
                    "validation_replay_verified":True,
                    "macro_rmse":result["validation_replay"]["macro_rmse"],
                    "test_not_indexed":True}),flush=True)
        print("PASS_ALL_SIX_EXTERNAL_FROZEN_VALIDATION_REPLAYS_NO_TEST_SCORING",flush=True)
    else:evaluate()
