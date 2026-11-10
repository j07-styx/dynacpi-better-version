"""Guarded blind strict-protein DAVIS comparison: 2-seed DynaCPI vs 2-seed official PerceiverCPI.

DO NOT CALL UNTIL BOTH official 30-epoch seeds are completed on the SAME frozen
train/validation split. Replays DynaCPI validation from frozen checkpoints
BEFORE generating or inspecting test labels, and writes a single immutable
result folder. No validation or test-model tweaking. NOT the published fivefold
protocol or a curated kinase-family test.
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from train_full_davis_v5 import (
    PREP,LowRankDynaCPI,normalize_protein,grid_prediction,metrics,sha,
    load_local_checkpoint)
from prepare_preregistered_davis_generalization_20261110 import OUT,SOURCE
HERE=Path(__file__).resolve().parent
OFFICIAL=HERE/"matched_perceivercpi_20261110"/"upstream"
PY=Path.home()/"dynacpi_v4"/".venv"/"Scripts"/"python.exe"
SEEDS=(2931,2932)
ARM="centered_esm_censored"
OFFICIAL_COMMIT="755f076ef51099a90332fd1ec8437f94e3750c0b"
CHOICES=("protein_strict_20261113","protein_strict_20261114")

def expected_training_run(run,seed,split):
    if not run.exists():
        raise RuntimeError(f"Official Perceiver seed {seed} not trained on {split.name}")
    path=run/"fold_0"/"model_0"/"model.pt"
    if not path.is_file():raise RuntimeError(f"No official checkpoint seed {seed}")
    text=(run/"verbose.log").read_text(encoding="utf-8",errors="replace")
    if "Epoch 29" not in text or "Overall test mse" not in text or "Elapsed time" not in text:
        raise RuntimeError(f"Seed {seed} missing completed 30 epochs")
    info=json.loads((run/"args.json").read_text(encoding="utf-8"))
    assert int(info["epochs"])==30 and int(info["ensemble_size"])==1
    assert int(info["pytorch_seed"])==seed and int(info["seed"])==seed
    must={"data_path":"train.csv","separate_val_path":"validation.csv",
          "separate_test_path":"validation_proxy_test.csv"}
    for field,filename in must.items():
        if Path(info[field]).resolve()!=(split/filename).resolve():
            raise RuntimeError(f"Official seed{seed}: {field} changed")
    return path

def score_pred_csv(output,rows,nprots,nchem):
    with output.open(newline="",encoding="utf-8") as fh:
        pred=list(csv.DictReader(fh))
    if len(pred)!=len(rows) or len(rows)!=nprots*nchem:
        raise RuntimeError(f"Prediction length mismatch: {len(pred)}, {len(rows)}")
    for k,(pr,raw) in enumerate(zip(pred,rows)):
        smi=pr["smiles"]
        try:
            alt=ast.literal_eval(smi)
            if isinstance(alt,list) and len(alt)==1:smi=alt[0]
        except (ValueError,SyntaxError):
            pass
        if smi!=raw["smiles"]:
            raise RuntimeError(f"Official SMILES row mismatch {k}")
    return np.array([float(pr["label"]) for pr in pred],dtype=np.float64).reshape(nprots,nchem)

def initialize_and_check(name):
    spec_dir=OUT/name
    p=spec_dir/"SPLIT_LOCKED.json"
    spec=json.loads(p.read_text(encoding="utf-8"))
    suite=json.loads((OUT/"SUITE_LOCKED.json").read_text(encoding="utf-8"))
    scenario=next(x for x in suite["scenarios"] if x["scenario"]==name)
    if sha(p)!=scenario["split_manifest_sha256"] or not spec["test_labels_not_exported"]:
        raise RuntimeError("Frozen split changed")
    if spec["category"]!="protein" or spec["training_seeds"]!=list(SEEDS):
        raise RuntimeError("Wrong preregistered protein protocol")
    for fname,digest in spec["input_csv_sha256"].items():
        if sha(spec_dir/fname)!=digest:raise RuntimeError("Train/validation CSV hash changed")
    if spec["input_csv_sha256"]["validation_proxy_test.csv"]!=spec["input_csv_sha256"]["validation.csv"]:
        raise RuntimeError("Official test proxy not an identical validation copy")
    if sha(PREP)!=spec["source_features_sha256"] or sha(SOURCE/"source"/"Y")!=spec["source_Y_sha256"]:
        raise RuntimeError("Raw DAVIS input changed")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=OFFICIAL,text=True).strip()!=OFFICIAL_COMMIT:
        raise RuntimeError("Official PerceiverCPI git commit changed")
    if subprocess.check_output(["git","diff","--name-only","--","*.py"],cwd=OFFICIAL,text=True).strip():
        raise RuntimeError("Official PerceiverCPI source modified")
    paths={seed:expected_training_run(
        spec_dir/f"perceiver_official_seed{seed}_30ep",seed,spec_dir)
        for seed in SEEDS}
    dy=spec_dir/"dynacpi_v5_fixed_train"
    frozen=dy/"VALIDATION_FROZEN.json"
    if not frozen.exists():raise RuntimeError("DynaCPI validation not frozen")
    v=json.loads(frozen.read_text(encoding="utf-8"))
    if v["split_manifest_sha256"]!=sha(p) or not v["test_labels_not_used"]:
        # This exact version of v5 script used a different naming convention.
        if v.get("split_manifest_sha256")==sha(p) and v.get("test_labels_never_read",False):
            pass
        else:raise RuntimeError("DynaCPI validation checkpoint protocol mismatch")
    if v["status"]!="VALIDATION_SELECTED_LOCKED_NO_TEST":
        raise RuntimeError("DynaCPI validation selection not locked")
    if [r["seed"] for r in v["train_records"]]!=list(SEEDS):
        raise RuntimeError("Frozen DynaCPI seed set differs")
    for item in v["train_records"]:
        local=dy/"checkpoints"/item["checkpoint_name"]
        if sha(local)!=item["checkpoint_sha256"]:
            raise RuntimeError("DynaCPI checkpoint hash changed")
    return spec_dir,spec,p,paths,dy,v

def dyna_predict_model(checkpoint,allp,prot,chem,prior,sd,device,split_hash):
    with torch.serialization.safe_globals([type(np.dtype(np.int32))]):
        ck=load_local_checkpoint(checkpoint)
    if ck["split_sha256"]!=split_hash or ck["variant"]!=ARM:
        raise RuntimeError("DynaCPI source checkpoint provenance mismatch")
    if ck["prepared_features_sha256"]!=sha(PREP):
        raise RuntimeError("DynaCPI feature provenance mismatch")
    model=LowRankDynaCPI().to(device)
    model.load_state_dict(ck["state_dict"],strict=True)
    out=grid_prediction(model,allp,prot,chem,prior,sd,device)
    del model
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenario",required=True,choices=CHOICES)
    args=parser.parse_args()
    root,spec,spec_path,official,dy,v=initialize_and_check(args.scenario)
    dest=root/"MATCHED_TWO_SEED_BLIND_TEST_ONCE"
    if dest.exists():raise RuntimeError("A previous official blind comparison exists")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ids=spec["indices"]
    tr=np.array(ids["protein_train"],dtype=int)
    val=np.array(ids["protein_val"],dtype=int)
    tst=np.array(ids["protein_test"],dtype=int)
    if ids["compound_train"]!=list(range(68)) or ids["compound_val"]!=list(range(68)):
        raise RuntimeError("These strict protein splits require 68 warm compounds")
    # Validate model prediction reconstruction against old validation metric
    # BEFORE accessing test labels.
    with np.load(PREP,allow_pickle=False) as z:
        prot_raw=z["pfeat"].astype(np.float32)
        chem_raw=z["bits"].astype(np.float32)
        smiles=z["smiles"].astype(str)
        ytrain=z["y"][tr].astype(np.float32)
        yval=z["y"][val].astype(np.float32)
    assert not set(tr)&set(val) and not set(tr)&set(tst)
    centered,_,_=normalize_protein(prot_raw,tr)
    device="cuda" if torch.cuda.is_available() else "cpu"
    prot=torch.as_tensor(centered["centered"],dtype=torch.float32,device=device)
    chem=torch.as_tensor(chem_raw,dtype=torch.float32,device=device)
    prior_train=ytrain.mean(axis=0).astype(np.float64)
    prior=torch.as_tensor(prior_train,dtype=torch.float32,device=device)
    sd=float(ytrain.std())
    val_pred=np.mean([dyna_predict_model(
        dy/"checkpoints"/f"{ARM}_seed{seed}.pt",val,prot,chem,prior,sd,device,sha(spec_path))
        for seed in SEEDS],axis=0)
    val_scores=metrics(val_pred,yval)
    for key in ("macro_rmse","pooled_rmse"):
        if abs(val_scores[key]-v["final_validation"][key])>1e-6:
            raise RuntimeError("DynaCPI validation replay differs; refuse test opening")
    # CHECKPOINTS FROZEN ON VALIDATION, ALL INPUT SHA CHECKS COMPLETE.
    # Only now read the original, previously sealed TEST labels.
    with np.load(PREP,allow_pickle=False) as z:
        ytest=z["y"][tst].astype(np.float64)
        floor=z["censored"][tst].astype(bool)
    seq=list(json.loads((SOURCE/"source"/"proteins.txt").read_text(encoding="utf-8")).values())
    rows=[{"smiles":str(smiles[j]),"sequence":seq[int(i)],"label":repr(float(ytest[k,j]))}
          for k,i in enumerate(tst) for j in range(68)]
    if len(rows)!=spec["test_pairs_not_exported"]:raise RuntimeError("Test row count wrong")
    dest.mkdir(parents=True,exist_ok=False)
    testcsv=dest/"test_exact_once.csv"
    with testcsv.open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=["smiles","sequence","label"])
        w.writeheader();w.writerows(rows)
    env=os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"})
    per_predictions=[]
    official_hashes={}
    for seed in SEEDS:
        prediction=dest/f"official_seed{seed}_predictions.csv"
        cmd=[str(PY),"-u","predict.py","--test_path",str(testcsv),
           "--preds_path",str(prediction),"--checkpoint_path",str(official[seed]),
           "--batch_size","128","--num_workers","0"]
        (dest/f"official_seed{seed}_command.json").write_text(json.dumps(cmd,indent=2),encoding="utf-8")
        with (dest/f"official_seed{seed}.log").open("w",encoding="utf-8") as log:
            p=subprocess.run(cmd,cwd=OFFICIAL,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        if p.returncode:raise RuntimeError(f"Official predict.py failed seed {seed}: {p.returncode}")
        per_predictions.append(score_pred_csv(prediction,rows,len(tst),68))
        official_hashes[str(seed)]={"checkpoint":sha(official[seed]),"predictions":sha(prediction)}
    dy_pred=np.mean([dyna_predict_model(
        dy/"checkpoints"/f"{ARM}_seed{seed}.pt",tst,prot,chem,prior,sd,device,sha(spec_path))
        for seed in SEEDS],axis=0)
    per_pred=np.mean(per_predictions,axis=0)
    if dy_pred.shape!=ytest.shape or per_pred.shape!=ytest.shape:raise RuntimeError("Wrong prediction shape")
    dyn=metrics(dy_pred,ytest)
    original=metrics(per_pred,ytest)
    clamp=metrics(np.maximum(per_pred,5.0),ytest)
    nofloor=~floor
    out={
        "status":"PREREGISTERED_STRICT_PROTEIN_MATCHED_TWO_SEED_INDEPENDENT_TEST",
        "scenario":args.scenario,"frozen_manifest_sha256":sha(spec_path),
        "original_DAVIS_Y_sha256":spec["source_Y_sha256"],
        "n_heldout_proteins":len(tst),"n_pairs":int(ytest.size),
        "n_test_floor_pairs":int(floor.sum()),"n_test_nonfloor_pairs":int(nofloor.sum()),
        "original_perceiver_git_commit":OFFICIAL_COMMIT,
        "official_by_seed_sha256":official_hashes,
        "dynacpi_checkpoint_hashes":{str(seed):sha(dy/"checkpoints"/f"{ARM}_seed{seed}.pt") for seed in SEEDS},
        "frozen_validation_replay":val_scores,
        "dynacpi_v5_two_seed":dyn,
        "perceiver_official_two_seed":original,
        "perceiver_official_two_seed_with_same_lower_clamp":clamp,
        "test_nonfloor_rmse":{
           "dynacpi":float(np.sqrt(np.mean((dy_pred-ytest)[nofloor]**2))),
           "perceiver":float(np.sqrt(np.mean((per_pred-ytest)[nofloor]**2)))
        },
        "dynacpi_minus_perceiver_macro_rmse":float(dyn["macro_rmse"]-original["macro_rmse"]),
        "dynacpi_minus_perceiver_pooled_rmse":float(dyn["pooled_rmse"]-original["pooled_rmse"]),
        "interpretation":"One predeclared strict protein-neighbor split. Not curated kinase families, not proof of universal generalization; retain negative results."
    }
    (dest/"FINAL_SAME_SPLIT_METRICS.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2),flush=True)
    print("BLIND_STRICT_PROTEIN_MATCHED_TWO_SEED_TEST_COMPLETE",flush=True)

if __name__=="__main__":main()
