"""One-time, guarded cold-chemistry DAVIS test comparison.

Requires completed, val-selected official PerceiverCPI seeds 2931 + 2932,
frozen DynaCPI low-rank checkpoints and exact preregistered split hashes.
Reconstructs DynaCPI VALIDATION predictions before reading any test labels.
NO test driven model fitting, checkpoint selection or protocol changes.
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
from train_full_davis_v5 import (PREP,normalize_protein,
    LowRankDynaCPI,grid_prediction,metrics,sha,load_local_checkpoint)
from train_preregistered_dynacpi_cold_chem_20261110 import (
    OUT,SOURCE,ARMS,SEEDS,NAME,build_chemical_prior)
HERE=Path(__file__).resolve().parent
UPSTREAM=HERE/"matched_perceivercpi_20261110"/"upstream"
PY=Path.home()/"dynacpi_v4"/".venv"/"Scripts"/"python.exe"
COMMIT="755f076ef51099a90332fd1ec8437f94e3750c0b"

def official_ckpt(specdir,seed):
    run=specdir/f"perceiver_official_seed{seed}_30ep"
    ck=run/"fold_0"/"model_0"/"model.pt"
    if not ck.exists():raise RuntimeError(f"Official seed {seed} not trained")
    log=(run/"verbose.log").read_text(encoding="utf-8",errors="replace")
    if "Epoch 29" not in log or "Overall test mse" not in log or "Elapsed time" not in log:
        raise RuntimeError(f"Official seed {seed} 30 epochs NOT COMPLETED")
    info=json.loads((run/"args.json").read_text(encoding="utf-8"))
    if int(info["epochs"])!=30 or int(info["pytorch_seed"])!=seed or int(info["seed"])!=seed:
        raise RuntimeError("Official seed or epoch budget changed")
    for field,fn in (("data_path","train.csv"),("separate_val_path","validation.csv"),
                     ("separate_test_path","validation_proxy_test.csv")):
        if Path(info[field]).resolve()!=(specdir/fn).resolve():
            raise RuntimeError("Official training input diverges from predeclared CSV")
    return ck

def read_predict(path,truthrows,np_,nc):
    with path.open("r",encoding="utf-8",newline="") as fh:rows=list(csv.DictReader(fh))
    if len(rows)!=np_*nc or len(truthrows)!=np_*nc:
        raise RuntimeError("Official prediction row count mismatch")
    for i,(a,b) in enumerate(zip(rows,truthrows)):
        smi=a["smiles"]
        try:
            obj=ast.literal_eval(smi)
            if isinstance(obj,list) and len(obj)==1:smi=obj[0]
        except (ValueError,SyntaxError):
            pass
        if smi!=b["smiles"]:raise RuntimeError("Prediction SMILES alignment mismatch "+str(i))
    return np.asarray([float(r["label"]) for r in rows],dtype=np.float64).reshape(np_,nc)

def run_dyna(ck_path,protein_idx,prot,chem,prior,sd,comp_idx,split_sha,expected_chem_ids,device):
    with torch.serialization.safe_globals([type(np.dtype(np.int32))]):
        ck=load_local_checkpoint(ck_path)
    if ck["split_sha256"]!=split_sha or ck["prepared_features_sha256"]!=sha(PREP):
        raise RuntimeError("DynaCPI checkpoint's split/features differ")
    if ck["variant"]!=NAME or int(ck["seed"]) not in SEEDS:
        raise RuntimeError("DynaCPI checkpoint model variant differs")
    if not np.array_equal(np.asarray(ck["trained_compound_ids"],dtype=int),expected_chem_ids):
        raise RuntimeError("DynaCPI checkpoint appears to have seen a heldout compound")
    if not np.allclose(ck["chemical_all_prior"],prior.cpu().numpy(),atol=1e-6):
        raise RuntimeError("DynaCPI chemical prior differs from training-only reconstruction")
    if abs(float(ck["label_sd"])-sd)>1e-6:raise RuntimeError("Training label SD mismatch")
    model=LowRankDynaCPI().to(device)
    model.load_state_dict(ck["state_dict"],strict=True)
    matrix=grid_prediction(model,protein_idx,prot,chem,prior,sd,device)[:,comp_idx]
    del model
    return matrix

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenario",required=True,choices=sorted(ARMS))
    a=parser.parse_args()
    scenario=a.scenario
    root=OUT/scenario
    final=root/"MATCHED_TWO_SEED_CHEM_BLIND_TEST_ONCE"
    if final.exists():raise RuntimeError("Already have a once-only result; do not overwrite")
    manifest=root/"SPLIT_LOCKED.json"
    lock=json.loads(manifest.read_text(encoding="utf-8"))
    suite=json.loads((OUT/"SUITE_LOCKED.json").read_text(encoding="utf-8"))
    declared=next(r for r in suite["scenarios"] if r["scenario"]==scenario)
    if lock["category"]!=ARMS[scenario] or sha(manifest)!=declared["split_manifest_sha256"]:
        raise RuntimeError("Frozen declared test split modified")
    if not lock["test_labels_not_exported"] or not lock["selection_locked_before_test"]:
        raise RuntimeError("Test has not been sealed by pre-registration")
    for fn,h in lock["input_csv_sha256"].items():
        if sha(root/fn)!=h:raise RuntimeError("Train/validation CSV changed")
    if lock["input_csv_sha256"]["validation.csv"]!=lock["input_csv_sha256"]["validation_proxy_test.csv"]:
        raise RuntimeError("Official training test is not val proxy")
    if sha(PREP)!=lock["source_features_sha256"] or sha(SOURCE/"source"/"Y")!=lock["source_Y_sha256"]:
        raise RuntimeError("Source affinity data changed")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=UPSTREAM,text=True).strip()!=COMMIT:
        raise RuntimeError("Official upstream source SHA changed")
    if subprocess.check_output(["git","diff","--name-only","--","*.py"],cwd=UPSTREAM,text=True).strip():
        raise RuntimeError("Original PerceiverCPI source modified")
    # Important: all official checkpoints must already have been chosen via
    # the TRAIN/VAL protocol before even loading the test affinity matrix.
    official={seed:official_ckpt(root,seed) for seed in SEEDS}
    dynrun=root/"dynacpi_cold_chem_fixed_train"
    frozen=json.loads((dynrun/"VALIDATION_FROZEN.json").read_text(encoding="utf-8"))
    if frozen["status"] not in (
        "VALIDATION_CHECKPOINTS_FROZEN_NO_TEST",
        "VALIDATION_CHECKPOINTS_RECOVERED_AND_FROZEN_NO_TEST"):
        raise RuntimeError("DynaCPI validation freeze not complete")
    if not frozen["test_labels_never_loaded"]:
        raise RuntimeError("DynaCPI training label exposure guard missing")
    if frozen["split_sha256"]!=sha(manifest) or frozen["training_seeds"]!=list(SEEDS):
        raise RuntimeError("DynaCPI freeze and split differ")
    cks={seed:dynrun/"checkpoints"/f"{NAME}_seed{seed}.pt" for seed in SEEDS}
    if not all(p.exists() for p in cks.values()):
        raise RuntimeError("A DynaCPI checkpoint is missing")
    # Retrain = forbidden; replay frozen checkpoints on validation first.
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ids=lock["indices"]
    pt=np.asarray(ids["protein_train"],dtype=int)
    pv=np.asarray(ids["protein_val"],dtype=int)
    ps=np.asarray(ids["protein_test"],dtype=int)
    ct=np.asarray(ids["compound_train"],dtype=int)
    cv=np.asarray(ids["compound_val"],dtype=int)
    cs=np.asarray(ids["compound_test"],dtype=int)
    assert not set(ct)&set(cv) and not set(ct)&set(cs)
    if scenario.startswith("double"):
        assert not set(pt)&set(pv) and not set(pt)&set(ps)
    with np.load(PREP,allow_pickle=False) as z:
        prot_features=z["pfeat"].astype(np.float32)
        chem_features=z["bits"].astype(np.float32)
        smiles=z["smiles"].astype(str)
        ytrain=z["y"][np.ix_(pt,ct)].astype(np.float32)
        yval=z["y"][np.ix_(pv,cv)].astype(np.float32)
    mean=ytrain.mean(axis=0,dtype=np.float64)
    anchor,_,_=build_chemical_prior(chem_features,ct,mean)
    protein_scaled,_,_=normalize_protein(prot_features,pt)
    device="cuda" if torch.cuda.is_available() else "cpu"
    prot=torch.as_tensor(protein_scaled["centered"],device=device,dtype=torch.float32)
    chem=torch.as_tensor(chem_features,device=device,dtype=torch.float32)
    prior=torch.as_tensor(anchor,device=device,dtype=torch.float32)
    label_sd=float(ytrain.std())
    vpred=np.mean([
        run_dyna(cks[seed],pv,prot,chem,prior,label_sd,cv,sha(manifest),ct,device)
        for seed in SEEDS],axis=0)
    replay=metrics(vpred,yval)
    for field in ["macro_rmse","pooled_rmse"]:
        if abs(replay[field]-frozen["final_validation"][field])>1e-6:
            raise RuntimeError("Previous DynaCPI validation replay mismatch; REFUSE TEST")
    # Sealed TEST labels are now accessed for FIRST TIME, after both models,
    # all four selected checkpoints and the source manifests have passed.
    with np.load(PREP,allow_pickle=False) as z:
        ytest=z["y"][np.ix_(ps,cs)].astype(np.float64)
        floor=z["censored"][np.ix_(ps,cs)].astype(bool)
    seqs=list(json.loads((SOURCE/"source"/"proteins.txt").read_text(encoding="utf-8")).values())
    rows=[{"smiles":str(smiles[int(j)]),"sequence":seqs[int(i)],
           "label":repr(float(ytest[ii,jj]))}
          for ii,i in enumerate(ps) for jj,j in enumerate(cs)]
    if len(rows)!=lock["test_pairs_not_exported"]:
        raise RuntimeError("Test cardinality unexpectedly changed")
    final.mkdir(parents=True,exist_ok=False)
    tcsv=final/"test_exact_once.csv"
    with tcsv.open("w",encoding="utf-8",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=["smiles","sequence","label"])
        w.writeheader();w.writerows(rows)
    env=os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2"})
    official_preds=[]
    official_provenance={}
    for seed in SEEDS:
        dest=final/f"official_seed{seed}_predictions.csv"
        cmd=[str(PY),"-u","predict.py","--test_path",str(tcsv),
            "--preds_path",str(dest),"--checkpoint_path",str(official[seed]),
            "--batch_size","128","--num_workers","0"]
        (final/f"official_seed{seed}_command.json").write_text(json.dumps(cmd,indent=2))
        with (final/f"official_seed{seed}.log").open("w",encoding="utf-8") as log:
            result=subprocess.run(cmd,cwd=UPSTREAM,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        if result.returncode:raise RuntimeError(f"Official seed {seed} prediction failed")
        official_preds.append(read_predict(dest,rows,len(ps),len(cs)))
        official_provenance[str(seed)]={"checkpoint_sha256":sha(official[seed]),"prediction_sha256":sha(dest)}
    official_mean=np.mean(official_preds,axis=0)
    dyna=np.mean([
        run_dyna(cks[seed],ps,prot,chem,prior,label_sd,cs,sha(manifest),ct,device)
        for seed in SEEDS],axis=0)
    dyn=metrics(dyna,ytest)
    orig=metrics(official_mean,ytest)
    clamped=metrics(np.maximum(official_mean,5),ytest)
    if not np.isfinite(dyna).all() or not np.isfinite(official_mean).all():
        raise RuntimeError("Nonnumeric model outputs")
    np.savez_compressed(final/"raw_predictions_private.npz",
       prediction_dynacpi=dyna,prediction_perceiver=official_mean,
       truth=ytest,protein_indices=ps,compound_indices=cs)
    nonfloor=~floor
    scores={
       "status":"FIRST_LOCKED_COLD_CHEM_TEST_TWO_SEED_BOTH_MODELS",
       "split":scenario,"category":lock["category"],
       "frozen_split_sha256":sha(manifest),
       "n_test_proteins":len(ps),"n_test_compounds":len(cs),
       "n_pairs":int(ytest.size),"n_nonfloor_pairs":int(nonfloor.sum()),
       "official_commit":COMMIT,"official_checkpoint_and_prediction_hashes":official_provenance,
       "dynacpi_checkpoint_sha256":{str(seed):sha(cks[seed]) for seed in SEEDS},
       "validation_reproduced":replay,
       "dynacpi_2seed":dyn,"official_perceiver_2seed":orig,
       "perceiver_same_floor_clamp":clamped,
       "dynacpi_minus_official_macro_rmse":float(dyn["macro_rmse"]-orig["macro_rmse"]),
       "dynacpi_minus_official_pooled_rmse":float(dyn["pooled_rmse"]-orig["pooled_rmse"]),
       "nonfloor_rmse":{
           "dynacpi":float(np.sqrt(np.mean((dyna-ytest)[nonfloor]**2))) if nonfloor.any() else None,
           "perceiver":float(np.sqrt(np.mean((official_mean-ytest)[nonfloor]**2))) if nonfloor.any() else None
       },
       "interpretation":"Matched one-split two-seed cold-chemistry result; all identities/scaffolds disjoint, not proof of universal or external generalization; independent dataset confirmation still required."
    }
    (final/"FINAL_LOCKED_TEST_METRICS.json").write_text(json.dumps(scores,indent=2),encoding="utf-8")
    print(json.dumps(scores,indent=2),flush=True)
    print("MATCHED_TWO_SEED_COLD_CHEM_EVALUATION_COMPLETE",flush=True)

if __name__=="__main__":main()
