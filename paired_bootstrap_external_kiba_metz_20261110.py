"""Predeclared paired uncertainty analysis for six external KIBA/Metz tests.

Sample PROTEIN sequence-similarity groups for protein-cold, compounds by exact
Murcko scaffold clusters for compound-cold, and independent crossed group draws
for double-cold. Identical resampled observations in both models. All 5,000
draws are descriptive cluster-bootstrap uncertainty, NOT clinical or universal
superiority proof. --smoke uses ONLY synthetic arrays, NO test labels.

This script MUST run only after corresponding once-only external test results
exist and validates their saved provenance/hashes first.
"""
from __future__ import annotations
import argparse,json
import numpy as np
from pathlib import Path
from prepare_external_kiba_metz_frozen_benchmarks_20261110 import (
    OUT,SOURCES,protein_clusters,scaffold_clusters,file_sha)
from train_external_sparse_dynacpi_v5_20261110 import metrics
DRAWS=5000
RNG_SEED=20261110
SCENARIOS=("protein_cold","compound_cold","double_cold")
DATASETS=("metz","kiba")
ROOT=OUT/"ALL_SIX_EXTERNAL_SAME_SPLIT_TESTS_ONCE"

def resample_pairs(pred_d,pred_p,y,protein_ids,compound_ids,protein_groups,compound_groups,
                   scenario,draws=DRAWS,seed=RNG_SEED):
    if pred_d.ndim!=1 or len(y)!=len(pred_d) or len(pred_p)!=len(y):
        raise ValueError("Sparse paired predictions must have same 1D shape")
    if not np.isfinite(y).all() or not np.isfinite(pred_d).all() or not np.isfinite(pred_p).all():
        raise ValueError("Invalid numeric paired scores")
    pred_d=pred_d.astype(np.float64)
    pred_p=pred_p.astype(np.float64)
    y=y.astype(np.float64)
    protein_ids=np.asarray(protein_ids,dtype=np.int32)
    compound_ids=np.asarray(compound_ids,dtype=np.int32)
    protein_groups=np.asarray(protein_groups,dtype=np.int32)
    compound_groups=np.asarray(compound_groups,dtype=np.int32)
    if len(protein_ids)!=len(y) or len(compound_ids)!=len(y):
        raise ValueError("Pair reference length mismatch")
    rprotein=protein_groups[protein_ids]
    rcompound=compound_groups[compound_ids]
    u_protein=np.unique(rprotein)
    u_compound=np.unique(rcompound)
    p_indices=np.searchsorted(u_protein,rprotein)
    c_indices=np.searchsorted(u_compound,rcompound)
    pid_unique, pid_local=np.unique(protein_ids,return_inverse=True)
    nprot=len(pid_unique)
    prot_group_for_pid=protein_groups[pid_unique]
    gp_for_pid=np.searchsorted(u_protein,prot_group_for_pid)
    err_d=(pred_d-y)**2
    err_p=(pred_p-y)**2
    rng=np.random.default_rng(seed)
    combined=[]
    failed_no_test_obs=0
    for iteration in range(draws):
        protein_mult=np.ones(len(u_protein),dtype=np.int32)
        chemical_mult=np.ones(len(u_compound),dtype=np.int32)
        if scenario in ("protein_cold","double_cold"):
            protein_mult=np.bincount(rng.integers(0,len(u_protein),size=len(u_protein)),
                                    minlength=len(u_protein)).astype(np.int32)
        if scenario in ("compound_cold","double_cold"):
            chemical_mult=np.bincount(rng.integers(0,len(u_compound),size=len(u_compound)),
                                     minlength=len(u_compound)).astype(np.int32)
        row_mult=protein_mult[p_indices].astype(np.float64)*chemical_mult[c_indices].astype(np.float64)
        if not row_mult.any():
            failed_no_test_obs+=1
            continue
        # Pooled paired RMSE counts all observed rows weighted by sampled cluster
        # multiplicities (identical draw for both original model predictions).
        pm=float(np.sqrt(np.average(err_d,weights=row_mult))-
                 np.sqrt(np.average(err_p,weights=row_mult)))
        # Per-protein macro RMSE computes chemical-resampling-weighted MSE for
        # each protein, then weights protein contributions by group draw count.
        per_row_cweight=chemical_mult[c_indices].astype(np.float64)
        denom=np.bincount(pid_local,weights=per_row_cweight,minlength=nprot)
        valid=(denom>0)&(protein_mult[gp_for_pid]>0)
        if not valid.any():
            failed_no_test_obs+=1
            continue
        v_d=np.bincount(pid_local,weights=err_d*per_row_cweight,minlength=nprot)
        v_p=np.bincount(pid_local,weights=err_p*per_row_cweight,minlength=nprot)
        macro_weights=protein_mult[gp_for_pid][valid].astype(np.float64)
        md=float(np.average(np.sqrt(v_d[valid]/denom[valid]),weights=macro_weights)-
                 np.average(np.sqrt(v_p[valid]/denom[valid]),weights=macro_weights))
        combined.append((md,pm))
    values=np.asarray(combined,dtype=np.float64).reshape(-1,2)
    if len(values)<int(.95*draws):
        raise RuntimeError("Too few valid bootstrap draws; double-cold too sparse for this protocol")
    return {
      "n_requested_draws":draws,
      "n_valid_draws":len(values),
      "n_empty_draws":failed_no_test_obs,
      "fixed_bootstrap_seed":seed,
      "bootstrap_unit":{
        "protein_cold":"connected_protein_3mer_neighbor_groups_035",
        "compound_cold":"exact_chemical_murcko_scaffold_groups",
        "double_cold":"crossed_protein_groups_and_chemical_scaffold_groups"
      }[scenario],
      "n_protein_groups":int(len(u_protein)),
      "n_compound_groups":int(len(u_compound)),
      "delta_sign":"DynaCPI - original PerceiverCPI, negative favors DynaCPI",
      "macro_delta_95pct_interval":np.quantile(values[:,0],[.025,.975]).tolist(),
      "pooled_delta_95pct_interval":np.quantile(values[:,1],[.025,.975]).tolist(),
      "fraction_macro_resamples_dynacpi_better":float(np.mean(values[:,0]<0)),
      "fraction_pooled_resamples_dynacpi_better":float(np.mean(values[:,1]<0)),
      "interpretation":"Descriptive paired cluster uncertainty from one split, not an independent statistical confirmation."
    }

def run_one(dataset,scenario,root=ROOT):
    folder=root/(dataset+"_"+scenario)
    f=folder/"TEST_RESULTS.json"
    raw=folder/"paired_raw_predictions_private.npz"
    out=folder/"PAIRED_CLUSTER_BOOTSTRAP.json"
    if out.exists():raise RuntimeError("Uncertainty output already exists; refuse resampling overwrite")
    if not f.exists() or not raw.exists():raise RuntimeError("External sealed test has not been evaluated")
    report=json.loads(f.read_text(encoding="utf-8"))
    if report["status"]!="SAME_SPLIT_TWO_SEED_ONCE_TEST_COMPLETE":
        raise RuntimeError("Pairwise test status invalid")
    if report["dataset"]!=dataset or report["scenario"]!=scenario:
        raise RuntimeError("Test output mislabeled")
    if report["paired_raw_pred_sha256"]!=file_sha(raw):
        raise RuntimeError("Previously frozen raw predictions hash changed")
    split_path=OUT/dataset/scenario/"SPLIT_LOCKED.json"
    spec=json.loads(split_path.read_text(encoding="utf-8"))
    if report["frozen_split_sha256"]!=file_sha(split_path) or not spec["test_labels_not_exported"]:
        raise RuntimeError("Frozen source split changed after test")
    audit=json.loads((OUT/dataset/"SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    row=next(x for x in audit["splits"] if x["name"]==scenario)
    if row["manifest_sha256"]!=file_sha(split_path):
        raise RuntimeError("Predeclared manifest SHA changed")
    with np.load(raw,allow_pickle=False) as z:
        y=z["labels"].astype(np.float64)
        d=z["pred_dynacpi"].astype(np.float64)
        p=z["pred_original_perceiver"].astype(np.float64)
        pids=z["protein_index"].astype(int)
        cids=z["compound_index"].astype(int)
    if len(y)!=spec["n_pairs"]["test"] or len(y)!=report["n_test_pairs"]:
        raise RuntimeError("Raw paired test pairs differ from preregistered count")
    source=SOURCES[dataset]["root"]
    if dataset=="kiba":
        sequences=list(json.loads((source/"proteins.txt").read_text(encoding="utf-8")).values())
        molecules=list(json.loads((source/"ligands_can.txt").read_text(encoding="utf-8")).values())
    else:
        import pandas as pd
        ptable=pd.read_csv(source/"prot_info.csv").sort_values("index")
        ctable=pd.read_csv(source/"drug_info.csv").sort_values("index")
        sequences=ptable.sequence.astype(str).tolist()
        molecules=ctable.smiles.astype(str).tolist()
    pg,_=protein_clusters(sequences)
    cg=scaffold_clusters(molecules)
    # Metric reconstruction check prevents silently pairing shifted source rows.
    a=metrics(d,y,pids);b=metrics(p,y,pids)
    for label,source_metric,expect in [
       ("Dyna macro",a["macro_rmse"],report["dynacpi_2seed"]["macro_rmse"]),
       ("Dyna pooled",a["pooled_rmse"],report["dynacpi_2seed"]["pooled_rmse"]),
       ("Original macro",b["macro_rmse"],report["official_perceiver_2seed"]["macro_rmse"]),
       ("Original pooled",b["pooled_rmse"],report["official_perceiver_2seed"]["pooled_rmse"])
    ]:
       if abs(source_metric-expect)>1e-8:raise RuntimeError(label+" does not replay")
    boot=resample_pairs(d,p,y,pids,cids,pg,cg,scenario)
    result={
      "status":"PAIRWISE_CLUSTER_BOOTSTRAP_DESCRIPTIVE",
      "dataset":dataset,"scenario":scenario,
      "n_test_pairs":len(y),
      "original_test_result_sha256":file_sha(f),
      "original_raw_predictions_sha256":file_sha(raw),
      "frozen_split_sha256":file_sha(split_path),
      "point_estimate_macro_delta":a["macro_rmse"]-b["macro_rmse"],
      "point_estimate_pooled_delta":a["pooled_rmse"]-b["pooled_rmse"],
      "bootstrap":boot
    }
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)

def smoke():
    # Correctly dominates in all three correlated bootstrap protocols.
    y=np.linspace(3,7,48,dtype=np.float64)
    d=y+.15
    p=y+.5
    proteins=np.repeat(np.arange(8),6)
    compounds=np.tile(np.arange(6),8)
    pg=np.array([0,0,1,2,3,4,4,5],dtype=np.int32)
    cg=np.array([0,1,1,2,3,4],dtype=np.int32)
    for case in SCENARIOS:
        score=resample_pairs(d,p,y,proteins,compounds,pg,cg,case,draws=250,seed=4321)
        if score["macro_delta_95pct_interval"][1]>=0 or score["pooled_delta_95pct_interval"][1]>=0:
            raise AssertionError("Synthetic paired dominance must yield negative delta")
    print("PASS_EXTERNAL_CLUSTER_BOOTSTRAP_SYNTHETIC_NO_TEST_DATA_READ",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=DATASETS)
    ap.add_argument("--scenario",choices=SCENARIOS)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--individual",action="store_true",
        help="Read the already once-scored incremental result without re-running the test")
    a=ap.parse_args()
    if a.smoke:smoke()
    elif a.dataset and a.scenario:
        root=(OUT/"INDIVIDUAL_EXTERNAL_SAME_SPLIT_TESTS_ONCE_20261111") if a.individual else ROOT
        run_one(a.dataset,a.scenario,root=root)
    else:ap.error("Select --smoke or --dataset and --scenario")
