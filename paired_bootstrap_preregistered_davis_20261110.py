"""Predeclared paired bootstrap protocol for four sealed DAVIS generalization tests.

Model metrics/split outcome must already be frozen and saved. This script cannot
create splits, train models, select checkpoints or access unscored test labels.
Protein-cold: resample connected sequence-neighbor groups.
Compound-cold: resample unseen compound identities.
Double-cold: independently resample protein-neighbor groups and compounds.
Both predictions are resampled on EXACTLY matched rows/columns.
Intervals are DESCRIPTIVE under these proxy clustering assumptions, not proof
of causal superiority or external chemical/biological generalization.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_preregistered_davis_generalization_20261110 import (
    OUT,SOURCE,sequence_components,sha)
N_DRAWS=8000
RANDOM_SEED=20261110
SCENARIOS={
"protein_strict_20261113":("MATCHED_TWO_SEED_BLIND_TEST_ONCE","FINAL_SAME_SPLIT_METRICS.json"),
"protein_strict_20261114":("MATCHED_TWO_SEED_BLIND_TEST_ONCE","FINAL_SAME_SPLIT_METRICS.json"),
"compound_scaffold_20261115":("MATCHED_TWO_SEED_CHEM_BLIND_TEST_ONCE","FINAL_LOCKED_TEST_METRICS.json"),
"double_cold_20261116":("MATCHED_TWO_SEED_CHEM_BLIND_TEST_ONCE","FINAL_LOCKED_TEST_METRICS.json")
}

def grouped_resample_rows(groups,rng):
    uniq=np.unique(groups)
    group_to_rows=[np.flatnonzero(groups==g) for g in uniq]
    choose=rng.integers(0,len(uniq),size=len(uniq))
    return np.concatenate([group_to_rows[k] for k in choose])

def rmse_deltas(dyna,perceiver,truth,rows,columns):
    da=np.square(dyna[np.ix_(rows,columns)]-truth[np.ix_(rows,columns)])
    pb=np.square(perceiver[np.ix_(rows,columns)]-truth[np.ix_(rows,columns)])
    return (float(np.mean(np.sqrt(da.mean(axis=1)))-np.mean(np.sqrt(pb.mean(axis=1)))),
            float(np.sqrt(da.mean())-np.sqrt(pb.mean())))

def bootstrap(dyna,perceiver,truth,protein_groups,scenario,n_draws=N_DRAWS,seed=RANDOM_SEED):
    rng=np.random.default_rng(seed)
    nprot,nchem=truth.shape
    if dyna.shape!=truth.shape or perceiver.shape!=truth.shape:
        raise ValueError("Paired predictions and truth differ in shape")
    if protein_groups.shape!=(nprot,):
        raise ValueError("Protein groups have wrong shape")
    deltas=np.empty((n_draws,2),dtype=np.float64)
    prot_index=np.arange(nprot)
    chem_index=np.arange(nchem)
    for i in range(n_draws):
        if scenario=="protein":
            row=grouped_resample_rows(protein_groups,rng)
            col=chem_index
        elif scenario=="compound":
            row=prot_index
            col=rng.integers(0,nchem,size=nchem)
        elif scenario=="double":
            row=grouped_resample_rows(protein_groups,rng)
            col=rng.integers(0,nchem,size=nchem)
        else:
            raise ValueError("Unknown preregistered scenario")
        deltas[i]=rmse_deltas(dyna,perceiver,truth,row,col)
    return {
       "bootstrap_drawwise_delta":"DynaCPI RMSE minus original PerceiverCPI RMSE (negative favors DynaCPI)",
       "n_draws":int(n_draws),"fixed_random_seed":int(seed),
       "resample_type":{"protein":"protein_sequence_connected_groups",
                        "compound":"cold_compound_identities",
                        "double":"two_way_protein_sequence_group_plus_cold_compound_identities"}[scenario],
       "n_resample_independent_units":(
          {"n_protein_similarity_groups":len(np.unique(protein_groups))}
          if scenario=="protein" else
          {"n_unseen_compound_ids":int(nchem)} if scenario=="compound" else
          {"n_protein_similarity_groups":len(np.unique(protein_groups)),
           "n_unseen_compound_ids":int(nchem)}),
       "macro_rmse_delta_95pct_descriptive_interval":list(map(float,np.quantile(deltas[:,0],[.025,.975]))),
       "pooled_rmse_delta_95pct_descriptive_interval":list(map(float,np.quantile(deltas[:,1],[.025,.975]))),
       "fraction_of_draws_dyna_better_macro":float(np.mean(deltas[:,0]<0)),
       "fraction_of_draws_dyna_better_pooled":float(np.mean(deltas[:,1]<0))
    }

def evaluate_one(name):
    local,metric_filename=SCENARIOS[name]
    directory=OUT/name/local
    metric_path=directory/metric_filename
    predictions=directory/"raw_predictions_private.npz"
    result=directory/"PAIRED_BOOTSTRAP_DESCRIPTIVE.json"
    if result.exists():raise RuntimeError("Do not overwrite a completed resampling result")
    if not metric_path.exists() or not predictions.exists():
        raise RuntimeError("Original preregistered paired TEST has not completed")
    record=json.loads(metric_path.read_text(encoding="utf-8"))
    declared=json.loads((OUT/name/"SPLIT_LOCKED.json").read_text(encoding="utf-8"))
    status=record["status"]
    if status not in ("PREREGISTERED_STRICT_PROTEIN_MATCHED_TWO_SEED_INDEPENDENT_TEST",
                     "FIRST_LOCKED_COLD_CHEM_TEST_TWO_SEED_BOTH_MODELS"):
        raise RuntimeError("Test result not a permitted frozen paired comparison")
    if declared["scenario"]!=name:
        raise RuntimeError("Wrong locked split")
    suite=json.loads((OUT/"SUITE_LOCKED.json").read_text(encoding="utf-8"))
    expected=next(x for x in suite["scenarios"] if x["scenario"]==name)
    if expected["split_manifest_sha256"]!=sha(OUT/name/"SPLIT_LOCKED.json"):
        raise RuntimeError("Splits were changed after preregistration")
    category=declared["category"]
    with np.load(predictions,allow_pickle=False) as z:
        dyna=z["prediction_dynacpi"].astype(np.float64)
        perceiver=z["prediction_perceiver"].astype(np.float64)
        truth=z["truth"].astype(np.float64)
        proteins=z["protein_indices"].astype(int)
        compounds=z["compound_indices"].astype(int)
    if truth.shape!=(len(proteins),len(compounds)) or truth.size!=record["n_pairs"]:
        raise RuntimeError("Raw test output shape differs from registered pair count")
    if not np.array_equal(proteins,np.asarray(declared["indices"]["protein_test"],dtype=int)):
        raise RuntimeError("Test proteins differ from frozen roles")
    if not np.array_equal(compounds,np.asarray(declared["indices"]["compound_test"],dtype=int)):
        raise RuntimeError("Test compounds differ from frozen roles")
    if not np.isfinite(dyna).all() or not np.isfinite(perceiver).all() or not np.isfinite(truth).all():
        raise RuntimeError("Invalid values in pair prediction/truth")
    seq=list(json.loads((SOURCE/"source"/"proteins.txt").read_text(encoding="utf-8")).values())
    groups,_=sequence_components(seq,.35)
    test_groups=groups[proteins]
    # Match preregistered original raw checkpoint metrics to within numerical rounding.
    if category=="protein":
        dyn_score=record["dynacpi_v5_two_seed"]
        per_score=record["perceiver_official_two_seed"]
    else:
        dyn_score=record["dynacpi_2seed"]
        per_score=record["official_perceiver_2seed"]
    allrows=np.arange(len(proteins));allcols=np.arange(len(compounds))
    macro,pooled=rmse_deltas(dyna,perceiver,truth,allrows,allcols)
    if abs(macro-(dyn_score["macro_rmse"]-per_score["macro_rmse"]))>1e-7:
        raise RuntimeError("Macro delta does not reproduce original frozen metric")
    if abs(pooled-(dyn_score["pooled_rmse"]-per_score["pooled_rmse"]))>1e-7:
        raise RuntimeError("Pooled delta does not reproduce original frozen metric")
    completed={
       "status":"POST_EVALUATION_DESCRIPTIVE_PAIRED_RESAMPLING",
       "scenario":name,
       "category":category,
       "frozen_split_sha256":sha(OUT/name/"SPLIT_LOCKED.json"),
       "source_paired_test_metric_file_sha256":sha(metric_path),
       "source_paired_raw_predictions_sha256":sha(predictions),
       "n_proteins":len(proteins),"n_compounds":len(compounds),
       "n_heldout_pairs":int(truth.size),
       "original_macro_delta":macro,"original_pooled_delta":pooled,
       "original_compared_to_original_unclamped_perceiver":True,
       "bootstrap":bootstrap(dyna,perceiver,truth,test_groups,category),
       "limitations":"Resampling inference is descriptive. Correlated groups, repeated targets, unbalanced label floors, 13 novel chemicals or singleton similarity groups constrain interpretation. This does NOT establish universal generalization."
    }
    result.write_text(json.dumps(completed,indent=2),encoding="utf-8")
    print(json.dumps(completed,indent=2),flush=True)

def smoke():
    rng=np.random.default_rng(2931)
    truth=rng.normal(size=(8,5))
    dyna=truth+.1
    perceiver=truth+.5
    groups=np.array([0,0,1,2,3,4,5,5],dtype=int)
    for cat in ("protein","compound","double"):
        b=bootstrap(dyna,perceiver,truth,groups,cat,n_draws=200,seed=20261110)
        if b["macro_rmse_delta_95pct_descriptive_interval"][1]>=0:
            raise AssertionError(f"Perfect synthetic dominance should yield negative paired macro delta: {cat}")
        if b["pooled_rmse_delta_95pct_descriptive_interval"][1]>=0:
            raise AssertionError(f"Perfect synthetic dominance should yield negative paired pooled delta: {cat}")
    print("PASS_SYNTHETIC_BOOSTRAP_NO_DATA_ACCESS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--scenario",choices=sorted(SCENARIOS))
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if args.smoke:smoke()
    elif args.scenario:evaluate_one(args.scenario)
    else:p.error("Provide --scenario or --smoke")
