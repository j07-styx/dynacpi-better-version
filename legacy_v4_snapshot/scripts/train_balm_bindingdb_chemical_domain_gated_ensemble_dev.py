"""Chemical-domain aware ridge/boosted-CPI gating, fitted and selected on TRAIN only.

Protein-group-disjoint GroupKFold avoids using the inspected development set for
gate selection. Reserved test labels and predictions are never requested.
"""
import json,time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from rdkit.Chem import rdFingerprintGenerator

from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,metric,fingerprint_rows)
from scripts.ablate_balm_bindingdb_ensemble_interaction_controls_dev import evaluate

OUT=ROOT/"research/balm_bindingdb_chemical_domain_gated_ensemble_traincv_dev.json"
MODES=("fixed25","low_similarity_higher_boost","high_similarity_higher_boost")
SEED=20261108

def nearest_chemical_cosine(train,valid,gen,cache,batch=192):
    """Maximum Morgan-bit cosine to training molecules; no labels used.

    Use unique SMILES to avoid duplicate compound evaluation while retaining
    molecule-level nearest neighbors. All fingerprints normalized per molecule.
    """
    known=sorted(set(r["smiles"] for r in train))
    queries=sorted(set(r["smiles"] for r in valid))
    x=fingerprint_rows([{"smiles":s} for s in known],gen,cache)
    y=fingerprint_rows([{"smiles":s} for s in queries],gen,cache)
    scores={}
    for start in range(0,len(queries),batch):
        cos=(y[start:start+batch]@x.T).max(axis=1).toarray().ravel()
        assert np.isfinite(cos).all()
        scores.update(zip(queries[start:start+batch],np.clip(cos,0.,1.)))
    return np.asarray([scores[r["smiles"]] for r in valid],dtype=np.float64)

def blend(ridge,booster,nearest,mode):
    assert np.shape(ridge)==np.shape(booster)==np.shape(nearest)
    if mode=="fixed25":
        weight=np.full_like(nearest,.25)
    elif mode=="low_similarity_higher_boost":
        weight=.4-.3*nearest
    elif mode=="high_similarity_higher_boost":
        weight=.1+.3*nearest
    else:
        raise ValueError(mode)
    assert (weight>=.0999).all() and (weight<=.4001).all()
    return (1-weight)*ridge+weight*booster

def performance(tr,valid,gen,fp_cache,chemcache,protcache):
    prediction=evaluate(tr,valid,gen,fp_cache,chemcache,protcache)
    ridge=prediction["ridge"]
    # Extract original joint boosted predictor from existing 75/25 reference.
    booster=(prediction["ridge_plus_joint_boost"]-.75*ridge)/.25
    similarity=nearest_chemical_cosine(tr,valid,gen,fp_cache)
    y=np.asarray([r["label"] for r in valid],dtype=np.float64)
    group=[r["group"] for r in valid]
    results={"ridge":metric(y,ridge,group)}
    for mode in MODES:
        results[mode]=metric(y,blend(ridge,booster,similarity,mode),group)
    return results,{
        "count":len(similarity),
        "mean":float(similarity.mean()),
        "fraction_exact_or_near_exact":float(np.mean(similarity>=.999)),
        "fraction_cosine_below_0p5":float(np.mean(similarity<.5))
    }

def main():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.time()
    parts=get_train_dev()
    train=parts["train"]
    dev=parts["inner_dev"]
    assert len(train)==15538 and len(dev)==3891
    groups=np.asarray([r["group"] for r in train])
    assert set(groups).isdisjoint({r["group"] for r in dev})
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fp_cache={};chemcache={};protcache={}
    folds=[]
    # The non-overlapping held-out components mean each original training pair
    # contributes to one and only one fold validation.
    splitter=GroupKFold(n_splits=4)
    for fold,(itr,iva) in enumerate(splitter.split(groups,groups=groups),1):
        tr=collapse_training([train[int(i)] for i in itr])
        val=[train[int(i)] for i in iva]
        assert set(r["group"] for r in tr).isdisjoint({r["group"] for r in val})
        scores,sims=performance(tr,val,gen,fp_cache,chemcache,protcache)
        item={
            "fold":fold,"train_unique":len(tr),"holdout_rows":len(val),
            "heldout_components":len(set(r["group"] for r in val)),
            "chemical_similarity":sims,"metrics":scores
        }
        folds.append(item)
        print("FOLD",fold,json.dumps(item),flush=True)
    averaged={
        name:{key:float(np.mean([f["metrics"][name][key] for f in folds]))
              for key in ("pooled_rmse","component_macro_rmse")}
        for name in ("ridge",)+MODES
    }
    ref=averaged["fixed25"]
    ranked=sorted([
        {"mode":mode,
         "pooled_cv_mean":averaged[mode]["pooled_rmse"],
         "macro_cv_mean":averaged[mode]["component_macro_rmse"],
         "relative_joint":.5*(
             averaged[mode]["pooled_rmse"]/ref["pooled_rmse"]+
             averaged[mode]["component_macro_rmse"]/ref["component_macro_rmse"])}
        for mode in MODES
    ],key=lambda x:x["relative_joint"])
    eligible=[x for x in ranked
              if x["pooled_cv_mean"]<=ref["pooled_rmse"] and
                 x["macro_cv_mean"]<=ref["component_macro_rmse"]]
    selected=eligible[0]["mode"] if eligible else "fixed25"
    print("TRAIN_CV_LOCKED",selected,json.dumps(averaged),flush=True)
    full=collapse_training(train)
    devscores,devsims=performance(full,dev,gen,fp_cache,chemcache,protcache)
    out={
        "status":"BALM_TRAIN_ONLY_GROUPK_FOLD_CHEMICAL_SIMILARITY_GATING_EXPLORATORY_DEV_NO_TEST",
        "architecture":"Ridge(Chemical Morgan+protein TFIDF) + molecule-similarity gated ordinary joint chemical/protein descriptor booster",
        "gating_formula":{
            "fixed25":"w=0.25",
            "low_similarity_higher_boost":"w=0.40-0.30*nearest_morgan_cosine",
            "high_similarity_higher_boost":"w=0.10+0.30*nearest_morgan_cosine",
            "prediction":"(1-w)*ridge + w*joint_booster"},
        "fold_count":len(folds),"cv_type":"GroupKFold(4), disjoint protein graph components, no overlap of heldout protein groups",
        "cv_folds":folds,"cv_mean":averaged,"cv_ranked":ranked,
        "selected_on_train_cv":selected,
        "dev_metrics_transparent_predeclared":devscores,
        "dev_locked_selected":devscores[selected],
        "dev_similarity_summary":devsims,
        "train_original_rows":len(train),
        "train_unique_protein_compound":len(full),
        "dev_rows":len(dev),
        "seconds":round(time.time()-start,2),
        "caveats":[
            "BALM target groups are approximation from sequence 3-mer similarity rather than verified homology.",
            "Original source lacks assay provenance and has inconsistent identical-input affinity labels.",
            "Development partition previously inspected many times; exploratory and not independent.",
            "Gate candidates examined on train-CV only; prospective test labels not accessed or predicted.",
            "Each similarity uses train-only chemical fingerprints, never heldout labels or test molecules."
        ]
    }
    assert len(full)==12315
    OUT.write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print("FINAL",json.dumps({"selected":selected,
                                "cv_reference":ref,
                                "cv_selected":averaged[selected],
                                "dev_reference":devscores["fixed25"],
                                "dev_selected":devscores[selected],
                                "dev_ridge":devscores["ridge"],
                                "dev_similarity":devsims}),flush=True)

if __name__=="__main__":main()

