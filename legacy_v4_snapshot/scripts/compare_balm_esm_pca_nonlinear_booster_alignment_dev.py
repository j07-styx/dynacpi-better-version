"""Frozen ESM2-conditioned nonlinear CPI booster under alignment+scaffold cold CV.

This changes the NONLINEAR predictor itself: include train-fold PCA features
from frozen protein ESM2, rather than giving ESM only to the linear Ridge arm.
All model weights, PCA, and target scaling are fitted on each outer fold TRAIN.
No frozen development or prospective test records/predictions/labels are loaded.
"""
import argparse
import hashlib
import json
import time

import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from dynacpi.balm_public_data_dev import ROOT, get_train_only
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import (
    bootstrap, cache_feature_map, fit_predict as ridge_predict,
)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import (
    HIST_PARAMS, data_features,
)
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training, metric,
)
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import (
    scaffold_filter, scaffold_key,
)

RULES=("identity30_cover80","identity40_cover80")
PCA_DIMS=(16,32)
NAMES=("train_mean","ridge_esm_kmer","nested_reference_joint",
       "nested_esm_pca16_joint","nested_esm_pca32_joint",
       "nested_esm_pca32_no_aa")
W=.25

def embed_matrix(rows,features,hcache):
    for r in rows:
        seq=r["protein"]
        if seq not in hcache:
            hcache[seq]=hashlib.sha256(seq.encode()).hexdigest()
    return np.asarray([features[hcache[r["protein"]]] for r in rows],
                      dtype=np.float32)

def transform_train_only_pca(tr,va,features,hcache,n_components):
    """Fit on unique training proteins only; never fit on validation vectors."""
    distinct={r["protein"] for r in tr}
    unique_proteins=sorted(distinct,key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    train_unique=embed_matrix([{"protein":s} for s in unique_proteins],features,hcache)
    assert len(unique_proteins)>=n_components+2
    compressor=PCA(n_components=n_components,svd_solver="randomized",
                   random_state=2103,iterated_power=4)
    compressor.fit(train_unique)
    return (
        compressor.transform(embed_matrix(tr,features,hcache)).astype(np.float32),
        compressor.transform(embed_matrix(va,features,hcache)).astype(np.float32),
        {"training_unique_proteins":len(unique_proteins),
         "pca_dimensions":n_components,
         "training_variance_explained":float(sum(compressor.explained_variance_ratio_))}
    )

def booster_models(tr,va,features,ccache,pcache,hcache):
    chemical,composition=data_features(tr,ccache,pcache)
    cv,pv=data_features(va,ccache,pcache)
    y=np.asarray([r["label"] for r in tr],dtype=np.float64)
    joint=np.column_stack([chemical,composition])
    joint_va=np.column_stack([cv,pv])
    base=HistGradientBoostingRegressor(**HIST_PARAMS)
    base.fit(joint,y)
    result={"reference_joint":base.predict(joint_va)}
    pca_log={}
    for dim in PCA_DIMS:
        a,b,manifest=transform_train_only_pca(tr,va,features,hcache,dim)
        pca_log[str(dim)]=manifest
        extra=HistGradientBoostingRegressor(**HIST_PARAMS)
        extra.fit(np.column_stack([joint,a]),y)
        result[f"esm_pca{dim}_joint"]=extra.predict(np.column_stack([joint_va,b]))
        if dim==32:
            protein_esm=HistGradientBoostingRegressor(**HIST_PARAMS)
            protein_esm.fit(np.column_stack([chemical,a]),y)
            result["esm_pca32_no_aa"]=protein_esm.predict(np.column_stack([cv,b]))
    return result,pca_log

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--group-rule",choices=RULES,required=True)
    args=parser.parse_args()
    rule=args.group_rule
    OUT=ROOT/"research"/f"balm_esm_conditioned_booster_sw_{rule}_scaffold_cold_dev.json"
    if OUT.exists():
        raise FileExistsError(OUT)
    start=time.time()
    report=json.loads((ROOT/"research"/f"balm_esm8m_nested_sw_{rule}_murcko_scaffold_cold_traincv_dev.json").read_text())
    alignment=json.loads((ROOT/"research/balm_train_only_smith_waterman_protein_group_audit_20261108.json").read_text())
    features,manifest=cache_feature_map()
    original=get_train_only()
    assert len(original)==15538
    sha={r["protein"]:hashlib.sha256(r["protein"].encode()).hexdigest() for r in original}
    assigned=alignment["sequence_sha256_to_alignment_groups"][rule]
    rows=[dict(r,group=assigned[sha[r["protein"]]]) for r in original]
    groups=np.asarray([r["group"] for r in rows])
    scaffold={s:scaffold_key(s) for s in set(r["smiles"] for r in rows)}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fpcache={};ccache={};pcache={};hashcache={}
    sq={name:[] for name in NAMES}
    counts=[];folds=[];seen=set()
    for i,(itr,iva) in enumerate(GroupKFold(4).split(groups,groups=groups),1):
        held=[rows[int(j)] for j in iva]
        fit=collapse_training(scaffold_filter([rows[int(j)] for j in itr],held,scaffold))
        group_set=set(r["group"] for r in held)
        assert not group_set & seen
        seen.update(group_set)
        assert not group_set & {r["group"] for r in fit}
        assert not {scaffold[r["smiles"]] for r in held} & {scaffold[r["smiles"]] for r in fit}
        prior=report["outer_folds"][i-1]
        assert len(held)==prior["heldout_pairs"] and len(fit)==prior["fit_unique_pairs"]
        beta=prior["inherited_beta_selected_inside_this_outer_fold"]
        ridge=ridge_predict(fit,held,gen,fpcache,features,hashcache)["chemical_kmer_esm8m"]
        boosting,log=booster_models(fit,held,features,ccache,pcache,hashcache)
        mu=float(np.mean([r["label"] for r in fit]))
        y=np.asarray([r["label"] for r in held],dtype=np.float64)
        g=np.asarray([r["group"] for r in held])
        pred={
            "train_mean":np.full(len(held),mu),
            "ridge_esm_kmer":ridge,
        }
        for name,boost in boosting.items():
            full=(1-W)*ridge+W*boost
            pred[f"nested_{name}"]=mu+beta*(full-mu)
        assert set(pred)==set(NAMES),(set(pred),set(NAMES))
        metrics={name:metric(y,p,g) for name,p in pred.items()}
        for m in ("pooled_rmse","component_macro_rmse"):
            assert abs(metrics["nested_reference_joint"][m]-
                       prior["outer_metrics"]["nested_kmer_esm"][m])<3e-5
        for group in sorted(group_set):
            mask=g==group
            counts.append(int(mask.sum()))
            for name in NAMES:
                sq[name].append(float(np.square(y[mask]-pred[name][mask]).sum()))
        fold_log={
            "fold":i,"fit_unique_pairs":len(fit),"heldout_original_pairs":len(held),
            "heldout_alignment_components":len(group_set),"scaffold_overlap":0,
            "inherited_beta":beta,"pca_fit":log,"metrics":metrics,
        }
        folds.append(fold_log)
        print("FOLD",i,json.dumps({
            "n_train":len(fit),"heldout":len(held),"beta":beta,
            "pca_log":log,"metrics":metrics
        }),flush=True)
    assert seen==set(groups)
    assert sum(counts)==15538 and len(counts)==alignment["rules"][rule]["connected_components"]
    result=bootstrap(sq,counts,references=("train_mean","nested_reference_joint"))
    report_output={
        "status":"EXPLORATORY_TRAIN_ONLY_ESM2_CONDITIONED_HISTBOOSTER_ALIGNMENT_AND_SCAFFOLD_COLD_NO_DEV_OR_TEST",
        "group_rule":rule,"train_rows":len(rows),
        "alignment_component_count":len(seen),
        "experiment":"Control nonlinear booster trained on 21 molecule descriptors and 21 amino acid features versus same booster augmented with 16/32 train-fold-fitted PCA of frozen ESM2 protein vectors; 32-component variant excluding amino-acid composition.",
        "protein_pca_policy":"fit PCA exclusively on the unique proteins in each OUTER training fold; apply frozen projection to heldout proteins",
        "ridge":"Morgan2048+protein char kmer+frozen ESM2 320 dims, Ridge alpha1, ESM scale0.25",
        "ensemble":"fixed 75pct Ridge +25pct booster, beta selected from earlier train-only inner-cold folds",
        "hist_booster_hyperparameters":HIST_PARAMS,
        "esm_source_hash":manifest["cache_sha256"],
        "models":NAMES,
        "outer_folds":folds,"out_of_fold":result,
        "seconds":round(time.time()-start,2),
        "caveats":[
            "Only original frozen TRAIN records; no development/test labels, features, or predictions accessed.",
            "Alpha, number of PCA dimensions and variants are exploratory feature ablations, not confirmatory analysis.",
            "Protein local alignment group rules are proxy for homology, not ortholog/domain validation.",
            "Strict Bemis-Murcko core exclusion may retain structurally similar compounds with different cores.",
            "BALM original assay methods, source DOI and construct provenance absent, conflicting replicate labels.",
            "ESM2 8M 320-D mean-pooled features truncate 85 long TRAIN protein sequences at 1022 AA.",
            "Metrics and bootstrap intervals do not correct previous exploratory model selection.",
        ]
    }
    OUT.write_text(json.dumps(report_output,indent=2)+"\n",encoding="utf8")
    print("FINAL",json.dumps({
        "rule":rule,"pooled":result["pooled_rmse"],
        "macro":result["component_macro_rmse"],
        "vs_reference":{
            k:v for k,v in result["bootstrap"]["comparisons"].items()
            if k.endswith("_minus_nested_reference_joint")
        },
        "seconds":report_output["seconds"],
    }),flush=True)

if __name__=="__main__":
    main()
