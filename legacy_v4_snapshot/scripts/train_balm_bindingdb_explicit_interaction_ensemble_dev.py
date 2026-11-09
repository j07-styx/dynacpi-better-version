"""Explicit chemical-protein biophysical interaction and ridge/tree ensemble.

TRAIN-only held-out protein group CV locks a blend before exploratory inner-dev.
Does NOT score the frozen reserved test.
"""
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.sparse import hstack
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,fingerprint_rows,metric,
)
from scripts.train_bindingdb_descriptor_protein_boosting_dev import (
    molecule_features,protein_features,AMINO,
)

OUT=ROOT/"research/balm_bindingdb_explicit_chemical_protein_interaction_ensemble_dev.json"
HIST_PARAMS=dict(max_iter=160,learning_rate=.05,max_leaf_nodes=15,
                 max_depth=5,min_samples_leaf=50,l2_regularization=15.,
                 early_stopping=False,random_state=2103)
BLENDS=(0.,.25,.5,.75,1.)
TARGET='pooled_and_macro_cv_mean'

def bio_interaction(chemical,protein):
    # 8 predeclared chemical descriptors and eight residue-chemistry summaries.
    cs=chemical[:,[0,1,2,3,4,5,7,10]]
    def grp(letters):
        return sum(protein[:,AMINO.index(x)] for x in letters)
    ps=np.column_stack([
       grp('DE'),grp('KRH'),grp('AILMFWV'),grp('FWY'),
       grp('STNQ'),grp('C'),grp('P'),protein[:,-1],
    ])
    return np.column_stack([
       chemical,protein,(cs[:,:,None]*ps[:,None,:]).reshape(len(chemical),-1),
    ]).astype(np.float32)

def data_features(rows,chemcache,protcache):
    c=np.empty((len(rows),21),dtype=np.float32)
    p=np.empty((len(rows),21),dtype=np.float32)
    for i,r in enumerate(rows):
        sm=r["smiles"];seq=r["protein"]
        if sm not in chemcache:chemcache[sm]=molecule_features(sm)
        if seq not in protcache:protcache[seq]=protein_features(seq)
        c[i]=chemcache[sm];p[i]=protcache[seq]
    return c,p

def fit_predict(tr,va,gen,fp_cache,chemcache,protcache):
    y=np.asarray([r["label"] for r in tr],dtype=np.float64)
    mu=y.mean();sd=y.std();assert sd>0
    trc=fingerprint_rows(tr,gen,fp_cache)
    vac=fingerprint_rows(va,gen,fp_cache)
    vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(3,4),
                              max_features=4096,sublinear_tf=True,dtype=np.float32)
    trp=vectorizer.fit_transform([r["protein"] for r in tr])
    vap=vectorizer.transform([r["protein"] for r in va])
    reg=Ridge(alpha=1.,solver="lsqr")
    reg.fit(hstack([trc,trp*.25],format="csr"),(y-mu)/sd)
    ridge=reg.predict(hstack([vac,vap*.25],format="csr"))*sd+mu

    c,p=data_features(tr,chemcache,protcache)
    cv,pv=data_features(va,chemcache,protcache)
    joint=HistGradientBoostingRegressor(**HIST_PARAMS)
    joint.fit(np.column_stack([c,p]),y)
    boosted_joint=joint.predict(np.column_stack([cv,pv]))
    crossed=HistGradientBoostingRegressor(**HIST_PARAMS)
    crossed.fit(bio_interaction(c,p),y)
    boosted_interaction=crossed.predict(bio_interaction(cv,pv))
    return {
       "ridge":ridge,
       "boost_joint":boosted_joint,
       "boost_explicit_interaction":boosted_interaction,
       **{f"ridge_interaction_blend_{w}":(1.-w)*ridge+w*boosted_interaction for w in BLENDS},
    }

def score(y,pred,groups):
    return metric(y,pred,groups)

def main():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.time()
    parts=get_train_dev()
    train=parts["train"]
    inner_dev=parts["inner_dev"]
    grp=np.asarray([r["group"] for r in train])
    assert len(train)==15538 and len(inner_dev)==3891
    assert set(grp).isdisjoint({r["group"] for r in inner_dev})
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fp_cache={};chemcache={};protcache={}
    cv=[]
    for fold,(itra,iva) in enumerate(GroupShuffleSplit(
       n_splits=3,test_size=.2,random_state=20261108
    ).split(grp,groups=grp),1):
        fit=collapse_training([train[int(i)] for i in itra])
        val=[train[int(i)] for i in iva]
        assert set(r["group"] for r in fit).isdisjoint(set(r["group"] for r in val))
        yy=np.asarray([r["label"] for r in val],dtype=np.float64)
        gg=np.asarray([r["group"] for r in val])
        preds=fit_predict(fit,val,fpgen,fp_cache,chemcache,protcache)
        base_mean=float(np.mean([r["label"] for r in fit]))
        results={"train_mean":score(yy,np.full_like(yy,base_mean),gg)}
        results.update({name:score(yy,v,gg) for name,v in preds.items()})
        cv.append({"fold":fold,"train_unique_pairs":len(fit),"holdout_pairs":len(val),
                   "holdout_components":len(set(gg)),"results":results})
        print("FOLD",fold,json.dumps({n:results[n] for n in (
           "train_mean","ridge","boost_joint","boost_explicit_interaction",
           "ridge_interaction_blend_0.25","ridge_interaction_blend_0.5")}),flush=True)
    averages={}
    for name in cv[0]["results"]:
        averages[name]={m:float(np.mean([f["results"][name][m] for f in cv]))
                        for m in ("pooled_rmse","component_macro_rmse")}
    baseline=averages["ridge"]
    ranked=sorted([
        {"name":name,"mean_pooled_cv":z["pooled_rmse"],
         "mean_macro_cv":z["component_macro_rmse"],
         "relative_joint":.5*(z["pooled_rmse"]/baseline["pooled_rmse"]+
                              z["component_macro_rmse"]/baseline["component_macro_rmse"])}
        for name,z in averages.items() if name != "train_mean"
    ],key=lambda x:(x["relative_joint"],x["mean_pooled_cv"]))
    eligible=[x for x in ranked if x["mean_pooled_cv"]<=baseline["pooled_rmse"] and
             x["mean_macro_cv"]<=baseline["component_macro_rmse"]]
    selected=eligible[0]["name"] if eligible else "ridge"
    print("TRAIN_ONLY_SELECTED",selected,json.dumps(ranked[:5]),flush=True)

    tr=collapse_training(train)
    pred=fit_predict(tr,inner_dev,fpgen,fp_cache,chemcache,protcache)
    yy=np.asarray([r["label"] for r in inner_dev],dtype=np.float64)
    gg=np.asarray([r["group"] for r in inner_dev])
    inner_mean=np.mean([r["label"] for r in tr])
    # For transparent reporting, dev metrics for every predeclared candidate.
    dev_metrics={"train_mean":score(yy,np.full_like(yy,inner_mean),gg)}
    dev_metrics.update({name:score(yy,v,gg) for name,v in pred.items()})
    out={
       "status":"BALM_FROZEN_INNER_DEV_EXPLORATORY_TRAIN_ONLY_MODEL_SELECTION",
       "training_raw_rows":len(train),"training_unique_pairs":len(tr),
       "inner_dev_rows":len(inner_dev),
       "train_groups":len(set(grp)),"dev_groups":len(set(gg)),
       "model":"Morgan2048 plus train-fitted protein tri/tetra-gram Ridge and explicit biochemical cross-feature gradient boosting",
       "explicit_interactions":"8 chemical RDKit descriptors x 8 protein residue composition summaries = 64 cross-product terms",
       "interaction_features":106,
       "ridge_alpha":1.,"ridge_protein_scale":.25,
       "boost_hyperparameters":HIST_PARAMS,
       "blend_weights_predeclared":BLENDS,
       "cv_folds":cv,"cv_mean":averages,"cv_ranked":ranked,
       "selected_by_train_cv":selected,
       "inner_dev_selected":dev_metrics[selected],
       "inner_dev_candidate_metrics":dev_metrics,
       "seconds":round(time.time()-start,2),
       "caveats":[
          "BALM source lacks assay DOI/method per record, label conflicts remain.",
          "Approximate protein trigram graph, not verified homology.",
          "Only three overlapping protein-heldout CV folds.",
          "Development set already repeatedly inspected; results exploratory, not test.",
          "No reserved prospective test loaded, fitted, predicted or evaluated.",
       ],
    }
    OUT.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("FINAL",json.dumps({"selected":selected,"train_cv_best":ranked[:4],
       "dev_selected":dev_metrics[selected],"dev_ridge":dev_metrics["ridge"],
       "dev_mean":dev_metrics["train_mean"]}),flush=True)

if __name__=="__main__":main()
