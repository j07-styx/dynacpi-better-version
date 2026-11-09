"""Low-rank explicit chemical × protein interaction features for BALM cold-target dev.

All TF-IDF vocabularies, SVD projections, ridge coefficients and model selection
are fitted inside training-only protein-component folds. Reserved test untouched.
"""
import json
import time
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize
from rdkit.Chem import rdFingerprintGenerator

from dynacpi.balm_public_data_dev import get_train_dev, ROOT
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training, fingerprint_rows, metric
)

OUT=ROOT/"research/balm_explicit_cross_interaction_svd_ridge_traincv_dev.json"
ALPHAS=(1.0,10.0)
CROSS_SCALES=(0.0,0.25,1.0)
PROTEIN_SCALE=0.25
RANK=16
SEED=20261108

def projected_pair(c_tr,c_va,p_tr,p_va):
    """Fit low-rank projections only on the training fold; build pairwise outer products."""
    cs=TruncatedSVD(n_components=RANK,random_state=SEED,n_iter=5)
    ps=TruncatedSVD(n_components=RANK,random_state=SEED,n_iter=5)
    ca=normalize(cs.fit_transform(c_tr).astype(np.float32))
    cb=normalize(cs.transform(c_va).astype(np.float32))
    pa=normalize(ps.fit_transform(p_tr).astype(np.float32))
    pb=normalize(ps.transform(p_va).astype(np.float32))
    def pair(x,y):
        return csr_matrix(np.einsum("bi,bj->bij",x,y,optimize=True).reshape(len(x),-1))
    return pair(ca,pa),pair(cb,pb)

def features(train,valid,fpgen,cache):
    chem_tr=fingerprint_rows(train,fpgen,cache)
    chem_va=fingerprint_rows(valid,fpgen,cache)
    vec=TfidfVectorizer(analyzer="char",ngram_range=(3,4),
                        max_features=4096,sublinear_tf=True,dtype=np.float32)
    protein_tr=vec.fit_transform([r["protein"] for r in train])
    protein_va=vec.transform([r["protein"] for r in valid])
    assert chem_tr.shape[1]==2048 and protein_tr.shape[1]<=4096
    pair_tr,pair_va=projected_pair(chem_tr,chem_va,protein_tr,protein_va)
    additive_train=hstack([chem_tr,protein_tr*PROTEIN_SCALE],format="csr")
    additive_valid=hstack([chem_va,protein_va*PROTEIN_SCALE],format="csr")
    return additive_train,additive_valid,pair_tr,pair_va

def fit_predict(xtr,xva,pairtr,pairva,y,alpha,cross_scale):
    if cross_scale==0.0:
        a,b=xtr,xva
    else:
        a=hstack([xtr,pairtr*cross_scale],format="csr")
        b=hstack([xva,pairva*cross_scale],format="csr")
    mu=float(np.mean(y));std=float(np.std(y));assert std>0
    model=Ridge(alpha=alpha,solver="lsqr")
    model.fit(a,(y-mu)/std)
    return model.predict(b)*std+mu

def rows_metrics(train,valid,fpgen,cache):
    xtr,xva,pa,pb=features(train,valid,fpgen,cache)
    ytr=np.array([x["label"] for x in train],dtype=np.float64)
    yva=np.array([x["label"] for x in valid],dtype=np.float64)
    groups=np.array([x["group"] for x in valid])
    mean=metric(yva,np.full(len(yva),ytr.mean()),groups)
    results={}
    for alpha in ALPHAS:
        for cross in CROSS_SCALES:
            pred=fit_predict(xtr,xva,pa,pb,ytr,alpha,cross)
            results[f"alpha={alpha}|cross={cross}"]=metric(yva,pred,groups)
    return mean,results

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    start=time.time()
    parts=get_train_dev()
    train=collapse_training(parts["train"])
    dev=parts["inner_dev"]
    groups=np.array([r["group"] for r in train])
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={}
    folds=[]
    splitter=GroupShuffleSplit(n_splits=3,test_size=.2,random_state=SEED)
    for idx,(tr,va) in enumerate(splitter.split(groups,groups=groups),start=1):
        train_fold=[train[int(i)] for i in tr]
        valid_fold=[train[int(i)] for i in va]
        assert set(groups[tr]).isdisjoint(set(groups[va]))
        base,results=rows_metrics(train_fold,valid_fold,fpgen,cache)
        folds.append({"fold":idx,"train_pairs":len(train_fold),"valid_pairs":len(valid_fold),
                      "train_groups":len(set(groups[tr])),"valid_groups":len(set(groups[va])),
                      "baseline":base,"models":results})
        print(json.dumps({"fold":idx,"train_pairs":len(train_fold),
                          "valid_pairs":len(valid_fold),
                          "cross_alpha1":results["alpha=1.0|cross=1.0"],
                          "additive_alpha1":results["alpha=1.0|cross=0.0"]}),flush=True)
    candidates=[]
    for key in folds[0]["models"]:
        pooled=np.mean([x["models"][key]["pooled_rmse"] for x in folds])
        macro=np.mean([x["models"][key]["component_macro_rmse"] for x in folds])
        base_p=np.mean([x["baseline"]["pooled_rmse"] for x in folds])
        base_m=np.mean([x["baseline"]["component_macro_rmse"] for x in folds])
        candidates.append({"name":key,"pooled":float(pooled),"macro":float(macro),
                           "balanced_normalized_cv":float(.5*(pooled/base_p+macro/base_m))})
    ranked=sorted(candidates,key=lambda x:(x["balanced_normalized_cv"],x["pooled"]))
    selected=ranked[0]["name"]
    assert selected in folds[0]["models"]
    print("TRAIN CV SELECTED",selected,"score",ranked[0],flush=True)
    # Development labels are used only now, after the selection above has been fixed.
    final_base,final_results=rows_metrics(train,dev,fpgen,cache)
    outcome={
       "status":"BALM_EXPLORATORY_GROUPED_TRAIN_CV_SELECTED_INTERACTION_DEV_NO_TEST",
       "train_unique_pairs":len(train),"dev_rows":len(dev),
       "train_protein_components":len(set(groups)),
       "dev_protein_components":len({r["group"] for r in dev}),
       "features":"Morgan chirality2048 + TF-IDF protein 3-4mers max4096 scale0.25, optional normalized rank16 by rank16 outer-product interaction learned via TruncatedSVD on training only",
       "rank":RANK,"alpha_grid":ALPHAS,"cross_scale_grid":CROSS_SCALES,
       "train_only_groupcv":{"folds":folds,"ranking":ranked,"selected":selected},
       "dev_baseline":final_base,
       "dev_selected":final_results[selected],
       "dev_fixed_comparator_additive_alpha1":final_results["alpha=1.0|cross=0.0"],
       "dev_fixed_interaction_alpha1":final_results["alpha=1.0|cross=1.0"],
       "duration_seconds":round(time.time()-start,2),
       "caveats":["No original assay source method or construct evidence approval",
                  "Three overlapping folds, exploratory dev has been repeatedly inspected",
                  "Trigam similarity graph is approximate protein homology",
                  "Reserved prospective test not accessed or scored"]
    }
    OUT.write_text(json.dumps(outcome,indent=2)+"\n",encoding="utf-8")
    print("FINAL",json.dumps({"winner":selected,"cv":ranked[0],
                              "dev_baseline":final_base,
                              "dev_selected":final_results[selected],
                              "dev_additive_alpha1":final_results["alpha=1.0|cross=0.0"],
                              "dev_interaction_alpha1":final_results["alpha=1.0|cross=1.0"]}),flush=True)

if __name__=="__main__":main()
