"""Train-only group-CV selection of protein-feature weight in additive CPI Ridge.

Defer development evaluation until the scale and regularization are locked.
Compare against a fixed chemical-only alpha10 reference. Test remains untouched.
"""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,fingerprint_rows,metric)
OUT=ROOT/"research/balm_bindingdb_frozen_protein_weighted_ridge_groupcv_dev.json"
SCALES=(0.0,0.25,0.5,1.0,2.0)
ALPHAS=(1.,10.,100.)

def representation(tr,va,gen,cache):
    chemtr=fingerprint_rows(tr,gen,cache)
    chemva=fingerprint_rows(va,gen,cache)
    tok=TfidfVectorizer(analyzer="char",ngram_range=(3,4),
                        max_features=4096,dtype=np.float32,sublinear_tf=True)
    p_tr=tok.fit_transform([r["protein"] for r in tr])
    p_va=tok.transform([r["protein"] for r in va])
    return chemtr,chemva,p_tr,p_va

def predict(xtr,ytr,xval,alpha):
    mu=float(ytr.mean());sd=float(ytr.std());assert sd>0
    model=Ridge(alpha=alpha,solver="lsqr")
    model.fit(xtr,(ytr-mu)/sd)
    return model.predict(xval)*sd+mu

def main():
    start=time.time()
    parts=get_train_dev()
    train=parts["train"]
    dev=parts["inner_dev"]
    group=np.asarray([r["group"] for r in train])
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={}
    cv=[];rank={}
    for f,(it,iv) in enumerate(GroupShuffleSplit(n_splits=3,test_size=.2,
                  random_state=20261108).split(group,groups=group),1):
        tr=collapse_training([train[int(i)] for i in it])
        va=[train[int(i)] for i in iv]
        c,d,p,q=representation(tr,va,gen,cache)
        y=np.asarray([r["label"] for r in tr])
        v=np.asarray([r["label"] for r in va])
        g=[r["group"] for r in va]
        base=metric(v,np.full_like(v,y.mean()),g)
        chempred=predict(c,y,d,10.)
        chem=metric(v,chempred,g)
        scores={}
        for scale in SCALES:
            if scale:
                train_x=hstack([c,p*scale],format="csr")
                val_x=hstack([d,q*scale],format="csr")
            else:train_x,val_x=c,d
            for alpha in ALPHAS:
                model=f"protein_scale_{scale}_alpha_{alpha}"
                value=metric(v,predict(train_x,y,val_x,alpha),g)
                scores[model]=value
                rank.setdefault(model,[]).append(value)
        cv.append({"fold":f,"train_unique":len(tr),"heldout":len(va),
                   "train_mean":base,"chemical_only_alpha10":chem})
        print(json.dumps({"fold":f,"baseline":base,"chemical":chem,
                          "best_pooled":min(scores.items(),key=lambda s:s[1]["pooled_rmse"])[:2],
                          "seconds":round(time.time()-start,1)}),flush=True)
    chem_p=np.mean([r["chemical_only_alpha10"]["pooled_rmse"] for r in cv])
    chem_m=np.mean([r["chemical_only_alpha10"]["component_macro_rmse"] for r in cv])
    ranked=[]
    for key,vals in rank.items():
        p=float(np.mean([x["pooled_rmse"] for x in vals]))
        m=float(np.mean([x["component_macro_rmse"] for x in vals]))
        ranked.append({"name":key,"pooled_cv":p,"macro_cv":m,
                       "joint_relative_to_chemical":float(.5*(p/chem_p+m/chem_m))})
    ranked.sort(key=lambda x:x["joint_relative_to_chemical"])
    candidates=[r for r in ranked if r["pooled_cv"]<=chem_p and r["macro_cv"]<=chem_m]
    selected=candidates[0] if candidates else None
    alltr=collapse_training(train)
    c,d,p,q=representation(alltr,dev,gen,cache)
    y=np.asarray([r["label"] for r in alltr])
    v=np.asarray([r["label"] for r in dev])
    g=np.asarray([r["group"] for r in dev])
    dev_chem=metric(v,predict(c,y,d,10.),g)
    if selected:
        name=selected["name"]
        scale=float(name.split("_alpha_")[0].split("scale_")[1])
        alpha=float(name.split("_alpha_")[1])
        if scale:
            trX=hstack([c,p*scale],format="csr")
            dvX=hstack([d,q*scale],format="csr")
        else:trX,dvX=c,d
        dev_selected=metric(v,predict(trX,y,dvX,alpha),g)
    else: dev_selected=dev_chem
    out={"status":"TRAIN_ONLY_GROUPCV_WEIGHTED_PROTEIN_RIDGE_NO_TEST",
         "train_unique":len(alltr),"dev":len(dev),
         "train_components":len(set(group)),
         "dev_components":len(set(g)),
         "protein_feature_scales":SCALES,
         "regularization_alphas":ALPHAS,
         "cv_folds":cv,
         "cv_chemical_reference_pooled":float(chem_p),
         "cv_chemical_reference_macro":float(chem_m),
         "cv_ranking":ranked,
         "cv_selected_or_chemical":selected or "chemical_only_alpha10",
         "dev_chemical_only_alpha10":dev_chem,
         "dev_cv_selected":dev_selected,
         "seconds":round(time.time()-start,1),
         "limitations":["All settings selected on grouped train-only folds; development already repeatedly inspected.",
           "Exact protein sequences blocked by approximate trigram-graph grouping, not verified homology identity.",
           "Prospective test untouched; source assay provenance unavailable."]}
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ("cv_folds","cv_ranking")},indent=2),flush=True)
    print(json.dumps({"top_five_cv":ranked[:5]},indent=2),flush=True)

if __name__=="__main__":main()
