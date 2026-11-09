"""Nonlinear low-dimensional CPI baseline on public BALM cold-protein DEV.

Fixed HistGradientBoosting parameters, train-fold feature mapping and group CV.
Only train & inner-dev labels are accessible; reserved test untouched.
"""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from train_bindingdb_descriptor_protein_boosting_dev import molecule_features,protein_features
from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
   collapse_training,metric)
OUT=ROOT/"research/balm_bindingdb_frozen_nonlinear_descriptor_protein_boosting_dev.json"
ARGS=dict(max_iter=180,learning_rate=.05,max_leaf_nodes=15,max_depth=5,
          min_samples_leaf=40,l2_regularization=15.,early_stopping=False,
          random_state=2103)
CANDIDATES=[(name,chem,weight) for name,chem,weight in (
    ("chemical_only",True,False),
    ("chemical_plus_protein",False,False),
    ("chemical_weighted_by_component",True,True),
    ("chemical_plus_protein_weighted",False,True))]
def data_matrices(rows,chemcache,protc):
    c=np.zeros((len(rows),21),dtype=np.float32)
    p=np.zeros((len(rows),21),dtype=np.float32)
    for i,r in enumerate(rows):
        sm=r["smiles"];seq=r["protein"]
        if sm not in chemcache:chemcache[sm]=molecule_features(sm)
        if seq not in protc:protc[seq]=protein_features(seq)
        c[i]=chemcache[sm];p[i]=protc[seq]
    return c,np.column_stack([c,p])
def calc(name,trY,trG,xtr,xval,with_weight):
    model=HistGradientBoostingRegressor(**ARGS)
    if with_weight:
        count=Counter(trG)
        w=np.array([1./np.sqrt(count[x]) for x in trG])
        w=w/w.mean()
        model.fit(xtr,trY,sample_weight=w)
    else:model.fit(xtr,trY)
    return model.predict(xval)
def main():
    start=time.time()
    data=get_train_dev()
    tr_raw=data["train"];dev=data["inner_dev"]
    ccache={};pcache={}
    ctr_raw,ptr_raw=data_matrices(tr_raw,ccache,pcache)
    cdev,pdev=data_matrices(dev,ccache,pcache)
    yy=np.asarray([r["label"] for r in dev],dtype=np.float64)
    gv=np.asarray([r["group"] for r in dev])
    tr_groups=np.asarray([r["group"] for r in tr_raw])
    fold_scores={name:[] for name,_,_ in CANDIDATES}
    cv_baseline=[]
    for fold,(itr,iva) in enumerate(GroupShuffleSplit(n_splits=3,
               test_size=.2,random_state=20261108).split(
               tr_groups,groups=tr_groups),1):
        tr=collapse_training([tr_raw[int(i)] for i in itr])
        va=[tr_raw[int(i)] for i in iva]
        c,p=data_matrices(tr,ccache,pcache)
        cv,cp=data_matrices(va,ccache,pcache)
        y=np.asarray([r["label"] for r in tr])
        v=np.asarray([r["label"] for r in va])
        g=np.asarray([r["group"] for r in va])
        groupt=np.asarray([r["group"] for r in tr])
        b=metric(v,np.full_like(v,y.mean()),g)
        cv_baseline.append(b)
        for name,is_chem,weighted in CANDIDATES:
            pred=calc(name,y,groupt,c if is_chem else p,
                      cv if is_chem else cp,weighted)
            fold_scores[name].append(metric(v,pred,g))
        print(json.dumps({"fold":fold,"n_train":len(tr),"n_dev":len(va),
                          "baseline":b,"models":{n:fold_scores[n][-1] for n,_,_ in CANDIDATES},
                          "elapsed_s":round(time.time()-start,1)}),flush=True)
    mean_p=float(np.mean([x["pooled_rmse"] for x in cv_baseline]))
    mean_m=float(np.mean([x["component_macro_rmse"] for x in cv_baseline]))
    ranks=[]
    for name,_,_ in CANDIDATES:
        p=float(np.mean([x["pooled_rmse"] for x in fold_scores[name]]))
        m=float(np.mean([x["component_macro_rmse"] for x in fold_scores[name]]))
        ranks.append({"name":name,"cv_pooled":p,"cv_macro":m,
                      "relative_joint":.5*(p/mean_p+m/mean_m)})
    ranks.sort(key=lambda z:z["relative_joint"])
    eligible=[r for r in ranks if r["cv_pooled"]<=mean_p and r["cv_macro"]<=mean_m]
    selected=eligible[0]["name"] if eligible else "train_mean"
    tr=collapse_training(tr_raw)
    c,p=data_matrices(tr,ccache,pcache)
    y=np.asarray([r["label"] for r in tr])
    grp=np.asarray([r["group"] for r in tr])
    dev_all={}
    for name,chem,weighted in CANDIDATES:
        pred=calc(name,y,grp,c if chem else p,cdev if chem else pdev,weighted)
        dev_all[name]=metric(yy,pred,gv)
    dev_mean=metric(yy,np.full_like(yy,y.mean()),gv)
    out={
      "status":"EXPLORATORY_PUBLIC_BALM_COLD_TARGET_NONLINEAR_DESCRIPTOR_DEV_ONLY",
      "training_pairs_unique":len(tr),
      "inner_dev_original_pairs":len(dev),
      "component_disjoint":True,
      "input_features":"21 RDKit physicochemical descriptors; 21 protein AA-composition features",
      "model":"HistGradientBoostingRegressor",
      "fixed_hyperparameters":ARGS,
      "cv_baseline":cv_baseline,
      "cv_models":fold_scores,
      "cv_ranking":ranks,
      "cv_selected":selected,
      "dev_train_only_mean":dev_mean,
      "dev_model_ablation":dev_all,
      "dev_cv_selected":dev_all.get(selected,dev_mean),
      "seconds":round(time.time()-start,1),
      "warnings":["Inner-dev repeatedly inspected; exploratory only.",
                  "Protein 3-mer graph grouping is not validated sequence homology clustering.",
                  "Published BALM source lacks original assay DOI and measurement protocol fields.",
                  "Prospective test not evaluated or used for model selection."]
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ("cv_baseline","cv_models","cv_ranking")},indent=2),flush=True)
    print(json.dumps({"top_cv":ranks},indent=2),flush=True)

if __name__=="__main__":main()
