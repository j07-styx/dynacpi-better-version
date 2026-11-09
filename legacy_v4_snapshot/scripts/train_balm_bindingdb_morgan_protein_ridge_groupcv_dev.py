"""Cold-target BALM BindingDB development: grouped CV for fixed ridge controls.

All model fitting, feature-selection vocabulary and training target aggregation
use only the corresponding training fold. Final reserved test is untouched.
"""
import json,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from rdkit import Chem,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

from dynacpi.balm_public_data_dev import get_train_dev,ROOT
RDLogger.DisableLog("rdApp.*")
OUT=ROOT/"research/balm_bindingdb_frozen_morgan_protein_ridge_groupcv_dev.json"

def metric(y,pred,groups):
    y=np.asarray(y,dtype=np.float64);pred=np.asarray(pred,dtype=np.float64)
    g=np.asarray(groups)
    err=(y-pred)**2
    return {"pooled_rmse":float(np.sqrt(err.mean())),
            "component_macro_rmse":float(np.mean([
                np.sqrt(err[g==key].mean()) for key in sorted(set(g))]))}

def collapse_training(records):
    unique=defaultdict(list)
    for r in records:
        unique[(r["protein"],r["smiles"])].append(r)
    out=[]
    for k,rows in unique.items():
        item=dict(rows[0])
        item["label"]=float(np.mean([r["label"] for r in rows]))
        out.append(item)
    return out

def fingerprint_rows(rows,gen,cache):
    rr=[];cc=[]
    for i,r in enumerate(rows):
        sm=r["smiles"]
        if sm not in cache:
            mol=Chem.MolFromSmiles(sm)
            assert mol is not None
            cache[sm]=tuple(gen.GetFingerprint(mol).GetOnBits())
        bits=cache[sm]
        rr.extend([i]*len(bits));cc.extend(bits)
    return normalize(csr_matrix((np.ones(len(cc),dtype=np.float32),
           (rr,cc)),shape=(len(rows),2048),dtype=np.float32))

def make_inputs(tr,va,gen,cache):
    c_tr=fingerprint_rows(tr,gen,cache)
    c_va=fingerprint_rows(va,gen,cache)
    vec=TfidfVectorizer(analyzer="char",ngram_range=(3,4),max_features=4096,
                        sublinear_tf=True,dtype=np.float32)
    p_tr=vec.fit_transform([r["protein"] for r in tr])
    p_va=vec.transform([r["protein"] for r in va])
    return {
       "chemical":(c_tr,c_va),
       "protein":(p_tr,p_va),
       "combined":(hstack([c_tr,p_tr],format="csr"),
                   hstack([c_va,p_va],format="csr"))
    }

def baseline(tr,va):
    y=np.asarray([r["label"] for r in va])
    g=[r["group"] for r in va]
    mu=float(np.mean([r["label"] for r in tr]))
    c=Counter(r["group"] for r in tr)
    weighted=float(np.average([r["label"] for r in tr],
                      weights=[1./np.sqrt(c[r["group"]]) for r in tr]))
    return {
      "global_mean":metric(y,np.full_like(y,mu),g),
      "component_balanced_mean":metric(y,np.full_like(y,weighted),g)
    }

def models(tr,va,gen,cache):
    xv=make_inputs(tr,va,gen,cache)
    y=np.asarray([r["label"] for r in tr])
    v=np.asarray([r["label"] for r in va])
    g=[r["group"] for r in va]
    mu=float(y.mean());std=float(y.std());assert std>0
    output={}
    for name,(xtr,xva) in xv.items():
        for alpha in (0.1,1.0,10.0):
            model=Ridge(alpha=alpha,solver="lsqr")
            model.fit(xtr,(y-mu)/std)
            pred=model.predict(xva)*std+mu
            output[name+"_alpha_"+str(alpha)]=metric(v,pred,g)
    return output

def main():
    start=time.time()
    parts=get_train_dev()
    train=parts["train"];dev=parts["inner_dev"]
    groups=np.asarray([r["group"] for r in train])
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={}
    foldresults=[]
    for i,(itr,iva) in enumerate(
      GroupShuffleSplit(n_splits=3,test_size=.2,random_state=20261108).split(
      groups,groups=groups),1):
        tr=collapse_training([train[int(j)] for j in itr])
        va=[train[int(j)] for j in iva]
        b=baseline(tr,va)
        score=models(tr,va,gen,cache)
        foldresults.append({"fold":i,"train_unique_rows":len(tr),"heldout_rows":len(va),
                            "baseline":b,"models":score})
        print(json.dumps({"fold":i,"train_unique_rows":len(tr),
                          "heldout_rows":len(va),"baseline":b,
                          "top_models":sorted(score.items(),key=lambda x:x[1]["pooled_rmse"])[:3],
                          "seconds":round(time.time()-start,1)}),flush=True)
    mean_p=float(np.mean([r["baseline"]["global_mean"]["pooled_rmse"] for r in foldresults]))
    mean_m=float(np.mean([r["baseline"]["global_mean"]["component_macro_rmse"] for r in foldresults]))
    ranked=[]
    for name in foldresults[0]["models"]:
        pooled=float(np.mean([r["models"][name]["pooled_rmse"] for r in foldresults]))
        macro=float(np.mean([r["models"][name]["component_macro_rmse"] for r in foldresults]))
        ranked.append({"model":name,"cv_pooled_mean":pooled,
                       "cv_macro_mean":macro,
                       "cv_joint":.5*(pooled/mean_p+macro/mean_m)})
    ranked.sort(key=lambda a:(a["cv_joint"],a["cv_pooled_mean"]))
    eligible=[r for r in ranked if r["cv_pooled_mean"]<mean_p and r["cv_macro_mean"]<mean_m]
    winner=eligible[0]["model"] if eligible else "global_mean"
    compressed=collapse_training(train)
    finalbase=baseline(compressed,dev)
    if winner=="global_mean":
        selected=finalbase["global_mean"]
    else:
        selected=models(compressed,dev,gen,cache)[winner]
    out={
     "status":"BALM_BINDINGDB_COLD_PROTEIN_GRAPH_GROUPCV_AND_INNER_DEV_ONLY",
     "train_original_rows":len(train),"train_unique_protein_compound":len(compressed),
     "inner_dev_rows":len(dev),
     "training_components":len(set(groups)),
     "inner_dev_components":len({r["group"] for r in dev}),
     "feature_design":"Train-fold fitted protein TFIDF char3-4mer up to 4096, Morgan radius2 chiral 2048 L2, Ridge alpha grid",
     "target_handling":"Collapse duplicate protein+SMILES training rows by mean within each train fold; do not aggregate development labels",
     "train_only_cv_folds":foldresults,
     "train_only_cv_ranking":ranked,
     "train_only_cv_selected":winner,
     "final_dev_baselines":finalbase,
     "final_dev_selected":selected,
     "seconds":round(time.time()-start,1),
     "caveats":[
       "Public BALM source may contain assay-context ambiguity and no source DOI fields.",
       "Trigram graph threshold0.5 is a similarity proxy, not proven homology clustering.",
       "Training-only grouped CV selects model and regularization; inner-dev is exploratory only.",
       "Reserved test partition not evaluated or accessed as Y values."
      ]
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items()
      if k not in ("train_only_cv_folds","train_only_cv_ranking")},indent=2),flush=True)
    print(json.dumps({"top_cv":ranked[:6]}),flush=True)

if __name__=="__main__":main()
