"""Assay-aware CPI metadata ablation on frozen cold-protein development split.

Unlike sequence+compound-only methods, this model sees assay protocol text.
Assay descriptions are available as input covariates without using outcome labels.
All feature fits and model selection are on training protein groups only.
"""
import hashlib,json,re,time
from pathlib import Path
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

from train_bindingdb_descriptor_protein_boosting_dev import (
    molecule_features,protein_features,records,PARAMS)
from train_bindingdb_cold_protein_prior_dev import metric

ROOT=Path(__file__).resolve().parents[1];A=ROOT/"data"/"audit"
OUT=ROOT/"research"/"bindingdb_assay_text_aware_boosting_inner_dev.json"
TERMS=[
    "intrinsic","apparent","thermal","fluorescen","dsf",
    "surface plasmon","spr","itc","isothermal","enzyme",
    "radioligand","kinetic","inhibition","binding affinity",
    "cell","recombinant","dissociation","temperature","buffer","ph",
    "calorimetric","hsa","plasma","serum"]
def load():
    m=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    tr=set(m["train_components"]);dv=set(m["inner_dev_components"]);te=set(m["test_components"])
    assert tr.isdisjoint(dv|te) and dv.isdisjoint(te)
    comp={}
    for r in records(A/"bindingdb_full_kd_protein_components.jsonl"):
        for p in r["members"]:comp[p]=r["component_id"]
    allowed={(r["sequence_sha256"],r["assay_key"]) for r in records(
        A/"bindingdb_full_kd_task_summary.jsonl") if r["unique_canonical_compounds"]>=15}
    train=[];dev=[]
    for r in records(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if (r["protein_seq_sha256"],r["assay_key"]) not in allowed or r["range_pKd"]>.1:continue
        c=comp[r["protein_seq_sha256"]]
        if c not in tr and c not in dv:continue
        item={"component":c,"protein":r["protein_seq_sha256"],"compound":r["smiles_isomeric"],
              "assay":r["assay_key"],"label":float(r["median_pKd"])}
        (train if c in tr else dev).append(item)
    assert len(train)==m["train_pairs"] and len(dev)==m["inner_dev_pairs"]
    needed_prot={r["protein"] for r in train+dev}
    seqs={}
    for r in records(A/"bindingdb_full_kd_candidates.jsonl"):
        s=r["sequence"]
        key=hashlib.sha256(s.encode()).hexdigest()
        if key in needed_prot:seqs[key]=s
        if len(seqs)==len(needed_prot):break
    assert len(seqs)==len(needed_prot)
    needed_assays={(r["protein"],r["assay"]) for r in train+dev}
    ctx={}
    for r in records(A/"bindingdb_full_kd_assay_contexts.jsonl"):
        k=(r["sequence_sha256"],r["assay_key"])
        if k in needed_assays:
            ctx[k]=r["description"] or ""
    assert len(ctx)==len(needed_assays),(len(ctx),len(needed_assays))
    return train,dev,seqs,ctx

def features(items,seqs,ctx,chemcache,protcache):
    x=np.empty((len(items),42),dtype=np.float32)
    text=[]
    for i,r in enumerate(items):
        sm=r["compound"]
        if sm not in chemcache:chemcache[sm]=molecule_features(sm)
        p=r["protein"]
        if p not in protcache:protcache[p]=protein_features(seqs[p])
        x[i,:21]=chemcache[sm]
        x[i,21:]=protcache[p]
        text.append(ctx[(p,r["assay"])])
    return x,text

def flags(text):
    return np.asarray([[float(term in s.lower()) for term in TERMS] +
                        [float(len(s)==0),float(len(s)>120)]
                        for s in text],dtype=np.float32)

def metadata(xtr,xv,traintext,valtext,kind):
    if kind=="none":return xtr,xv
    a,b=flags(traintext),flags(valtext)
    if kind=="flags":return np.column_stack([xtr,a]),np.column_stack([xv,b])
    assert kind=="text_svd"
    tfidf=TfidfVectorizer(analyzer="char",ngram_range=(3,5),
                          min_df=3,max_features=3000,dtype=np.float32,
                          sublinear_tf=True)
    t=tfidf.fit_transform(traintext)
    v=tfidf.transform(valtext)
    n=min(12,max(1,t.shape[1]-1))
    svd=TruncatedSVD(n_components=n,random_state=2103)
    return np.column_stack([xtr,a,svd.fit_transform(t)]),np.column_stack([xv,b,svd.transform(v)])

def score(model,xtr,ytr,xv,yv,groups):
    model.fit(xtr,ytr)
    return metric(yv,model.predict(xv),groups)

def main():
    start=time.time()
    train,dev,seqs,ctx=load()
    chemcache={};protcache={}
    xtr,trtext=features(train,seqs,ctx,chemcache,protcache)
    xdev,devtext=features(dev,seqs,ctx,chemcache,protcache)
    y=np.asarray([r["label"] for r in train],dtype=np.float32)
    vy=np.asarray([r["label"] for r in dev],dtype=np.float32)
    groups=np.asarray([r["component"] for r in train])
    vgroups=np.asarray([r["component"] for r in dev])
    names=("none","flags","text_svd")
    folds={k:[] for k in names}
    baseline=[]
    for n,(it,iv) in enumerate(GroupShuffleSplit(
        n_splits=3,test_size=.2,random_state=20261108).split(groups,groups=groups),1):
        mu=float(y[it].mean())
        b=metric(y[iv],np.full(len(iv),mu),groups[iv])
        baseline.append(b)
        for kind in names:
            a,bx=metadata(xtr[it],xtr[iv],
                          [trtext[int(i)] for i in it],
                          [trtext[int(i)] for i in iv],kind)
            v=score(HistGradientBoostingRegressor(**PARAMS),
                    a,y[it],bx,y[iv],groups[iv])
            folds[kind].append(v)
        print(json.dumps({"fold":n,"train_pairs":len(it),"dev_pairs":len(iv),
                          "mean":baseline[-1],
                          **{k:folds[k][-1] for k in names},
                          "seconds":round(time.time()-start,1)}),flush=True)
    avg=lambda d,k:float(np.mean([z[k] for z in d]))
    pooled=avg(baseline,"pooled_rmse");macro=avg(baseline,"component_macro_rmse")
    ranking=[]
    for kind in names:
        p=avg(folds[kind],"pooled_rmse")
        q=avg(folds[kind],"component_macro_rmse")
        ranking.append({"model":kind,"cv_pooled_rmse":p,"cv_component_macro_rmse":q,
                        "cv_relative_joint":.5*(p/pooled+q/macro)})
    ranking.sort(key=lambda r:r["cv_relative_joint"])
    eligible=[r for r in ranking if r["cv_pooled_rmse"]<pooled
             and r["cv_component_macro_rmse"]<macro]
    best=eligible[0]["model"] if eligible else "train_mean"
    mu=float(y.mean())
    dev_base=metric(vy,np.full(len(vy),mu),vgroups)
    if best=="train_mean":
        dev_metric=dev_base
    else:
        a,b=metadata(xtr,xdev,trtext,devtext,best)
        dev_metric=score(HistGradientBoostingRegressor(**PARAMS),a,y,b,vy,vgroups)
    out={
      "status":"EXPLORATORY_TRAIN_GROUP_CV_AND_FROZEN_INNER_DEV_ONLY_NO_TEST",
      "architecture":"chemical RDKit21 plus protein-composition21 HistGB, optionally with assay description flags and train-fitted character TFIDF SVD",
      "train_pairs":len(train),"development_pairs":len(dev),
      "train_components":len(set(groups)),"development_components":len(set(vgroups)),
      "train_assay_descriptions_missing_fraction":float(np.mean([len(s)==0 for s in trtext])),
      "dev_assay_descriptions_missing_fraction":float(np.mean([len(s)==0 for s in devtext])),
      "train_assay_unique":len({(r["protein"],r["assay"]) for r in train}),
      "dev_assay_unique":len({(r["protein"],r["assay"]) for r in dev}),
      "feature_flag_terms":TERMS,
      "model_hyperparameters_fixed":PARAMS,
      "train_group_cv_baselines":baseline,
      "train_group_cv_models":folds,
      "train_group_cv_ranking":ranking,
      "train_only_cv_selected":best,
      "frozen_dev_train_mean":dev_base,
      "frozen_dev_cv_selected":dev_metric,
      "seconds":round(time.time()-start,1),
      "limitations":["A descriptive exploratory assay-conditioning comparison; current train/dev contains severe cross-assay label conflicts.",
                     "Assay text may distinguish distinct endpoints, not only technique; it is not yet a verified transferable variable.",
                     "Train-only protein-group CV used for model selection; development repeatedly inspected.",
                     "Prospective test set never scored; no research-paper superiority claim."]
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ("train_group_cv_baselines","train_group_cv_models","train_group_cv_ranking","feature_flag_terms")},indent=2),flush=True)

if __name__=="__main__":
    main()
