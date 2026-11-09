"""Hierarchically shrunken assay-class mean: a simple modality shift control.

Protein-component CV within TRAIN chooses granularity and shrinkage.
Inner-development is scored only once after selection. Test untouched.
"""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from ablate_bindingdb_assay_metadata_identifiability_dev import read,signature
from train_bindingdb_cold_protein_prior_dev import metric

ROOT=Path(__file__).resolve().parents[1];A=ROOT/"data"/"audit"
OUT=ROOT/"research"/"bindingdb_assay_class_prior_inner_dev.json"


def load():
    manifest=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    tr=set(manifest["train_components"])
    dv=set(manifest["inner_dev_components"])
    te=set(manifest["test_components"])
    assert not tr&dv and not tr&te and not dv&te
    cmap={}
    for g in read(A/"bindingdb_full_kd_protein_components.jsonl"):
        for s in g["members"]:cmap[s]=g["component_id"]
    allowed={(r["sequence_sha256"],r["assay_key"]) for r in read(
        A/"bindingdb_full_kd_task_summary.jsonl") if r["unique_canonical_compounds"]>=15}
    train=[];dev=[]
    for r in read(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if (r["protein_seq_sha256"],r["assay_key"]) not in allowed or r["range_pKd"]>.1:continue
        c=cmap[r["protein_seq_sha256"]]
        if c not in tr and c not in dv:continue
        row={"label":float(r["median_pKd"]),"component":c,
             "assay_key":(r["protein_seq_sha256"],r["assay_key"])}
        (train if c in tr else dev).append(row)
    assert len(train)==manifest["train_pairs"] and len(dev)==manifest["inner_dev_pairs"]
    wanted={r["assay_key"] for r in train+dev}
    contexts={}
    for r in read(A/"bindingdb_full_kd_assay_contexts.jsonl"):
        key=r["sequence_sha256"],r["assay_key"]
        if key in wanted:contexts[key]=signature(r["description"] or "")
    assert len(contexts)==len(wanted)
    for row in train+dev:
        row["endpoint"],row["modality"]=contexts[row.pop("assay_key")]
    return train,dev

def prior(train,val,mode,lambda_,component_balanced=False):
    y=np.asarray([r["label"] for r in train])
    group=np.asarray([r["component"] for r in train])
    if component_balanced:
        counts=Counter(group)
        w=np.asarray([1./np.sqrt(counts[g]) for g in group])
    else:
        w=np.ones(len(train))
    w=w/w.mean()
    global_mean=float(np.average(y,weights=w))
    if mode=="endpoint":
        key=lambda r:r["endpoint"]
    elif mode=="modality":
        key=lambda r:r["modality"]
    elif mode=="both":
        key=lambda r:(r["endpoint"],r["modality"])
    else:
        raise ValueError(mode)
    sums=defaultdict(float)
    sizes=defaultdict(float)
    for r,weight in zip(train,w):
        k=key(r)
        sums[k]+=r["label"]*weight
        sizes[k]+=weight
    means={k:global_mean+(sums[k]/sizes[k]-global_mean)*sizes[k]/(sizes[k]+lambda_) for k in sums}
    return np.asarray([means.get(key(r),global_mean) for r in val]),global_mean

def main():
    train,dev=load()
    groups=np.asarray([r["component"] for r in train])
    configs=[(mode,lam,balanced) for mode in ("endpoint","modality","both")
             for lam in (10.,100.,1000.,10000.) for balanced in (False,True)]
    baselines=[]
    results={str(c):[] for c in configs}
    for i,(itr,iva) in enumerate(
        GroupShuffleSplit(n_splits=3,test_size=.2,random_state=20261108).split(
        groups,groups=groups),1):
        tr=[train[int(j)] for j in itr]
        va=[train[int(j)] for j in iva]
        y=np.asarray([r["label"] for r in va]);g=np.asarray([r["component"] for r in va])
        mu=float(np.mean([r["label"] for r in tr]))
        base=metric(y,np.full_like(y,mu),g)
        baselines.append(base)
        for config in configs:
            pred,_=prior(tr,va,*config)
            results[str(config)].append(metric(y,pred,g))
        print(json.dumps({"fold":i,"train_pairs":len(tr),"val_pairs":len(va),"mean":base}),flush=True)
    avg=lambda xs,k:float(np.mean([r[k] for r in xs]))
    mp=avg(baselines,"pooled_rmse");mm=avg(baselines,"component_macro_rmse")
    ranked=[]
    for c in configs:
        stats=results[str(c)]
        p=avg(stats,"pooled_rmse");m=avg(stats,"component_macro_rmse")
        ranked.append({"mode":c[0],"lambda":c[1],"component_balanced":c[2],
                       "cv_pooled_rmse":p,"cv_macro_rmse":m,
                       "relative_joint":.5*(p/mp+m/mm)})
    ranked.sort(key=lambda z:(z["relative_joint"],z["cv_pooled_rmse"]))
    eligible=[r for r in ranked if r["cv_pooled_rmse"]<=mp and r["cv_macro_rmse"]<=mm]
    selected=eligible[0] if eligible else None
    y=np.asarray([r["label"] for r in dev]);g=np.asarray([r["component"] for r in dev])
    mu=float(np.mean([r["label"] for r in train]))
    baseline=metric(y,np.full_like(y,mu),g)
    if selected:
        pred,_=prior(train,dev,selected["mode"],selected["lambda"],selected["component_balanced"])
        observed=metric(y,pred,g)
    else:observed=baseline
    report={"status":"EXPLORATORY_ASSAY_CLASS_PRIOR_TRAIN_GROUP_CV_AND_INNER_DEV_NO_TEST",
            "training_pairs":len(train),"dev_pairs":len(dev),
            "training_components":len(set(groups)),"dev_components":len(set(g)),
            "candidate_models":len(configs),
            "baseline_cv_pooled_mean":mp,"baseline_cv_macro_mean":mm,
            "cv_best_ranked":ranked[:15],"selected_cv":selected or "train_only_mean",
            "development_train_only_mean":baseline,"development_selected":observed,
            "warnings":["Keywords and assay protocol are not verified harmonized endpoints.",
                        "No prospective test labels used for architecture selection or evaluation.",
                        "Development has been inspected in prior experiments; exploratory only."]}
    OUT.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="cv_best_ranked"},indent=2),flush=True)

if __name__=="__main__":
    main()
