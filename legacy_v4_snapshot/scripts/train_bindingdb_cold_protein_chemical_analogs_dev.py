"""Cold-protein nearest-chemical analog correction, trained by group-only CV.

No reserved prospective-test labels or features used. Component-conditional
chemical residuals correct a conservative protein sequence prior, with
noninferiority gates selected only within training-group CV.
"""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

from train_bindingdb_cold_protein_prior_dev import read, metric

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "audit"
OUT = ROOT / "research" / "bindingdb_cold_protein_conditional_chemical_analog_dev.json"
RDLogger.DisableLog("rdApp.*")


def load():
    manifest=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    tr=set(manifest["train_components"])
    dv=set(manifest["inner_dev_components"])
    te=set(manifest["test_components"])
    assert tr.isdisjoint(dv|te) and dv.isdisjoint(te)
    comp={}
    for row in read(A/"bindingdb_full_kd_protein_components.jsonl"):
        for s in row["members"]:
            comp[s]=row["component_id"]
    tasks={(r["sequence_sha256"],r["assay_key"])
           for r in read(A/"bindingdb_full_kd_task_summary.jsonl")
           if r["unique_canonical_compounds"]>=15}
    train=[]
    dev=[]
    for row in read(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if ((row["protein_seq_sha256"],row["assay_key"]) not in tasks
            or row["range_pKd"]>.1):
            continue
        c=comp[row["protein_seq_sha256"]]
        if c not in tr and c not in dv:
            continue
        item={"protein":row["protein_seq_sha256"],"component":c,
              "compound":row["compound_sha256"],"smiles":row["smiles_isomeric"],
              "label":float(row["median_pKd"])}
        (train if c in tr else dev).append(item)
    assert len(train)==manifest["train_pairs"] and len(dev)==manifest["inner_dev_pairs"]
    needed={x["protein"] for x in train+dev}
    seqs={}
    for row in read(A/"bindingdb_full_kd_candidates.jsonl"):
        s=row["sequence"]
        k=hashlib.sha256(s.encode()).hexdigest()
        if k in needed:
            seqs[k]=s
        if len(seqs)==len(needed):
            break
    assert len(seqs)==len(needed)
    return train,dev,seqs


def build_prior_and_chemical_analogs(training, validation, seqs, fp):
    byc=defaultdict(list)
    for row in training:
        byc[row["component"]].append(row)
    components=sorted(byc)
    mu=float(np.mean([r["label"] for r in training]))
    prots=sorted({r["protein"] for r in training})
    val_prots=sorted({r["protein"] for r in validation})
    vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(3,4),
                               max_features=8192,sublinear_tf=True,dtype=np.float32)
    xtr=normalize(vectorizer.fit_transform([seqs[p] for p in prots]))
    xva=normalize(vectorizer.transform([seqs[p] for p in val_prots]))
    idx={p:i for i,p in enumerate(prots)}
    centroids=[]
    comp_means=[]
    # For chemical analog residuals: one value per compound per component.
    comp_fps=[]
    comp_residuals=[]
    for c in components:
        rows=byc[c]
        cprots=sorted({r["protein"] for r in rows})
        centroids.append(xtr[[idx[p] for p in cprots]].mean(axis=0))
        pmean=[np.mean([r["label"] for r in rows if r["protein"]==p])
               for p in cprots]
        cmean=float(np.mean(pmean))
        comp_means.append(cmean)
        bymol=defaultdict(list)
        for r in rows:
            bymol[r["compound"]].append(r["label"]-cmean)
        mols=sorted(bymol)
        comp_fps.append([fp[m] for m in mols])
        comp_residuals.append(np.asarray(
            [np.median(bymol[m]) for m in mols],dtype=np.float32))
    centroids=normalize(vstack([csr_matrix(m) for m in centroids]))
    similarity=(xva @ centroids.T).toarray()
    mean_by_protein={}
    analog_by_pair={}
    for p_i,p in enumerate(val_prots):
        row=similarity[p_i]
        top=np.argsort(-row,kind="stable")[:3]
        p_sim=np.maximum(0,row[top])
        mean_by_protein[p]=(float(np.average(
            np.asarray(comp_means)[top],
            weights=np.maximum(p_sim**4,1e-12))),float(p_sim.max()),
            top,p_sim)
    # Cache unique (protein, compound) pairs since assays can repeat.
    for row in validation:
        key=(row["protein"],row["compound"])
        if key in analog_by_pair:
            continue
        raw_prior,smax,top,p_sim=mean_by_protein[row["protein"]]
        residuals=[]
        chem_sim=[]
        for c_index in top:
            # Tanimoto nearest molecule within *training* protein component.
            similarities=np.asarray(
                DataStructs.BulkTanimotoSimilarity(fp[row["compound"]],
                                                   comp_fps[int(c_index)]),
                dtype=np.float32)
            # top 3 analogs smooth out assay noise.
            best=np.argsort(-similarities,kind="stable")[:3]
            top_s=similarities[best]
            w=np.maximum(top_s,1e-8)**4
            residuals.append(float(np.average(comp_residuals[int(c_index)][best],
                                              weights=w)))
            chem_sim.append(float(top_s.max()))
        residuals=np.asarray(residuals)
        chem_sim=np.asarray(chem_sim)
        weights=np.maximum(p_sim,1e-8)**4*np.maximum(chem_sim,1e-8)**2
        chem_res=float(np.average(residuals,weights=weights))
        chem_conf=float(np.max(chem_sim))
        analog_by_pair[key]=(chem_res,chem_conf,smax)
    base=np.asarray([mu+mean_by_protein[r["protein"]][1]**2*
                  (mean_by_protein[r["protein"]][0]-mu)
                  for r in validation],dtype=np.float64)
    chem=np.asarray([analog_by_pair[(r["protein"],r["compound"])][0]
                     for r in validation],dtype=np.float64)
    chemconf=np.asarray([analog_by_pair[(r["protein"],r["compound"])][1]
                         for r in validation],dtype=np.float64)
    protconf=np.asarray([analog_by_pair[(r["protein"],r["compound"])][2]
                         for r in validation],dtype=np.float64)
    return base,chem,chemconf,protconf


def main():
    start=time.time()
    train,dev,seqs=load()
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    smiles={}
    for row in train+dev:
        old=smiles.setdefault(row["compound"],row["smiles"])
        assert old==row["smiles"]
    fp={}
    for k,sm in smiles.items():
        mol=Chem.MolFromSmiles(sm)
        if mol is None:raise ValueError("Invalid canonical SMILES "+k)
        fp[k]=gen.GetFingerprint(mol)
    groups=np.asarray([r["component"] for r in train])
    folds=[]
    for fold,(it,iv) in enumerate(GroupShuffleSplit(
          n_splits=3,test_size=.2,random_state=20261108).split(groups,groups=groups),1):
        tr=[train[int(i)] for i in it]
        va=[train[int(i)] for i in iv]
        prior,chemical,csim,psim=build_prior_and_chemical_analogs(tr,va,seqs,fp)
        y=np.asarray([r["label"] for r in va])
        g=np.asarray([r["component"] for r in va])
        mean=float(np.mean([r["label"] for r in tr]))
        folds.append((y,g,mean,prior,chemical,csim,psim))
        print(json.dumps({"fold":fold,"train_pairs":len(tr),"val_pairs":len(va),
                          "mean":metric(y,np.full_like(y,mean),g),
                          "prior":metric(y,prior,g),
                          "chemical_residual_q10_q50_q90":np.quantile(chemical,[.1,.5,.9]).tolist(),
                          "elapsed_s":round(time.time()-start,1)}),flush=True)
    params=[(scale,chem_pow,prot_pow,cap)
            for scale in (0.,.1,.25,.5,1.)
            for chem_pow in (0,2,4)
            for prot_pow in (0,2)
            for cap in (.25,.5,1.)]
    base_pooled=float(np.mean([metric(y,np.full_like(y,mean),g)["pooled_rmse"]
                                for y,g,mean,_,_,_,_ in folds]))
    base_macro=float(np.mean([metric(y,np.full_like(y,mean),g)["component_macro_rmse"]
                               for y,g,mean,_,_,_,_ in folds]))
    prior_pooled=float(np.mean([metric(y,prior,g)["pooled_rmse"]
                                 for y,g,mean,prior,*_ in folds]))
    prior_macro=float(np.mean([metric(y,prior,g)["component_macro_rmse"]
                                for y,g,mean,prior,*_ in folds]))
    ranked=[]
    for scale,chem_pow,prot_pow,cap in params:
        scores=[]
        for y,g,mean,prior,chemical,csim,psim in folds:
            gated=scale*np.clip(chemical,-cap,cap)*csim**chem_pow*psim**prot_pow
            scores.append(metric(y,prior+gated,g))
        pool=float(np.mean([v["pooled_rmse"] for v in scores]))
        macro=float(np.mean([v["component_macro_rmse"] for v in scores]))
        ranked.append({"scale":scale,"chem_power":chem_pow,
                       "protein_power":prot_pow,"residual_cap":cap,
                       "cv_pooled_rmse":pool,"cv_macro_rmse":macro,
                       "cv_joint_relative_error":.5*(pool/base_pooled+macro/base_macro)})
    ranked.sort(key=lambda x:(x["cv_joint_relative_error"],x["cv_pooled_rmse"]))
    # Require training-only CV improvement on BOTH metrics over conservative prior,
    # otherwise deploy prior without correction.
    eligible=[r for r in ranked if r["cv_pooled_rmse"]<prior_pooled
              and r["cv_macro_rmse"]<prior_macro and r["scale"]>0]
    best=eligible[0] if eligible else None
    base,chem,csim,psim=build_prior_and_chemical_analogs(train,dev,seqs,fp)
    y=np.asarray([r["label"] for r in dev])
    g=np.asarray([r["component"] for r in dev])
    mean=float(np.mean([r["label"] for r in train]))
    if best:
        gated=best["scale"]*np.clip(chem,-best["residual_cap"],best["residual_cap"])*csim**best["chem_power"]*psim**best["protein_power"]
        selected=base+gated
    else:
        selected=base
    b=metric(y,np.full_like(y,mean),g)
    p=metric(y,base,g)
    v=metric(y,selected,g)
    component_scores=[]
    for c in np.unique(g):
        mask=g==c
        component_scores.append({
            "pair_count":int(mask.sum()),
            "mean_rmse":float(np.sqrt(np.mean((y[mask]-mean)**2))),
            "prior_rmse":float(np.sqrt(np.mean((y[mask]-base[mask])**2))),
            "selected_rmse":float(np.sqrt(np.mean((y[mask]-selected[mask])**2)))})
    output={
        "status":"EXPLORATORY_TRAIN_CV_AND_INNER_DEV_ONLY_NO_PROSPECTIVE_TEST",
        "model":"top3 protein similarity priors plus bounded chemical nearest-analog residual within training component",
        "train_pairs":len(train),"inner_dev_pairs":len(dev),"train_components":len(set(groups)),
        "validation_components":len(set(g)),
        "train_only_cv_mean_baseline":{"pooled":base_pooled,"macro":base_macro},
        "train_only_cv_conservative_protein_prior":{"pooled":prior_pooled,"macro":prior_macro},
        "train_only_cv_top_hybrids":ranked[:20],
        "cv_selected_hybrid_or_prior":best if best else "conservative_protein_prior_without_chemistry",
        "inner_dev_train_mean":b,"inner_dev_conservative_prior":p,
        "inner_dev_cv_selected_model":v,
        "components_better_than_prior":int(sum(
             x["selected_rmse"]<x["prior_rmse"] for x in component_scores)),
        "components_better_than_mean":int(sum(
             x["selected_rmse"]<x["mean_rmse"] for x in component_scores)),
        "component_scores_anonymous":sorted(component_scores,key=lambda x:x["pair_count"],reverse=True),
        "elapsed_s":round(time.time()-start,1),
        "limitations":["Training-only CV chooses correction; development set already repeatedly inspected.",
                       "No reserved test labels or predictions used.",
                       "Assay and construct provenance incomplete.",
                       "Protein 3-4mer TF-IDF similarity is not guaranteed homologous construct identity."]
    }
    OUT.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items()
                      if k not in ("component_scores_anonymous","train_only_cv_top_hybrids")},indent=2),flush=True)


if __name__=="__main__":
    main()
