"""Matched Ridge ablations and representation shuffles on BALM cold-protein DEV.

Train-only duplicate averaging and vocabulary. 10 independently fixed
protein/compound identity shuffles, never touching reserved-test labels.
"""
import json,time
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,fingerprint_rows,metric)

OUT=ROOT/"research/balm_bindingdb_frozen_paired_input_permutation_ablation_dev.json"

def fit(xtrain,xval,ytr,mean,std):
    model=Ridge(alpha=10.0,solver="lsqr")
    model.fit(xtrain,(ytr-mean)/std)
    return model.predict(xval)*std+mean

def bootstrap_components(y,groups,baseline,pred,ns=10000):
    groups=np.asarray(groups)
    unique=sorted(set(groups))
    count=[];sse_baseline=[];sse_pred=[];rmsediff=[]
    for c in unique:
        mask=(groups==c)
        n=int(mask.sum())
        u=float(np.sum((y[mask]-baseline[mask])**2))
        v=float(np.sum((y[mask]-pred[mask])**2))
        count.append(n);sse_baseline.append(u);sse_pred.append(v)
        rmsediff.append(np.sqrt(u/n)-np.sqrt(v/n))
    count=np.asarray(count,dtype=np.float64)
    sse_baseline=np.asarray(sse_baseline)
    sse_pred=np.asarray(sse_pred)
    rng=np.random.default_rng(20261108)
    inds=rng.integers(0,len(unique),size=(ns,len(unique)))
    pool=np.sqrt(sse_baseline[inds].sum(axis=1)/count[inds].sum(axis=1))-np.sqrt(
        sse_pred[inds].sum(axis=1)/count[inds].sum(axis=1))
    macro=np.asarray(rmsediff)[inds].mean(axis=1)
    return {"number_components":len(unique),"n_resamples":ns,
       "pooled_improvement_bootstrap_q025_q975":np.quantile(pool,[.025,.975]).tolist(),
       "macro_improvement_bootstrap_q025_q975":np.quantile(macro,[.025,.975]).tolist(),
       "fraction_pooled_improvement":float(np.mean(pool>0)),
       "fraction_macro_improvement":float(np.mean(macro>0))}

def main():
    start=time.time()
    parts=get_train_dev()
    tr=collapse_training(parts["train"]);va=parts["inner_dev"]
    y=np.asarray([r["label"] for r in tr])
    truth=np.asarray([r["label"] for r in va])
    group=np.asarray([r["group"] for r in va])
    mu=float(y.mean());std=float(y.std());assert std>0
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={}
    chem_tr=fingerprint_rows(tr,gen,cache)
    chem_va=fingerprint_rows(va,gen,cache)
    protein_vec=TfidfVectorizer(analyzer="char",ngram_range=(3,4),
                                max_features=4096,sublinear_tf=True,dtype=np.float32)
    prot_tr=protein_vec.fit_transform([r["protein"] for r in tr])
    prot_va=protein_vec.transform([r["protein"] for r in va])
    names=["chemical","protein","combined"]
    actual={}
    preds={}
    matrices={
      "chemical":(chem_tr,chem_va),
      "protein":(prot_tr,prot_va),
      "combined":(hstack([chem_tr,prot_tr],format="csr"),
                  hstack([chem_va,prot_va],format="csr"))
    }
    for name,(a,b) in matrices.items():
        pred=fit(a,b,y,mu,std)
        preds[name]=pred
        actual[name]=metric(truth,pred,group)
        print(json.dumps({"model":name,"score":actual[name]}),flush=True)
    globalmean=np.full_like(truth,mu)
    prior=metric(truth,globalmean,group)
    protseq=sorted({r["protein"] for r in tr})
    protindex={p:i for i,p in enumerate(protseq)}
    protidx=np.asarray([protindex[r["protein"]] for r in tr])
    proto=protein_vec.transform(protseq)
    chemseq=sorted({r["smiles"] for r in tr})
    chemindex={s:i for i,s in enumerate(chemseq)}
    chemidx=np.asarray([chemindex[r["smiles"]] for r in tr])
    unique_smiles_rows=[{"smiles":s} for s in chemseq]
    chemproto=fingerprint_rows(unique_smiles_rows,gen,cache)
    rng=np.random.default_rng(20261108)
    nulls={"protein_identity_permuted_training":[],"chemical_identity_permuted_training":[]}
    for n in range(12):
        pmap=rng.permutation(len(protseq))
        assert np.any(pmap!=np.arange(len(protseq)))
        shuffled_p=proto[pmap[protidx]]
        pred=fit(hstack([chem_tr,shuffled_p],format="csr"),
                 matrices["combined"][1],y,mu,std)
        nulls["protein_identity_permuted_training"].append(metric(truth,pred,group))
        cmap=rng.permutation(len(chemseq))
        shuffled_chem=chemproto[cmap[chemidx]]
        pred2=fit(hstack([shuffled_chem,prot_tr],format="csr"),
                  matrices["combined"][1],y,mu,std)
        nulls["chemical_identity_permuted_training"].append(metric(truth,pred2,group))
        if (n+1)%4==0:
            print(json.dumps({"shuffles_completed":n+1,"elapsed_sec":round(time.time()-start,1)}),flush=True)
    summary={}
    for name,runs in nulls.items():
        pool=np.asarray([r["pooled_rmse"] for r in runs])
        macro=np.asarray([r["component_macro_rmse"] for r in runs])
        summary[name]={
         "n_permutations":len(runs),
         "pooled_mean":float(pool.mean()),
         "pooled_q025_q975":np.quantile(pool,[.025,.975]).tolist(),
         "macro_mean":float(macro.mean()),
         "macro_q025_q975":np.quantile(macro,[.025,.975]).tolist(),
         "actual_better_than_all_shuffled_pooled":bool(np.all(
             actual["combined"]["pooled_rmse"]<pool)),
         "actual_better_than_all_shuffled_macro":bool(np.all(
             actual["combined"]["component_macro_rmse"]<macro))
        }
    compare={
      "combined_minus_chemical_pooled_rmse":
           actual["combined"]["pooled_rmse"]-actual["chemical"]["pooled_rmse"],
      "combined_minus_chemical_macro_rmse":
           actual["combined"]["component_macro_rmse"]-actual["chemical"]["component_macro_rmse"],
      "combined_minus_protein_pooled_rmse":
           actual["combined"]["pooled_rmse"]-actual["protein"]["pooled_rmse"]
    }
    out={
      "status":"EXPLORATORY_BALM_COLD_PROTEIN_INPUT_ABLATION_NO_TEST_LABEL_ACCESS",
      "train_unique_pairs":len(tr),"dev_original_rows":len(va),
      "train_only_alpha":10.,
      "feature_design":"Morgan chiral2048 L2, protein 3-4mer TFIDF4096 trained on train only, targets standardized using train labels",
      "train_mean":prior,"actual_models":actual,"paired_comparisons":compare,
      "paired_component_bootstrap_combined_vs_chemical":bootstrap_components(
             truth,group,preds["chemical"],preds["combined"]),
      "paired_component_bootstrap_combined_vs_train_mean":bootstrap_components(
             truth,group,globalmean,preds["combined"]),
      "permuted_feature_nulls":summary,
      "permuted_run_metrics":nulls,
      "seconds":round(time.time()-start,1),
      "warnings":["Exploratory internal-development results; repeated looks, no independent test claims.",
                  "Twelve randomized identities are a diagnostic, not a calibrated randomization significance test.",
                  "Protein 3mer-graph split is approximate and not validated alignment homology.",
                  "No assay/source provenance in public Arrow cohort; same-input discordance persists.",
                  "Prospective test labels not accessed."]
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ("permuted_run_metrics",)},indent=2),flush=True)

if __name__=="__main__":main()
