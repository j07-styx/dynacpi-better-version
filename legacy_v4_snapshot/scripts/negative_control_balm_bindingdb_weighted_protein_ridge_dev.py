"""Paired negative controls for CV-selected low-protein-weight Ridge.

Tests whether benefit is from true alignment of protein sequence features
with affinities, not random extra dimensions. Reserved test is inaccessible.
"""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.sparse import hstack
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,fingerprint_rows,metric)
from ablate_balm_bindingdb_frozen_chemical_protein_ridge_dev import bootstrap_components

OUT=ROOT/"research/balm_bindingdb_weighted_protein_feature_negative_control_dev.json"
def train_predict(x,y,xval,alpha):
    mu=float(np.mean(y));sd=float(np.std(y))
    reg=Ridge(alpha=alpha,solver="lsqr")
    reg.fit(x,(y-mu)/sd)
    return reg.predict(xval)*sd+mu
def main():
    start=time.time()
    config=json.loads((ROOT/"research/balm_bindingdb_frozen_protein_weighted_ridge_groupcv_dev.json").read_text())
    selected=config["cv_selected_or_chemical"]
    assert selected["name"]=="protein_scale_0.25_alpha_1.0"
    data=get_train_dev()
    train=collapse_training(data["train"]);val=data["inner_dev"]
    y=np.asarray([r["label"] for r in train]);v=np.asarray([r["label"] for r in val])
    groups=np.asarray([r["group"] for r in val])
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={}
    chem_tr=fingerprint_rows(train,gen,cache)
    chem_val=fingerprint_rows(val,gen,cache)
    tokenizer=TfidfVectorizer(analyzer="char",ngram_range=(3,4),
                    max_features=4096,sublinear_tf=True,dtype=np.float32)
    p_tr=tokenizer.fit_transform([r["protein"] for r in train])
    p_val=tokenizer.transform([r["protein"] for r in val])
    scale=.25
    X=hstack([chem_tr,p_tr*scale],format="csr")
    Z=hstack([chem_val,p_val*scale],format="csr")
    actual=train_predict(X,y,Z,1.)
    chemical_reference=train_predict(chem_tr,y,chem_val,10.)
    chemical_same_alpha=train_predict(chem_tr,y,chem_val,1.)
    a=metric(v,actual,groups)
    c=metric(v,chemical_reference,groups)
    assert np.isclose(a["pooled_rmse"],config["dev_cv_selected"]["pooled_rmse"],atol=1e-7)
    assert np.isclose(c["pooled_rmse"],config["dev_chemical_only_alpha10"]["pooled_rmse"],atol=1e-7)
    seqs=sorted({r["protein"] for r in train})
    keys={k:i for i,k in enumerate(seqs)}
    indices=np.asarray([keys[r["protein"]] for r in train])
    pool=tokenizer.transform(seqs)
    seed=20261108
    rng=np.random.default_rng(seed)
    nperm=32
    controls=[]
    for k in range(nperm):
        perm=rng.permutation(len(seqs))
        permuted_train=pool[perm[indices]]
        xx=hstack([chem_tr,permuted_train*scale],format="csr")
        p=train_predict(xx,y,Z,1.)
        controls.append(metric(v,p,groups))
        if (k+1)%8==0:
            print(json.dumps({"protein_shuffles_finished":k+1,
                              "seconds":round(time.time()-start,1)}),flush=True)
    pool_scores=np.asarray([r["pooled_rmse"] for r in controls])
    macro_scores=np.asarray([r["component_macro_rmse"] for r in controls])
    out={
      "status":"EXPLORATORY_CV_SELECTED_PROTEIN_WEIGHT_NEGATIVE_CONTROL_NO_TEST",
      "selected_parameters":selected,
      "train_unique_pairs":len(train),
      "inner_dev_rows":len(val),"inner_dev_components":len(set(groups)),
      "train_mean":metric(v,np.full_like(v,y.mean()),groups),
      "chemical_only_alpha10":c,
      "chemical_only_alpha1":metric(v,chemical_same_alpha,groups),
      "selected_weighted_chemical_protein":a,
      "paired_bootstrap_selected_vs_chemical_alpha10":
          bootstrap_components(v,groups,chemical_reference,actual),
      "n_training_protein_assignment_shuffles":nperm,
      "protein_assignment_shuffle_seed":seed,
      "null_mean_pooled":float(pool_scores.mean()),
      "null_mean_macro":float(macro_scores.mean()),
      "null_pooled_q025_q975":np.quantile(pool_scores,[.025,.975]).tolist(),
      "null_macro_q025_q975":np.quantile(macro_scores,[.025,.975]).tolist(),
      "actual_better_than_all_shuffles_pooled":bool(np.all(a["pooled_rmse"]<pool_scores)),
      "actual_better_than_all_shuffles_macro":bool(np.all(a["component_macro_rmse"]<macro_scores)),
      "observed_fraction_shuffles_worse_pooled":float(np.mean(a["pooled_rmse"]<pool_scores)),
      "observed_fraction_shuffles_worse_macro":float(np.mean(a["component_macro_rmse"]<macro_scores)),
      "shuffle_metrics":controls,
      "seconds":round(time.time()-start,1),
      "limits":["Thirty-two shuffles are descriptive and not a calibrated p-value for repeated model selection.",
                "Development is repeatedly inspected, so do not infer confirmatory significance.",
                "Protein graph split is approximate, not validated alignment homology.",
                "Reserved test set remains untouched and assay provenance remains absent."]
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ("shuffle_metrics",)},indent=2),flush=True)
if __name__=="__main__":main()
