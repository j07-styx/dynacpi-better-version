"""Protein-similarity permutation negative control.

Keeps the trained component label profiles and validation examples fixed,
but permutes which training-component sequence centroid owns each profile.
Prevents claiming a meaningful protein signal from shrinkage alone.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from train_bindingdb_cold_protein_chemical_analogs_dev import load
from train_bindingdb_cold_protein_prior_dev import metric

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research/bindingdb_cold_protein_similarity_permutation_control_dev.json"


def main():
    train,dev,seqs=load() # excludes reserved prospective test by construction
    byc=defaultdict(list)
    for r in train:
        byc[r["component"]].append(r)
    components=sorted(byc)
    proteins=sorted({r["protein"] for r in train})
    dev_proteins=sorted({r["protein"] for r in dev})
    vector=TfidfVectorizer(analyzer="char",ngram_range=(3,4),
                           max_features=8192,sublinear_tf=True,dtype=np.float32)
    tr_x=normalize(vector.fit_transform([seqs[p] for p in proteins]))
    va_x=normalize(vector.transform([seqs[p] for p in dev_proteins]))
    lookup={p:i for i,p in enumerate(proteins)}
    centroid=[]
    component_mean=[]
    for c in components:
        rows=byc[c]
        ps=sorted({r["protein"] for r in rows})
        centroid.append(tr_x[[lookup[p] for p in ps]].mean(axis=0))
        component_mean.append(float(np.mean([
            np.mean([r["label"] for r in rows if r["protein"]==p])
            for p in ps])))
    centroid=normalize(vstack([csr_matrix(x) for x in centroid]))
    sim=np.asarray((va_x @ centroid.T).toarray(),dtype=np.float64)
    means=np.asarray(component_mean)
    mu=float(np.mean([r["label"] for r in train]))
    dev_index={p:i for i,p in enumerate(dev_proteins)}
    sample_index=np.asarray([dev_index[r["protein"]] for r in dev])
    y=np.asarray([r["label"] for r in dev])
    groups=np.asarray([r["component"] for r in dev])

    def predict(s):
        ids=np.argsort(-s,axis=1,kind="stable")[:,:3]
        x=np.maximum(0,np.take_along_axis(s,ids,axis=1))
        weights=np.maximum(x**4,1e-12)
        neighbors=(means[ids]*weights).sum(axis=1)/weights.sum(axis=1)
        protein_prediction=mu+np.max(x,axis=1)**2*(neighbors-mu)
        return protein_prediction[sample_index]

    obs=predict(sim)
    actual=metric(y,obs,groups)
    prior_reference=json.loads((ROOT/"research/bindingdb_cold_protein_sequence_prior_inner_dev.json").read_text())["dev_safe_pooled_model"]
    assert np.isclose(actual["pooled_rmse"],prior_reference["pooled_rmse"],atol=1e-7)
    assert np.isclose(actual["component_macro_rmse"],prior_reference["component_macro_rmse"],atol=1e-7)
    rng=np.random.default_rng(20261108)
    nperms=1000
    results=[]
    for j in range(nperms):
        perm=rng.permutation(len(components))
        scrambled=sim[:,perm]
        results.append(metric(y,predict(scrambled),groups))
    rmse=np.asarray([r["pooled_rmse"] for r in results])
    macro=np.asarray([r["component_macro_rmse"] for r in results])
    out={
        "status":"DEVELOPMENT_PERMUTATION_NEGATIVE_CONTROL_NOT_INDEPENDENT_TEST",
        "seed":20261108,
        "number_of_permutations":nperms,
        "permutation_unit":"training-protein-component sequence-prototype assignment, fixed train label means",
        "train_protein_components":len(components),
        "dev_protein_components":len(set(groups)),
        "train_pairs":len(train),"development_pairs":len(dev),
        "observed":actual,
        "permuted_pooled_rmse":{"mean":float(rmse.mean()),"median":float(np.median(rmse)),
                                "q025":float(np.quantile(rmse,.025)),"q975":float(np.quantile(rmse,.975))},
        "permuted_component_macro_rmse":{"mean":float(macro.mean()),"median":float(np.median(macro)),
                                "q025":float(np.quantile(macro,.025)),"q975":float(np.quantile(macro,.975))},
        "empirical_p_lower_rmse_better":{"pooled":float((1+np.sum(rmse<=actual["pooled_rmse"]))/(nperms+1)),
                                          "macro":float((1+np.sum(macro<=actual["component_macro_rmse"]))/(nperms+1))},
        "warnings":[
          "An exploratory negative control on a repeatedly inspected development set, NOT confirmatory significance.",
          "Permuting centroids breaks potential correlations between training component size, protein sequence, and label profile.",
          "A low empirical rank against randomized identities does not demonstrate biological causality or paper-level performance.",
          "No prospective test labels or predictions accessed; provisional assay and construct provenance."
        ]
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)


if __name__=="__main__":
    main()
