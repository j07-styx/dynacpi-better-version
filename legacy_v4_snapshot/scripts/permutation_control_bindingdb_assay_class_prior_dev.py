"""Shuffle assay modality WITHIN each train protein component, keeping target and label.

Tests whether assay-class mean has signal beyond balanced global label shift.
Fixed hyperparameters came from train-group CV; development is exploratory.
"""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from train_bindingdb_assay_class_prior_dev import load,prior
from train_bindingdb_cold_protein_prior_dev import metric
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"research"/"bindingdb_assay_class_prior_permutation_control_dev.json"

def main():
    tr,dev=load()
    config=json.loads((ROOT/"research/bindingdb_assay_class_prior_inner_dev.json").read_text())
    selected=config["selected_cv"]
    assert selected["mode"]=="modality" and selected["lambda"]==100. and selected["component_balanced"]
    y=np.asarray([r["label"] for r in dev])
    group=np.asarray([r["component"] for r in dev])
    pred,balanced_mean=prior(tr,dev,"modality",100.,True)
    actual=metric(y,pred,group)
    assert np.isclose(actual["pooled_rmse"],config["development_selected"]["pooled_rmse"])
    unweighted_mean=float(np.mean([r["label"] for r in tr]))
    balanced=metric(y,np.full_like(y,balanced_mean),group)
    global_baseline=metric(y,np.full_like(y,unweighted_mean),group)
    # Precompute modal class counts and protein-component sampling weights.
    train_groups=np.asarray([r["component"] for r in tr])
    train_labels=np.asarray([r["label"] for r in tr])
    categories=sorted({r["modality"] for r in tr+dev})
    index={k:i for i,k in enumerate(categories)}
    train_codes=np.asarray([index[r["modality"]] for r in tr])
    dev_codes=np.asarray([index[r["modality"]] for r in dev])
    counts=Counter(train_groups)
    weights=np.asarray([1./np.sqrt(counts[g]) for g in train_groups])
    weights=weights/weights.mean()
    assert np.isclose(np.average(train_labels,weights=weights),balanced_mean)
    indices=[np.flatnonzero(train_groups==g) for g in sorted(counts)]
    rng=np.random.default_rng(20261108)
    null=[]
    for _ in range(1000):
        shuffled=train_codes.copy()
        for ids in indices:
            shuffled[ids]=rng.permutation(shuffled[ids])
        size=np.bincount(shuffled,weights=weights,minlength=len(categories))
        total=np.bincount(shuffled,weights=weights*train_labels,minlength=len(categories))
        groupmeans=balanced_mean + (total/np.maximum(size,1e-10)-balanced_mean)*size/(size+100.)
        p=groupmeans[dev_codes]
        null.append(metric(y,p,group))
    pooled=np.asarray([q["pooled_rmse"] for q in null])
    macro=np.asarray([q["component_macro_rmse"] for q in null])
    out={
      "status":"EXPLORATORY_FIXED_ASSAY_PRIOR_WITHIN_COMPONENT_MODALITY_NEGATIVE_CONTROL_NO_TEST",
      "train_pairs":len(tr),"development_pairs":len(dev),
      "train_protein_components":len(counts),"dev_components":len(set(group)),
      "shuffle_unit":"assay-modality token per record, permuted within each train protein component",
      "n_permutations":len(null),"seed":20261108,
      "global_train_mean":unweighted_mean,"component_balanced_global_train_mean":balanced_mean,
      "global_train_mean_metrics":global_baseline,
      "balanced_global_train_mean_metrics":balanced,
      "actual_assay_modality_prior":actual,
      "permuted_pooled":{"median":float(np.median(pooled)),
                         "q025":float(np.quantile(pooled,.025)),
                         "q975":float(np.quantile(pooled,.975))},
      "permuted_macro":{"median":float(np.median(macro)),
                        "q025":float(np.quantile(macro,.025)),
                        "q975":float(np.quantile(macro,.975))},
      "empirical_lower_rmse_rank":{"pooled":float((1+np.sum(pooled<=actual["pooled_rmse"]))/(len(null)+1)),
                                   "macro":float((1+np.sum(macro<=actual["component_macro_rmse"]))/(len(null)+1))},
      "warnings":["Exploratory negative control after multiple dev inspections, not confirmatory statistical significance.",
                  "Modality parsing is heuristic and may hide target- or assay-specific context.",
                  "Reserved prospective test excluded."]
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)

if __name__=="__main__":main()
