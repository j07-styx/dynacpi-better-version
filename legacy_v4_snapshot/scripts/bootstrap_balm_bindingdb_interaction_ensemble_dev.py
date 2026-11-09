"""Protein-component paired bootstrap for the locked BALM ridge+interaction blend.

Fits train only and does NOT access reserved test. Bootstrap is a development
uncertainty diagnostic, not an independent validation claim.
"""
import json
import time
from collections import defaultdict

import numpy as np
from rdkit.Chem import rdFingerprintGenerator

from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import collapse_training
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import fit_predict,BLENDS

OUT=ROOT/"research/balm_bindingdb_interaction_ensemble_paired_group_bootstrap_dev.json"
def main():
    if OUT.exists():raise FileExistsError(OUT)
    tic=time.time()
    parts=get_train_dev()
    train=collapse_training(parts["train"])
    dev=parts["inner_dev"]
    assert len(train)==12315 and len(dev)==3891
    assert set(r["group"] for r in train).isdisjoint({r["group"] for r in dev})
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    preds=fit_predict(train,dev,gen,{},{},{})
    names=("ridge","ridge_interaction_blend_0.25","boost_explicit_interaction")
    truth=np.array([r["label"] for r in dev],dtype=np.float64)
    groups=np.array([r["group"] for r in dev])
    all_groups=np.unique(groups)
    n=len(all_groups)
    counts=np.array([(groups==g).sum() for g in all_groups],dtype=np.int64)
    sq=np.array([
        [np.sum(np.square(truth[groups==g]-preds[k][groups==g])) for g in all_groups]
        for k in names
    ],dtype=np.float64)
    pergroup_rmse=np.sqrt(sq/counts[None,:])
    num_wins=int(np.sum(pergroup_rmse[1]<pergroup_rmse[0]))
    point_pooled=np.sqrt(np.sum(sq,axis=1)/counts.sum())
    point_macro=pergroup_rmse.mean(axis=1)
    rng=np.random.default_rng(20261108)
    repetitions=5000
    indices=rng.integers(0,n,size=(repetitions,n),dtype=np.int32)
    nresample=counts[indices].sum(axis=1)
    boot_pooled=np.sqrt(sq[:,indices].sum(axis=2)/nresample[None,:])
    boot_macro=pergroup_rmse[:,indices].mean(axis=2)
    delta_pooled=boot_pooled[1]-boot_pooled[0]
    delta_macro=boot_macro[1]-boot_macro[0]
    output={
        "status":"BALM_INNER_DEV_PAIRED_GROUP_BOOTSTRAP_NO_RESERVED_TEST",
        "locked_train_cv_selected_ensemble":"75pct ridge + 25pct interaction boosted descriptors",
        "dev_rows":len(dev),"dev_protein_graph_components":n,
        "group_count_min":int(counts.min()),"group_count_max":int(counts.max()),
        "group_macro_wins_vs_ridge":num_wins,
        "group_macro_losses_vs_ridge":int(n-num_wins-np.sum(pergroup_rmse[1]==pergroup_rmse[0])),
        "point_estimates":{
            name:{"pooled_rmse":float(point_pooled[i]),
                  "macro_component_rmse":float(point_macro[i])}
            for i,name in enumerate(names)
        },
        "paired_delta_ensemble_minus_ridge":{
            "pooled_rmse":float(point_pooled[1]-point_pooled[0]),
            "macro_component_rmse":float(point_macro[1]-point_macro[0])
        },
        "bootstrap":{
            "method":"nonparametric resample of held-out protein graph components with replacement",
            "seed":20261108,"replicates":repetitions,
            "pooled_rmse_delta_95pct_percentile_CI":
                [float(v) for v in np.quantile(delta_pooled,[.025,.975])],
            "macro_rmse_delta_95pct_percentile_CI":
                [float(v) for v in np.quantile(delta_macro,[.025,.975])],
            "bootstrap_probability_pooled_delta_below_zero":float(np.mean(delta_pooled<0)),
            "bootstrap_probability_macro_delta_below_zero":float(np.mean(delta_macro<0))
        },
        "seconds":round(time.time()-tic,2),
        "caveats":[
            "Development data have been inspected repeatedly and are not a fresh holdout.",
            "Bootstrap groups are approximate protein trigram graph components, not verified homology groups.",
            "Assay context is missing in source, and same-input label conflicts remain.",
            "No independent prospective test evaluation.",
        ]
    }
    expected=json.loads((ROOT/"research/balm_bindingdb_explicit_chemical_protein_interaction_ensemble_dev.json").read_text())
    for key in ("ridge","ridge_interaction_blend_0.25"):
        assert abs(output["point_estimates"][key]["pooled_rmse"]-
              expected["inner_dev_candidate_metrics"][key]["pooled_rmse"])<1e-8
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(output,indent=2),flush=True)

if __name__=="__main__":main()
