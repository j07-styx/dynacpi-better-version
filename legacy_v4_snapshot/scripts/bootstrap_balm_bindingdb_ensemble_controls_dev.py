"""Paired component bootstrap for controlled BALM Ridge+boosting ensembles.

Only TRAIN model fits and repeatedly inspected frozen development predictions.
"""
import json,time
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import collapse_training
from scripts.ablate_balm_bindingdb_ensemble_interaction_controls_dev import evaluate

OUT=ROOT/"research/balm_bindingdb_boost_ensemble_control_group_bootstrap_dev.json"
def main():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.time()
    parts=get_train_dev()
    train=collapse_training(parts["train"])
    dev=parts["inner_dev"]
    assert len(train)==12315 and len(dev)==3891
    assert set(r["group"] for r in train).isdisjoint({r["group"] for r in dev})
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    pred=evaluate(train,dev,gen,{},{},{})
    names=list(pred)
    truth=np.asarray([r["label"] for r in dev],dtype=np.float64)
    groups=np.asarray([r["group"] for r in dev])
    ids=np.unique(groups)
    counts=np.array([(groups==g).sum() for g in ids],dtype=np.int64)
    sq=np.array([[np.sum((truth[groups==g]-pred[n][groups==g])**2)
                 for g in ids] for n in names],dtype=np.float64)
    point_pooled=np.sqrt(sq.sum(axis=1)/len(dev))
    point_macro=np.sqrt(sq/counts).mean(axis=1)
    rng=np.random.default_rng(20261108)
    sampled=rng.integers(0,len(ids),size=(5000,len(ids)))
    sampled_counts=counts[sampled].sum(axis=1)
    pooled=np.sqrt(sq[:,sampled].sum(axis=2)/sampled_counts)
    macro=np.sqrt(sq/counts)[:,sampled].mean(axis=2)
    pairs=[(n,"ridge") for n in names if n!="ridge"]
    pairs += [("ridge_plus_joint_boost","ridge_plus_chemical_boost"),
              ("ridge_plus_explicit_interaction_boost","ridge_plus_joint_boost")]
    comparisons={}
    for candidate,control in pairs:
        a=names.index(candidate);b=names.index(control)
        dp=pooled[a]-pooled[b]
        dm=macro[a]-macro[b]
        comparisons[candidate+"_minus_"+control]={
            "pooled_delta":float(point_pooled[a]-point_pooled[b]),
            "pooled_95pct_group_bootstrap_ci":[float(t) for t in np.quantile(dp,[.025,.975])],
            "macro_delta":float(point_macro[a]-point_macro[b]),
            "macro_95pct_group_bootstrap_ci":[float(t) for t in np.quantile(dm,[.025,.975])],
            "groups_candidate_better_than_control":int(np.sum(sq[a]/counts<sq[b]/counts))
        }
    output={
        "status":"EXPLORATORY_BALM_GROUP_BOOTSTRAP_CONTROLS_NO_TEST",
        "model_names":names,"dev_rows":len(dev),"heldout_protein_components":len(ids),
        "replicates":5000,"rng_seed":20261108,
        "rmse_pooled":{n:float(point_pooled[i]) for i,n in enumerate(names)},
        "rmse_macro":{n:float(point_macro[i]) for i,n in enumerate(names)},
        "paired_comparisons":comparisons,
        "seconds":round(time.time()-start,2),
        "caveats":["Retrospective negative controls after inspecting development scores; exploratory only",
                   "Shared development set, not independent testing",
                   "Protein groups reflect approximate tri-gram graph, not verified homology",
                   "Missing assay provenance and conflicting same-input labels",
                   "Reserved prospective test never evaluated"]
    }
    previous=json.loads((ROOT/"research/balm_bindingdb_ridge_boosting_interaction_controls_dev.json").read_text())
    for n in names:
        assert abs(previous["development_metrics"][n]["pooled_rmse"]-
                   output["rmse_pooled"][n])<1e-8
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(output,indent=2),flush=True)

if __name__=="__main__":main()
