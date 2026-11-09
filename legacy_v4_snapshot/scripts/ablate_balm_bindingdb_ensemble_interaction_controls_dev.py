"""Matched negative controls for BALM fixed 25% ensemble.

Compare benefit of boosting itself versus cross-product protein interactions.
All evaluations on train-only group folds and existing frozen development.
No reserved test evaluation.
"""
import json,time
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,metric)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import (
    data_features,fit_predict,HIST_PARAMS
)

OUT=ROOT/"research/balm_bindingdb_ridge_boosting_interaction_controls_dev.json"
FIXED_BLEND=.25
def evaluate(tr,va,gen,fp_cache,ccache,pcache):
    predictions=fit_predict(tr,va,gen,fp_cache,ccache,pcache)
    c,_=data_features(tr,ccache,pcache)
    cv,_=data_features(va,ccache,pcache)
    model=HistGradientBoostingRegressor(**HIST_PARAMS)
    model.fit(c,np.array([r["label"] for r in tr],dtype=np.float64))
    ridge=predictions["ridge"]
    outputs={
        "ridge":ridge,
        "ridge_plus_chemical_boost":.75*ridge+.25*model.predict(cv),
        "ridge_plus_joint_boost":.75*ridge+.25*predictions["boost_joint"],
        "ridge_plus_explicit_interaction_boost":predictions["ridge_interaction_blend_0.25"]
    }
    return outputs

def run():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.time()
    datasets=get_train_dev()
    train=datasets["train"]
    dev=datasets["inner_dev"]
    assert len(train)==15538 and len(dev)==3891
    groups=np.array([r["group"] for r in train])
    assert set(groups).isdisjoint({r["group"] for r in dev})
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fp_cache={};ccache={};pcache={}
    folds=[]
    for i,(itr,iva) in enumerate(GroupShuffleSplit(
        n_splits=3,test_size=.2,random_state=20261108
    ).split(groups,groups=groups),1):
        tr=collapse_training([train[int(j)] for j in itr])
        val=[train[int(j)] for j in iva]
        assert set(r["group"] for r in tr).isdisjoint(set(r["group"] for r in val))
        yy=np.array([r["label"] for r in val],dtype=np.float64)
        g=[r["group"] for r in val]
        preds=evaluate(tr,val,gen,fp_cache,ccache,pcache)
        scores={k:metric(yy,v,g) for k,v in preds.items()}
        folds.append({"fold":i,"train_unique":len(tr),"heldout_rows":len(val),"results":scores})
        print("FOLD",i,json.dumps(scores),flush=True)
    names=list(folds[0]["results"])
    cv_average={n:{k:float(np.mean([f["results"][n][k] for f in folds]))
                   for k in ("pooled_rmse","component_macro_rmse")}
                for n in names}
    full=collapse_training(train)
    devpred=evaluate(full,dev,gen,fp_cache,ccache,pcache)
    yy=np.array([r["label"] for r in dev],dtype=np.float64)
    dev_scores={name:metric(yy,values,[r["group"] for r in dev])
                for name,values in devpred.items()}
    prior=json.loads((ROOT/"research/balm_bindingdb_explicit_chemical_protein_interaction_ensemble_dev.json").read_text())
    for n in ("ridge","ridge_plus_explicit_interaction_boost"):
        original="ridge" if n=="ridge" else "ridge_interaction_blend_0.25"
        assert abs(dev_scores[n]["pooled_rmse"]-
            prior["inner_dev_candidate_metrics"][original]["pooled_rmse"])<1e-8
    output={
        "status":"BALM_NEGATIVE_CONTROL_EXPLORATORY_INNER_DEV_NO_TEST",
        "hypothesis":"Is selected blend improvement attributable to explicit compound-protein interaction products rather than any boosting diversity?",
        "fixed_blend_weight":FIXED_BLEND,
        "boost_hyperparameters":HIST_PARAMS,
        "train_unique_pairs":len(full),"inner_dev_rows":len(dev),
        "train_only_groupcv_folds":folds,"cv_mean_metrics":cv_average,
        "development_metrics":dev_scores,"seconds":round(time.time()-start,2),
        "caveats":[
            "Control comparisons are post-hoc after seeing original development results.",
            "Selection of 25pct interaction blend was train-CV based, but using same hyperparameters for new controls does not make the post-hoc comparison independent.",
            "The interaction cross features are not mechanistically validated; source assay metadata missing.",
            "Protein group graph based on approximate trigram similarity; reserved prospective test untouched.",
        ]
    }
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print("FINAL",json.dumps({"cv":cv_average,"dev":dev_scores}),flush=True)

if __name__=="__main__":run()
