"""Independent on-disk numeric verification of the exploratory ensemble experiment."""
import json,hashlib
from pathlib import Path
import numpy as np
from run_profile_study import metric
HERE=Path(__file__).resolve().parent
OUT=HERE/"interaction_study_20261109/ensemble_stack_shrink_20261109"
data=json.loads((OUT/"RESULTS.json").read_text(encoding="utf-8"))
assert data["status"]=="COMPLETE_EXPLORATORY_NESTED_ENSEMBLE_STACK_SHRINK_NOT_INDEPENDENT_TEST"
assert len(data["folds"])==3
assert data["source_sha256"]==hashlib.sha256((HERE/"run_ensemble_stack_shrink_20261109.py").read_bytes()).hexdigest()
rmses={"chemical_only":[],"anchored_prediction_ensemble":[],"inner_selected_stack":[]}
for rec in data["folds"]:
    with np.load(OUT/rec["prediction_cache_local_only"],allow_pickle=False) as z:
        y=z["truth"]
        q=z["chemical"]
        three=z["seed_predictions"]
        ens=z["ensemble"]
        stack=z["selected_stack"]
    assert y.shape==q.shape==ens.shape==stack.shape==(8,18)
    assert three.shape==(3,8,18)
    assert np.allclose(ens,three.mean(0),atol=1e-12,rtol=0)
    if "replay_against_original" in rec:
        assert len(rec["replay_against_original"])==3
        assert max(abs(x["difference"]) for x in rec["replay_against_original"])<=0.003
    else:
        # Folds 0/1 came from the resumable pre-precision-correction session;
        # check their three-seed mean against the immutable original study.
        old=json.loads((HERE/"interaction_study_20261109/study_results.json").read_text(encoding="utf-8"))
        reference=float(np.mean([old["folds"][rec["fold"]]["models"]["anchored_neural_seed"+str(seed)]["double_cold"]["macro_rmse"] for seed in (2931,2932,2933)]))
        replay=float(np.mean([metric(three[j],y)["macro_rmse"] for j in range(3)]))
        assert abs(reference-replay)<0.003,(reference,replay)
    for name,pred in (("chemical_only",q),("anchored_prediction_ensemble",ens),("inner_selected_stack",stack)):
        result=metric(pred,y)["macro_rmse"]
        expected=rec["metrics"][name]["macro_rmse"]
        assert abs(result-expected)<1e-12,(name,result,expected)
        rmses[name].append(result)
    print("VERIFIED_FOLD",rec["fold"],
        "chemical",round(rmses["chemical_only"][-1],6),
        "ensemble",round(rmses["anchored_prediction_ensemble"][-1],6),
        "stack",round(rmses["inner_selected_stack"][-1],6),
        "selected",rec["inner_selection"]["weight"],rec["inner_selection"]["gate"])
for name,values in rmses.items():
    avg=float(np.mean(values))
    ref=data["aggregates"][name]["target_macro_rmse"]
    assert abs(avg-ref)<1e-12
    print("VERIFIED_MEAN",name,round(avg,9))
print("VERIFY_SAVED_RESULTS_PASS")
