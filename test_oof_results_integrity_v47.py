"""Integrity verification of frozen v4.7 local predictions and nested scores."""
import json,hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"interaction_study_20261109/oof_residual_4inner_deterministic_20261109"

def macro(p,y):
    return float(np.sqrt(np.mean((p-y)**2,axis=1)).mean())

def main():
    result=json.loads((OUT/"RESULTS.json").read_text(encoding="utf-8"))
    assert result["status"]=="COMPLETE_EXPLORATORY_REUSED_OUTER_SPLITS"
    assert len(result["folds"])==3
    script=HERE/"run_oof_residual_v47_20261109.py"
    assert result["protocol_sha256"]==hashlib.sha256(script.read_bytes()).hexdigest()
    assert result["determinism"]["deterministic_algorithms"]
    for fold,record in enumerate(result["folds"]):
        assert record["fold"]==fold
        assert len(record["inner_validation"])==4
        assert all(rec["train_proteins"]==12 and rec["train_compounds"]>=20
                   for rec in record["inner_validation"])
        for arm in ("insample","oof"):
            chosen=record["inner_selection"][arm]
            selected=min(chosen["grid"],key=lambda row:(row["inner_macro_rmse"],row["weight"]))
            assert selected==chosen["selection"]
        with np.load(OUT/record["local_pair_cache"],allow_pickle=False) as z:
            y=z["truth"];base=z["chemical_only"]
            arms={"insample":z["in_sample_seeds"],"oof":z["oof_seeds"]}
        assert y.shape==base.shape==(8,18)
        assert all(pred.shape==(3,8,18) for pred in arms.values())
        scores=record["metrics"]["double_cold"]
        assert abs(macro(base,y)-scores["chemical_only"]["macro_rmse"])<1e-12
        for arm,pred in arms.items():
            w=record["inner_selection"][arm]["selection"]["weight"]
            ens=pred.mean(0)
            out=base+w*(ens-base)
            assert abs(macro(ens,y)-scores[arm+"_ensemble"]["macro_rmse"])<1e-12
            assert abs(macro(out,y)-scores[arm+"_inner_selected"]["macro_rmse"])<1e-12
        print("PASS_FOLD",fold,
              "baseline",round(macro(base,y),6),
              "in_sample",round(scores["insample_inner_selected"]["macro_rmse"],6),
              "oof",round(scores["oof_inner_selected"]["macro_rmse"],6))
    for regime in ("double_cold","cold_compound","cold_protein"):
        for key,stats in result["aggregates"][regime].items():
            observed=np.mean([fold["metrics"][regime][key]["macro_rmse"]
                             for fold in result["folds"]])
            assert abs(observed-stats["target_macro_rmse"])<1e-12
    print("PASS_V47_RESULTS_INTEGRITY",result["status"])

if __name__=="__main__":
    main()
