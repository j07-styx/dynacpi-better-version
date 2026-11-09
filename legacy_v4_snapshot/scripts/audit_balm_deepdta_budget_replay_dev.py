"""Audit DeepDTA same-seed 10 versus 30 budget prediction trajectories.

This is a *replay consistency audit*, not a new model or selection rule.
GPU stochastic training trajectories can diverge even for same seed and data.
"""
import json
from pathlib import Path
import numpy as np

from dynacpi.balm_public_data_dev import ROOT

DIR=ROOT/"research/balm_deepdta_multiseed_dev_20261108"
OUT=ROOT/"research/balm_deepdta_budget_replay_divergence_audit_20261108.json"

def main():
    if OUT.exists():raise FileExistsError(OUT)
    result={
        "status":"EXPLORATORY_DEEPDTA_GPU_BUDGET_REPLAY_DIVERGENCE_AUDIT_NO_TEST",
        "scope":"Same original frozen inner DEV, identical seed, implementation, optimizer and input rows. Separate 10-epoch and 30-epoch training jobs.",
        "seed_diagnostics":{},
        "warnings":[
            "Exact same seed/initial epoch does not ensure identical later CUDA training trajectories here.",
            "The cause of divergence was not experimentally isolated. Do not assert it is necessarily CUDA nondeterminism.",
            "This confounds treating 30 epochs as literal continuation of the saved 10-epoch training trajectory.",
            "No reruns, retraining or outcome-based exclusion performed.",
            "No prospective test inputs or labels accessed.",
        ]
    }
    for seed in (2601,2602):
        a=json.loads((DIR/f"deepdta_seed{seed}_ep10.json").read_text())
        b=json.loads((DIR/f"deepdta_seed{seed}_ep30.json").read_text())
        assert a["dev_row_sha256"]==b["dev_row_sha256"]
        first=np.asarray(a["dev_predictions_by_epoch"],dtype=np.float64)
        prefix=np.asarray(b["dev_predictions_by_epoch"][:10],dtype=np.float64)
        assert first.shape==prefix.shape==(10,3891)
        meanabs=np.mean(np.abs(first-prefix),axis=1)
        maxabs=np.max(np.abs(first-prefix),axis=1)
        assert maxabs[0]==0.
        result["seed_diagnostics"][str(seed)]={
            "same_initial_epoch_predictions":True,
            "identical_predictions_all_first_10_epochs":bool(np.array_equal(first,prefix)),
            "per_epoch_mean_absolute_prediction_difference":[float(x) for x in meanabs],
            "max_abs_prediction_difference_epoch2_to10":float(max(maxabs[1:])),
            "mean_absolute_prediction_difference_epoch2_to10":float(np.mean(meanabs[1:])),
            "first_divergent_epoch":next(
                (int(i+1) for i,x in enumerate(maxabs) if x>0),None),
        }
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print("REPLAY_AUDIT",json.dumps(result),flush=True)

if __name__=="__main__":main()
