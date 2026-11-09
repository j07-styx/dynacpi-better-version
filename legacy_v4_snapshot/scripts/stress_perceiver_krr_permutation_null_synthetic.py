"""Fabricated comparison of old LOO gate vs support-permutation calibrated gate.

Runs only on synthetic labels emitted by the preexisting fabricated-stress
fixture. No actual BindingDB tasks/labels, checkpoints or reserved TEST data.
Never overwrite the public output and never optimize guard hyperparameters
on these fake trials.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from scripts.stress_perceiver_krr_guard_synthetic import (
    fake_trial,SMILES,N_TRIALS,SEED
)
from dynacpi.perceiver_krr_geometry_guard import compare_krr_geometry_with_support_validation
from dynacpi.perceiver_krr_support_permutation_guard import (
    PermutationGuardSettings,compare_krr_with_support_permutation_guard
)

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"research/perceiver_krr_support_permutation_fabricated_stress_20261109.json"


def synthetic_permutation_stress(n_trials=N_TRIALS,seed=SEED):
    if not isinstance(n_trials,int) or isinstance(n_trials,bool) or n_trials<16:
        raise ValueError("At least sixteen fabricated trials required")
    if not isinstance(seed,int) or isinstance(seed,bool) or seed<0:
        raise ValueError("Invalid fabricated trial seed")
    results={}
    settings=PermutationGuardSettings()
    for scenario in ("geometry_signal","geometry_null"):
        for k in (5,6):
            rng=np.random.default_rng(seed+int(k)*1000+
                (0 if scenario=="geometry_signal" else 1))
            sq={arm:[] for arm in (
                "perceiver_krr","loo_geometry","support_permutation_geometry"
            )}
            loo_active=0
            perm_active=0
            perm_pvals=[]
            perm_attempted=0
            for i in range(n_trials):
                ep,p,g,true_query=fake_trial(
                    rng,k,scenario=="geometry_signal")
                old=compare_krr_geometry_with_support_validation(ep,p,g,True)
                new=compare_krr_with_support_permutation_guard(
                    ep,p,g,True,permutation_settings=settings)
                if any(x.perceiver_krr_pKd!=y.perceiver_krr_pKd
                       for x,y in zip(old,new)):
                    raise AssertionError("Strong original Perceiver KRR baseline changed")
                loo_active+=int(old[0].gate>0)
                perm_active+=int(new[0].permutation_gate_passed)
                if new[0].null_pvalue is not None:
                    perm_attempted+=1
                    perm_pvals.append(new[0].null_pvalue)
                for name,preds in (
                    ("perceiver_krr",[r.perceiver_krr_pKd for r in new]),
                    ("loo_geometry",[r.geometry_krr_pKd for r in old]),
                    ("support_permutation_geometry",[
                        r.permutation_guarded_pKd for r in new]),
                ):
                    sq[name].extend((float(y)-float(x))**2
                                    for x,y in zip(preds,true_query))
            rmse={arm:round(float(np.sqrt(np.mean(val))),6)
                  for arm,val in sq.items()}
            results[scenario+"_K"+str(k)]={
                "fabricated_trials":n_trials,
                "support_count":k,
                "fictional_query_count_per_trial":len(SMILES)-k,
                "old_LOO_gate_active_trials":loo_active,
                "permutation_guard_applied_trials":perm_active,
                "permutation_nulls_computed_trials":perm_attempted,
                "mean_permutation_pvalue_when_computed":(
                    round(float(np.mean(perm_pvals)),6)
                    if perm_pvals else None),
                "fabricated_RMSE":rmse,
                "note":"All labels and geometry effects are deliberately fabricated.",
            }
    return {
        "status":"FABRICATED_SUPPORT_PERMUTATION_RELIABILITY_ONLY_NOT_CPI_EVIDENCE",
        "source":"Only local synthetic formula-based test fixtures",
        "seed":seed,"trial_count_per_case":n_trials,
        "n_support_null_permutations":settings.n_null_permutations,
        "support_null_max_pvalue":settings.max_pvalue,
        "test_split_used":False,
        "real_affinity_labels_used":False,
        "neural_training_performed":False,
        "results":results,
        "limitations":[
            "Permutation test assumes exchangeability of support geometry disagreement conditional on the chemical baseline; this may be false for real protein/chemical groups.",
            "The add-one Monte Carlo p-value is a support-only screening heuristic, not proof of improvement on independently held-out queries.",
            "Support chemical panel is tiny and family independence has not been reviewed.",
            "Synthetic trials contain the deliberately generated signal and cannot establish empirical PerceiverCPI superiority."
        ],
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite previous stress report")
    result=synthetic_permutation_stress()
    with OUTPUT.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
