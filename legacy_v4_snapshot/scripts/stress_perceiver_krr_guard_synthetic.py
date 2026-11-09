"""Synthetic-only KRR geometry support-gate stress battery.

Never reads BindingDB, model checkpoints, curated source manifests or test
outcomes. Fabricates all labels/predictions from elementary random functions.
The output is NOT a drug-target accuracy benchmark or model win.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from dynacpi.episode_data import PredictionEpisode,SupportObservation,QueryCompound
from dynacpi.perceiver_krr_geometry_guard import compare_krr_geometry_with_support_validation
from dynacpi.perceiver_geometry_negative_control import shuffle_compound_disagreement

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"research/perceiver_krr_guard_fabricated_noise_stress_20261109.json"
SMILES=("CCO","CCCO","CCCCO","CC(C)O","CCN","CCCCCO","CCCN","CC(C)CO")
D=np.array([-.5,-.35,-.1,.3,.5,.65,-.3,.45],dtype=float)
N_TRIALS=128
SEED=2601


def chemistry():
    canon=tuple(Chem.MolToSmiles(Chem.MolFromSmiles(s),
                                 canonical=True,isomericSmiles=True) for s in SMILES)
    cid=tuple(sha256(s.encode("utf-8")).hexdigest() for s in canon)
    return cid,canon


def fake_trial(rng, k, geometry_coupled):
    cid,canon=chemistry()
    p={j:6.0 for j in cid}
    g={j:6.0+float(d) for j,d in zip(cid,D)}
    beta=0.8 if geometry_coupled else 0.0
    noise=rng.normal(0,.15,len(cid))
    fictional_y=6.0+.25+beta*D+noise
    support=tuple(SupportObservation(cid[i],canon[i],float(fictional_y[i]),
                                     "invented-assay") for i in range(k))
    queries=tuple(QueryCompound(cid[i],canon[i]) for i in range(k,len(cid)))
    episode=PredictionEpisode(
        task_id="invented-example",sequence_sha256="fake",
        protein_component_id="fake",assay_key="invented-assay",
        support=support,queries=queries,status="PRE_SPLIT_UNAPPROVED")
    return episode,p,g,fictional_y[k:]


def synthetic_stress(n_trials=N_TRIALS,seed=SEED):
    if not isinstance(n_trials,int) or n_trials<16:
        raise ValueError("Need >=16 synthetic trials")
    if not isinstance(seed,int) or seed<0:
        raise ValueError("Invalid synthetic seed")
    results={}
    for scenario in ("geometry_signal","geometry_null"):
        for k in (5,6):
            rng=np.random.default_rng(seed+int(k)*1000+(0 if scenario=="geometry_signal" else 1))
            errors={name:[] for name in ("perceiver_krr","krr_plus_geometry","krr_plus_shuffled_geometry")}
            active=0
            shuffle_active=0
            gains=[]
            for trial in range(n_trials):
                episode,p,g,fictional_query_y=fake_trial(rng,k,scenario=="geometry_signal")
                real=compare_krr_geometry_with_support_validation(episode,p,g,True)
                shuffled_g=shuffle_compound_disagreement(
                    episode,p,g,shuffle_seed=seed+trial)
                shuffled=compare_krr_geometry_with_support_validation(episode,p,shuffled_g,True)
                assert all(a.perceiver_krr_pKd==b.perceiver_krr_pKd for a,b in zip(real,shuffled))
                active+=int(real[0].gate>0)
                shuffle_active+=int(shuffled[0].gate>0)
                gains.append(real[0].loo_relative_gain)
                for name,preds in (
                    ("perceiver_krr",[a.perceiver_krr_pKd for a in real]),
                    ("krr_plus_geometry",[a.geometry_krr_pKd for a in real]),
                    ("krr_plus_shuffled_geometry",[a.geometry_krr_pKd for a in shuffled]),
                ):
                    errors[name].extend((float(y)-float(v))**2
                                        for y,v in zip(fictional_query_y,preds))
            key=scenario+"_K"+str(k)
            rmse={name:round(float(np.sqrt(np.mean(err))),6)
                  for name,err in errors.items()}
            results[key]={
                "fabricated_trials":n_trials,
                "support_size":k,
                "fabricated_queries_per_trial":len(SMILES)-k,
                "geometry_gate_active_trials":active,
                "shuffled_geometry_gate_active_trials":shuffle_active,
                "mean_LOO_relative_gain":round(float(np.mean(gains)),6),
                "fabricated_RMSE":rmse,
                "geometry_minus_krr_RMSE":round(
                    rmse["krr_plus_geometry"]-rmse["perceiver_krr"],6),
                "note":"Generated labels are built from a known signal or no signal, not from observed drug binding."
            }
    return {
        "status":"FABRICATED_KRR_SUPPORT_GATE_STRESS_NOT_SCIENTIFIC_EVIDENCE",
        "seed":seed,
        "trial_count_per_case":n_trials,
        "geometry_signal_beta":0.8,
        "geometry_null_beta":0.0,
        "random_noise_sd":0.15,
        "assay_shift":0.25,
        "approval_or_training_performed":False,
        "results":results,
        "warning":"Mathematical synthetic noise stress only. Hyperparameters are prototype defaults, not optimized or validated against actual CPI. Do not use these results as an experimental improvement claim.",
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite synthetic test result")
    result=synthetic_stress()
    with OUTPUT.open("x",encoding="utf-8") as out:
        json.dump(result,out,indent=2)
        out.write("\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
