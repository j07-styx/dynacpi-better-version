"""Nested training-only exploratory protein-conditioned corrections on double-cold DAVIS.

Predeclared additive ESM/sequence kernel with chemical prior; alpha=0 is
an honest allowed outcome. Replays EXISTING outer folds but learns all
hyperparameters on inner folds within each outer TRAIN partition only.

THE OUTER FOLDS HAVE ALREADY BEEN INSPECTED IN PREVIOUS EXPERIMENTS:
this is exploratory comparative engineering, not confirmatory generalization.
No use of 8 prior DAVIS DEV targets, source-unapproved BindingDB, or TEST data.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
SOURCE=Path(r"C:\Users\J0124\dynacpi_v4")
sys.path.insert(0,str(HERE))
from profile_model import AffinityProfile, tanimoto, normalize
from run_profile_study import partition_groups, metric, inner_select

BASE=HERE/"interaction_study_20261109"
OUTPUT=BASE/"conditional_kernel_20261109"
LAMBDAS=(0.1,1.0,10.0,100.0)
ALPHAS=(0.0,0.25,0.5,1.0)
PROTEIN_KERNELS=("esm_raw","esm_center","sequence_3mer")
CHEMICAL_KERNELS=("tanimoto","tanimoto_sq")
# CV optimization objective: double-cold RMSE (two fixed inner folds),
# tie breaks smaller alpha, larger ridge, simpler ESM.
TRAINING_SEED="v44_inner_doublecold_protein_conditional_fixed_v1"

def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,p)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def seq_embed(sequence):
    # 20 canonical residues, hashed contiguous trigrams, fixed vocabulary.
    alphabet="ACDEFGHIKLMNPQRSTVWY"
    index={a:i for i,a in enumerate(alphabet)}
    x=np.zeros((len(alphabet)**3,),dtype=np.float64)
    for i in range(len(sequence)-2):
        a,b,c=sequence[i:i+3]
        if a in index and b in index and c in index:
            x[index[a]*400+index[b]*20+index[c]]+=1.0
    return x

def p_kernel_fit(train,query,kind):
    train=np.asarray(train,dtype=np.float64)
    query=np.asarray(query,dtype=np.float64)
    if kind=="esm_center":
        mean=train.mean(0)
        tr=normalize(train-mean)
        qu=normalize(query-mean)
    else:
        tr=normalize(train)
        qu=normalize(query)
    dot_tr=np.clip(tr@tr.T,-1,1)
    dot_qu=np.clip(qu@tr.T,-1,1)
    # Constant RBF scale fixed via TRAIN-only pair distances.
    dist=np.maximum(0,2-2*dot_tr)
    positive=dist[np.triu_indices(len(tr),1)]
    positive=positive[positive>1e-10]
    scale=float(np.median(positive)) if len(positive) else 1.0
    kt=np.exp(-dist/max(scale,1e-10))
    kq=np.exp(-np.maximum(0,2-2*dot_qu)/max(scale,1e-10))
    # Center K on training proteins (protein residual is zero-mean).
    cm=kt.mean(0,keepdims=True)
    gm=float(kt.mean())
    kt=kt-kt.mean(1,keepdims=True)-cm+gm
    kq=kq-kq.mean(1,keepdims=True)-cm+gm
    return (kt+kt.T)/2,kq

def fit_predict(train_protein, query_protein, train_bit, query_bit,
                y_train, protein_kind, chemical_kind, lam, alpha,
                chemical_regularization):
    # Chemical-only GP uses TRAIN proteins and TRAIN compounds exclusively.
    chem=AffinityProfile(chemical_ridge=chemical_regularization,
                         residual_ridge=10.0,mode="chemical").fit(train_protein["esm_raw"],
                                                                  train_bit,y_train)
    prior=chem.chemical(query_bit)
    if alpha==0.0:
        return np.broadcast_to(prior,(len(query_protein["esm_raw"]),len(query_bit))).copy(),prior
    pk=protein_kind
    ptrain,pquery=p_kernel_fit(train_protein[pk],query_protein[pk],pk)
    ck=tanimoto(train_bit,train_bit)
    cq=tanimoto(query_bit,train_bit)
    if chemical_kind=="tanimoto_sq":
        ck=ck**2
        cq=cq**2
    # Residual relative to mean observed affinity for each TRAIN compound.
    residual=y_train-y_train.mean(0,keepdims=True)
    eigp,up=np.linalg.eigh(ptrain)
    eigc,uc=np.linalg.eigh(ck)
    eigp=np.clip(eigp,0,None)
    eigc=np.clip(eigc,0,None)
    coeff=(up.T@residual@uc)/(eigp[:,None]*eigc[None,:]+lam)
    a=up@coeff@uc.T
    effect=pquery@a@cq.T
    return np.broadcast_to(prior,(len(query_protein["esm_raw"]),len(query_bit)))+alpha*effect,prior

def score_block(pred,truth):
    return metric(pred,truth)

def main():
    if (OUTPUT/"RESULTS.json").exists():
        raise RuntimeError("Refuse overwrite previous experiment.")
    study=json.loads((BASE/"study_results.json").read_text(encoding="utf-8"))
    if study["status"]!="HISTORICAL_TRAIN_ONLY_NESTED_CROSS_VALIDATION_NOT_PROSPECTIVE_TEST":
        raise RuntimeError("Invalid prior protocol")
    train=[r for r in json.loads((SOURCE/"data/engineering_pilot/rows.json").read_text(encoding="utf-8")) if r["split"]=="train"]
    pids=study["protein_hashes"];cids=study["compound_hashes"]
    if len(pids)!=24 or len(cids)!=54 or len(train)!=1296:
        raise RuntimeError("Engineering DAVIS panel changed")
    ymap={(r["sequence"],r["compound"]):float(r["y"]) for r in train}
    Y=np.array([[ymap[p,c] for c in cids] for p in pids],dtype=np.float64)
    feats=np.load(SOURCE/"data/staging/morgan_radius2_2048_chiral.npz")
    stored={cid:bits for cid,bits in zip(feats["ids"].astype(str),feats["bits"])}
    B=np.stack([stored[c] for c in cids]).astype(np.float64)
    sequences=json.loads((SOURCE/"data/staging/sequences.json").read_text(encoding="utf-8"))
    esm=[];triples=[]
    for pid in pids:
        obj=torch.load(SOURCE/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(pid+".pt"),
                       map_location="cpu",weights_only=True)
        assert obj["sequence_hash"]==pid
        esm.append(obj["embeddings"].float().mean(0).numpy())
        triples.append(seq_embed(sequences[pid]))
    E=np.stack(esm);T=np.stack(triples)
    features={"esm_raw":E,"esm_center":E,"sequence_3mer":T}
    protein_groups=[[i] for i in range(24)]
    chemical_groups=study["chemical_groups"]
    outer_protein=study["outer_protein_folds"]
    outer_chem=study["outer_chemical_folds"]
    assert len(outer_protein)==len(outer_chem)==3
    output={"status":"EXPLORATORY_REUSED_OUTER_SPLIT_NOT_INDEPENDENT_TEST",
            "interpretation":"Outer folds were evaluated in earlier work, preventing a confirmatory claim.",
            "model":"train-only chemical prior plus centered protein-chemical kernel residual",
            "candidate_grid":{"ridge":LAMBDAS,"alpha":ALPHAS,"protein":PROTEIN_KERNELS,
                              "chemical":CHEMICAL_KERNELS},
            "inner_selection":"2 paired inner folds, minimize DOUBLE_COLD target-macro RMSE",
            "training_only_rows_sha256":sha(SOURCE/"data/engineering_pilot/rows.json"),
            "legacy_results_sha256":sha(BASE/"study_results.json"),
            "protein_id_disjoint":True,"protein_family_disjoint":False,
            "compound_scaffold_similarity_threshold":0.5,
            "outer_fold_ids_preserved":True,"folds":[]}
    dump(OUTPUT/"PROTOCOL.json",output)
    # Always process model selection from scratch on each outer training rectangle.
    for fold in range(3):
        pi=sorted(set(range(24))-set(outer_protein[fold]))
        ci=sorted(set(range(54))-set(outer_chem[fold]))
        po=outer_protein[fold];co=outer_chem[fold]
        assert len(pi)==16 and len(ci)==36
        pi_local=partition_groups([[i] for i in range(len(pi))],2,
                                  TRAINING_SEED+"|protein|"+str(fold))
        groups=[[ci.index(i) for i in g if i in ci] for g in chemical_groups]
        groups=[g for g in groups if g]
        ci_local=partition_groups(groups,2,TRAINING_SEED+"|chemical|"+str(fold))
        # Chemical regularization preselected on the same inner folds for chem-only:
        # first calculate chemical-only score for each candidate WITHOUT using outer labels.
        trial=[]
        for rchem in (0.0001,0.01,0.1,1.0,10.0):
            vals=[]
            for inner in range(2):
                pfit=[pi[j] for j in range(len(pi)) if j not in set(pi_local[inner])]
                cfit=[ci[j] for j in range(len(ci)) if j not in set(ci_local[inner])]
                ptest=[pi[j] for j in pi_local[inner]]
                ctest=[ci[j] for j in ci_local[inner]]
                pp={"esm_raw":E[pfit],"esm_center":E[pfit],"sequence_3mer":T[pfit]}
                qp={"esm_raw":E[ptest],"esm_center":E[ptest],"sequence_3mer":T[ptest]}
                pred,_=fit_predict(pp,qp,B[cfit],B[ctest],Y[np.ix_(pfit,cfit)],
                                    "esm_raw","tanimoto",10.,0.,rchem)
                vals.append(score_block(pred,Y[np.ix_(ptest,ctest)])["macro_rmse"])
            trial.append({"rchem":rchem,"inner_doublecold_rmse":float(np.mean(vals))})
        selected_chemical=min(trial,key=lambda z:(z["inner_doublecold_rmse"],z["rchem"]))
        chemreg=selected_chemical["rchem"]
        # Fit residual candidates on the two inner training rectangles. Train caches are
        # not shared with any outer validation protein or compound.
        candidate_scores=[]
        for pk in PROTEIN_KERNELS:
            for ck in CHEMICAL_KERNELS:
                for ridge in LAMBDAS:
                    for alpha in ALPHAS:
                        vals=[]
                        for inner in range(2):
                            pfit=[pi[j] for j in range(len(pi)) if j not in set(pi_local[inner])]
                            cfit=[ci[j] for j in range(len(ci)) if j not in set(ci_local[inner])]
                            ptest=[pi[j] for j in pi_local[inner]]
                            ctest=[ci[j] for j in ci_local[inner]]
                            pp={"esm_raw":E[pfit],"esm_center":E[pfit],"sequence_3mer":T[pfit]}
                            qp={"esm_raw":E[ptest],"esm_center":E[ptest],"sequence_3mer":T[ptest]}
                            pred,_=fit_predict(pp,qp,B[cfit],B[ctest],Y[np.ix_(pfit,cfit)],
                                               pk,ck,ridge,alpha,chemreg)
                            vals.append(score_block(pred,Y[np.ix_(ptest,ctest)])["macro_rmse"])
                        candidate_scores.append({"protein":pk,"chemical":ck,
                             "ridge":ridge,"alpha":alpha,
                             "inner_doublecold_rmse":float(np.mean(vals))})
        # Tiebreak smaller alpha then larger ridge then deterministic iteration order.
        best=min(candidate_scores,key=lambda z:(z["inner_doublecold_rmse"],z["alpha"],-z["ridge"]))
        pp={k:v[pi] for k,v in features.items()}
        def evaluate(pidx,cidx):
            qp={k:v[pidx] for k,v in features.items()}
            candidate,prior=fit_predict(pp,qp,B[ci],B[cidx],Y[np.ix_(pi,ci)],
                   best["protein"],best["chemical"],best["ridge"],best["alpha"],chemreg)
            # Swap the protein identity as a direct counterfactual (not label tuning).
            rotated_idx=list(reversed(pidx))
            shuffled={k:v[rotated_idx] for k,v in features.items()}
            shuffled_predictions,_=fit_predict(pp,shuffled,B[ci],B[cidx],Y[np.ix_(pi,ci)],
                   best["protein"],best["chemical"],best["ridge"],best["alpha"],chemreg)
            chem_only=np.broadcast_to(prior,candidate.shape)
            yy=Y[np.ix_(pidx,cidx)]
            return {"new":score_block(candidate,yy),
                    "chemical_only_same_training_ridge":score_block(chem_only,yy),
                    "protein_id_swapped_counterfactual":score_block(shuffled_predictions,yy),
                    "mean_abs_protein_effect":float(np.mean(np.abs(candidate-chem_only))),
                    "mean_abs_protein_swap_effect":float(np.mean(np.abs(candidate-shuffled_predictions))),
                    "prediction_variance_within_protein":float(np.mean(np.var(candidate,axis=1)))}
        rec={"fold":fold,"train_proteins":16,"train_compounds":36,
            "heldout_proteins":len(po),"heldout_compounds":len(co),
            "chemical_regularization":chemreg,
            "chemical_inner_choices":trial,
            "selected_candidate":best,
            "candidate_count":len(candidate_scores),
            "inner_best_candidates":sorted(candidate_scores,
               key=lambda z:z["inner_doublecold_rmse"])[:5],
            "regimes":{
                "cold_protein":evaluate(po,ci),
                "cold_compound":evaluate(pi,co),
                "double_cold":evaluate(po,co)}}
        output["folds"].append(rec)
        dump(OUTPUT/"PROGRESS.json",output)
        print(json.dumps({"fold":fold,"best":best,
             "chemical_ridge":chemreg,
             "double_cold_new_rmse":rec["regimes"]["double_cold"]["new"]["macro_rmse"],
             "double_cold_chemical_rmse":rec["regimes"]["double_cold"]["chemical_only_same_training_ridge"]["macro_rmse"],
             "protein_effect_mae":rec["regimes"]["double_cold"]["mean_abs_protein_effect"]}),flush=True)
    output["aggregates"]={}
    for regime in ("cold_protein","cold_compound","double_cold"):
        output["aggregates"][regime]={}
        for key in ("new","chemical_only_same_training_ridge","protein_id_swapped_counterfactual"):
            output["aggregates"][regime][key]=float(np.mean([f["regimes"][regime][key]["macro_rmse"] for f in output["folds"]]))
        output["aggregates"][regime]["mean_abs_protein_effect"]=float(np.mean(
            [f["regimes"][regime]["mean_abs_protein_effect"] for f in output["folds"]]))
    # Existing frozen baseline is shown separately, not used for tuning:
    output["historically_seen_old_doublecold_chemical_rmse"]=float(np.mean(
         [f["models"]["chemical_profile"]["double_cold"]["macro_rmse"] for f in study["folds"]]))
    dump(OUTPUT/"RESULTS.json",output)
    print(json.dumps({"status":"DONE","doublecold":output["aggregates"]["double_cold"],
       "previous_baseline":output["historically_seen_old_doublecold_chemical_rmse"]}),flush=True)

if __name__=="__main__":
    main()
