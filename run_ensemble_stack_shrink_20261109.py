"""Nested ensemble / stacking / chemical-and-protein similarity shrink on DAVIS TRAIN.

This code runs independent cross-fitted TRAIN-only inner neural ensembles for
choosing a blend/gate, then refits each outer seed on all outer TRAIN data.
The old outer folds have ALREADY BEEN SEEN in earlier v4.3/v4.4 work, so results
are exploratory only. No original DAVIS DEV, BindingDB, or reserved TEST labels.

Target-macro RMSE is equal-weighted over protein identities.
No outer labels used to select blend, gate or training epoch.
Checkpoint snapshots are NOT overwritten. Only aggregate JSON is publishable.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
OLD=Path(r"C:\Users\J0124\dynacpi_v4")
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(OLD/"scripts"))
from gpu_lease import gpu_lease
from repaired_model import RepairedSequenceFoundation
from profile_model import AffinityProfile, normalize, tanimoto
from run_profile_study import inner_select, partition_groups, metric, forward

PREV=HERE/"interaction_study_20261109"
OUT=HERE/"interaction_study_20261109"/"ensemble_stack_shrink_20261109"
SEEDS=(2931,2932,2933)
EPOCHS=5
LR=3e-4
WD=1e-4
WEIGHTS=(0.,.25,.5,.75,1.)
GATES=("global","chemical_tanimoto","chemical_times_protein_cosine")
CHEM_THRESHOLD=.5
PROTO="20261109_nest2_3seed_5epoch_replayed_outer_with_zero_anchor_v1"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def make_groups(groups,global_indices):
    m={old:i for i,old in enumerate(global_indices)}
    g=[[m[j] for j in group if j in m] for group in groups]
    return [x for x in g if x]


def prior_fit(pi,ci,pfeat,bits,y,groups):
    """Select prior on subtraining data ONLY (never its inner validation)."""
    ch,_=inner_select(pfeat[pi],bits[ci],y[np.ix_(pi,ci)],make_groups(groups,ci))
    ridge=ch["chemical"]["chemical_ridge"]
    model=AffinityProfile(ridge,10.0,mode="chemical").fit(pfeat[pi],bits[ci],y[np.ix_(pi,ci)])
    return model.chemical(bits),ridge


def train_anchored_predict(pi,ci,queries_p,queries_c,sm,bits,y,emb,prior):
    """Reproduce original run_profile_study anchored neural configuration."""
    scale=float(y[np.ix_(pi,ci)].std())
    if scale<=0:raise ValueError("Invalid train-only scale")
    tensor=torch.tensor(bits,dtype=torch.float32,device="cuda")
    fit_sm=[sm[i] for i in ci]
    # Match run_profile_study exactly: subtract/scalings happen in float64
    # BEFORE one final float32 cast. Precasting labels changes Adam updates.
    outputs=[]
    history=[]
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        m=RepairedSequenceFoundation().cuda()
        nn.init.zeros_(m.pair_head[-1].weight)
        nn.init.zeros_(m.pair_head[-1].bias)
        optimizer=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
        losses=[]
        for epoch in range(1,EPOCHS+1):
            m.train()
            current=[]
            indices=np.random.default_rng(seed*100+epoch).permutation(len(pi))
            for index in indices:
                p=pi[index]
                optimizer.zero_grad(set_to_none=True)
                c=m.compound(fit_sm,tensor[ci])
                pred=forward(m,c,emb[p])
                label=torch.tensor((y[p,ci]-prior[ci])/scale,device="cuda",dtype=torch.float32)
                loss=(pred-label).square().mean()
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite neural training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step()
                current.append(float(loss.detach()))
            losses.append(float(np.mean(current)))
        m.eval()
        with torch.inference_mode():
            compounds=m.compound(sm,tensor)
            vals=np.stack([(forward(m,compounds,emb[p]).float().cpu().numpy()*scale
                   +prior).astype(np.float64) for p in queries_p])
        outputs.append(vals[:,queries_c])
        history.append({"seed":seed,"train_standardized_mse":losses})
        del m,optimizer
        torch.cuda.empty_cache()
    return np.stack(outputs),history


def gate(pi,ci,qp,qc,pfeat,bits,mode):
    if mode=="global":return np.ones((len(qp),len(qc)),dtype=np.float64)
    sims=tanimoto(bits[qc],bits[ci])
    nearest=sims.max(axis=1)
    chemical=np.clip(nearest,0,1)
    if mode=="chemical_tanimoto":return np.broadcast_to(chemical[None,:],(len(qp),len(qc))).copy()
    if mode=="chemical_times_protein_cosine":
        pcos=normalize(pfeat[qp])@normalize(pfeat[pi]).T
        psim=np.clip(pcos.max(axis=1),0,1)
        return psim[:,None]*chemical[None,:]
    raise ValueError("Unknown gate "+str(mode))


def predict_combo(base,ens,g,weight):
    return base+weight*g*(ens-base)


def pick_inner(inner_cases):
    score_grid=[]
    for mode in GATES:
        for weight in WEIGHTS:
            vals=[metric(predict_combo(z["prior"],z["ensemble"],z["gates"][mode],weight),z["y"])["macro_rmse"]
                  for z in inner_cases]
            score_grid.append({"gate":mode,"weight":weight,
               "inner_mean_target_macro_rmse":float(np.mean(vals)),
               "inner_fold_target_macro_rmse":[float(v) for v in vals]})
    best=min(score_grid,key=lambda z:(z["inner_mean_target_macro_rmse"],z["weight"],GATES.index(z["gate"])))
    return best,score_grid


def paired_bootstrap(diff_by_target,fold_by_target,n=10000):
    dif=np.asarray(diff_by_target,dtype=np.float64)
    folds=np.asarray(fold_by_target,dtype=np.int64)
    assert len(dif)==24 and set(folds)=={0,1,2}
    rng=np.random.default_rng(9937)
    uniform=rng.integers(0,len(dif),size=(n,len(dif)))
    means=dif[uniform].mean(1)
    # Fold resampling captures additional partition variability, but with
    # only 3 folds is extremely unstable (descriptive, not inferential).
    fmean=np.array([dif[folds==f].mean() for f in range(3)])
    resampled=fmean[rng.integers(0,3,size=(n,3))].mean(1)
    return {"paired_mean_delta":float(dif.mean()),
       "descriptive_protein_bootstrap_95pct":[float(x) for x in np.percentile(means,[2.5,97.5])],
       "descriptive_outer_fold_bootstrap_95pct":[float(x) for x in np.percentile(resampled,[2.5,97.5])],
       "n_distinct_proteins":len(dif),"n_reused_outer_folds":3,
       "not_confirmatory":True}


def main():
    start=time.monotonic()
    torch.set_num_threads(2)
    if (OUT/"RESULTS.json").exists():
        raise RuntimeError("Refusing to overwrite completed experiment")
    # Resume only after verifying the same fixed protocol and original folds;
    # retain earlier incomplete/failed logs in their original directories.
    continuing=(OUT/"PROGRESS.json").exists()
    old=json.loads((PREV/"study_results.json").read_text(encoding="utf-8"))
    if old["status"]!="HISTORICAL_TRAIN_ONLY_NESTED_CROSS_VALIDATION_NOT_PROSPECTIVE_TEST":
        raise RuntimeError("Unexpected prior protocol")
    train=[r for r in json.loads((OLD/"data/engineering_pilot/rows.json").read_text(encoding="utf-8"))
           if r["split"]=="train"]
    pids=old["protein_hashes"]
    cids=old["compound_hashes"]
    assert len(pids)==24 and len(cids)==54 and len(train)==1296
    obs={(r["sequence"],r["compound"]):float(r["y"]) for r in train}
    y=np.array([[obs[p,c] for c in cids] for p in pids],dtype=np.float64)
    smiles=json.loads((OLD/"data/staging/compounds.json").read_text(encoding="utf-8"))
    sm=[smiles[c] for c in cids]
    fpfile=OLD/"data/staging/morgan_radius2_2048_chiral.npz"
    z=np.load(fpfile)
    fps={cid:vec for cid,vec in zip(z["ids"].astype(str),z["bits"])}
    bits=np.stack([fps[c] for c in cids]).astype(np.float64)
    emb=[]
    for pid in pids:
        obj=torch.load(OLD/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(pid+".pt"),
                       map_location="cpu",weights_only=True)
        if obj["sequence_hash"]!=pid:raise RuntimeError("ESM identity mismatch")
        emb.append(obj["embeddings"].float().cuda())
    pfeat=np.stack([e.mean(0).cpu().numpy() for e in emb])
    groups=old["chemical_groups"]
    outcome={
       "status":"RUNNING_EXPLORATORY_REUSED_DAVIS_TRAIN_OUTER_FOLDS",
       "protocol":PROTO,"train_proteins":24,"train_compounds":54,"train_pairs":1296,
       "same_prior_selection_as_original":True,"seeds":list(SEEDS),
       "epochs":EPOCHS,"learning_rate":LR,"weight_decay":WD,
       "candidate_weights":list(WEIGHTS),"candidate_gates":list(GATES),
       "inner_model_selection":"retrain THREE anchored seeds on each of two paired inner train-only rectangles; select target-macro mean RMSE for double-cold; no outer labels used",
       "ensemble":"unweighted mean of per-pair predictions from seeds 2931 2932 2933",
       "outer_splits_sha256":sha(PREV/"study_results.json"),
       "training_data_sha256":sha(OLD/"data/engineering_pilot/rows.json"),
       "fingerprints_sha256":sha(fpfile),
       "source_sha256":sha(HERE/"run_ensemble_stack_shrink_20261109.py"),
       "limitations":[
         "These same outer folds were previously scored and informed this experiment; exploratory only.",
         "Protein identities but not families are independent in each fold.",
         "Compound scaffold-neighbor groups held out via preexisting Tanimoto 0.5 threshold.",
         "No original DAVIS DEV labels or unapproved BindingDB/test data are used.",
         "No PerceiverCPI same-protocol baseline exists; this cannot claim superiority.",
         "Mean protein-fingerprint similarities are not calibrated confidence.",
         "Three seeds are correlated and 24 proteins give broad uncertainty."
       ],
       "folds":[]}
    if continuing:
        stored=json.loads((OUT/"PROGRESS.json").read_text(encoding="utf-8"))
        if stored.get("protocol")!=PROTO or stored.get("training_data_sha256")!=outcome["training_data_sha256"]:
            raise RuntimeError("Cannot resume different experiment or modified dataset")
        if len(stored["folds"]) not in (1,2):
            raise RuntimeError("Unexpected incomplete fold count")
        outcome["folds"]=stored["folds"]
        outcome["replay_audit"]="Historical v4.3 CUDA replays can drift about 0.0015 RMSE for seed 2933 on outer fold 2; 0.003 tolerance recorded, not model selection."
        outcome["source_sha256"]=sha(HERE/"run_ensemble_stack_shrink_20261109.py")
    else:
        dump(OUT/"PROTOCOL.json",outcome)
    all_per_protein=[]
    for prior_fold in outcome["folds"]:
        for item in prior_fold["per_protein_paired_delta"]:
            all_per_protein.append({"fold":prior_fold["fold"],
              "ens_delta":item["ensemble_minus_baseline"],
              "stack_delta":item["stack_minus_baseline"]})
    for fold in range(len(outcome["folds"]),3):
        pi=sorted(set(range(24))-set(old["outer_protein_folds"][fold]))
        ci=sorted(set(range(54))-set(old["outer_chemical_folds"][fold]))
        po=old["outer_protein_folds"][fold]
        co=old["outer_chemical_folds"][fold]
        assert len(pi)==16 and len(ci)==36 and len(po)==8 and len(co)==18
        # Identical original TRAIN-only chemistry-prior selection on outer fit.
        prior,ridge=prior_fit(pi,ci,pfeat,bits,y,groups)
        base=np.broadcast_to(prior[co],(len(po),len(co))).copy()
        target=y[np.ix_(po,co)]
        previous=old["folds"][fold]["models"]["chemical_profile"]["double_cold"]["macro_rmse"]
        observed=metric(base,target)["macro_rmse"]
        assert abs(previous-observed)<1e-7,(fold,previous,observed)
        outer_preds,outer_history=train_anchored_predict(pi,ci,po,co,sm,bits,y,emb,prior)
        assert outer_preds.shape==(3,8,18)
        replay=[]
        for index,seed in enumerate(SEEDS):
            was=old["folds"][fold]["models"]["anchored_neural_seed"+str(seed)]["double_cold"]["macro_rmse"]
            got=metric(outer_preds[index],target)["macro_rmse"]
            replay.append({"seed":seed,"previous_dev_cv_rmse":float(was),"replay_rmse":float(got),"difference":float(got-was)})
            if abs(was-got)>0.003:
                raise RuntimeError(f"Failed bounded CUDA replay consistency f={fold} seed={seed} old={was} new={got}")
        ens=outer_preds.mean(0)

        # Nested double-cold selection. Inner holdout labels exist only during
        # selecting the blend for this OUTER fit (never used to fit outer seeds).
        pigroups=partition_groups([[i] for i in range(len(pi))],2,
                  "ensemble_stack_inner_protein_"+str(fold))
        cgroups=partition_groups(make_groups(groups,ci),2,
                  "ensemble_stack_inner_chemical_"+str(fold))
        cases=[]
        inner_details=[]
        for k in range(2):
            fit_pi=[pi[j] for j in range(len(pi)) if j not in set(pigroups[k])]
            fit_ci=[ci[j] for j in range(len(ci)) if j not in set(cgroups[k])]
            val_pi=[pi[j] for j in pigroups[k]]
            val_ci=[ci[j] for j in cgroups[k]]
            assert not set(fit_pi)&set(val_pi)
            assert not set(fit_ci)&set(val_ci)
            internal_prior,internal_ridge=prior_fit(fit_pi,fit_ci,pfeat,bits,y,groups)
            net,history=train_anchored_predict(fit_pi,fit_ci,val_pi,val_ci,sm,bits,y,emb,internal_prior)
            base_in=np.broadcast_to(internal_prior[val_ci],(len(val_pi),len(val_ci))).copy()
            ensemble_in=net.mean(0)
            tr=y[np.ix_(val_pi,val_ci)]
            gates={g:gate(fit_pi,fit_ci,val_pi,val_ci,pfeat,bits,g) for g in GATES}
            cases.append({"prior":base_in,"ensemble":ensemble_in,"y":tr,"gates":gates})
            inner_details.append({"inner_fold":k,"train_proteins":len(fit_pi),
                   "train_compounds":len(fit_ci),"validation_proteins":len(val_pi),
                   "validation_compounds":len(val_ci),"chemical_ridge":internal_ridge,
                   "baseline_rmse":metric(base_in,tr)["macro_rmse"],
                   "ensemble_rmse":metric(ensemble_in,tr)["macro_rmse"]})
        selection,grid=pick_inner(cases)
        gouter=gate(pi,ci,po,co,pfeat,bits,selection["gate"])
        stack=predict_combo(base,ens,gouter,selection["weight"])
        check_zero=predict_combo(base,ens,np.ones_like(base),0.0)
        assert np.array_equal(check_zero,base)
        m={"chemical_only":metric(base,target),
           "anchored_seed_mean_rmse":float(np.mean([metric(q,target)["macro_rmse"] for q in outer_preds])),
           "anchored_prediction_ensemble":metric(ens,target),
           "inner_selected_stack":metric(stack,target),
           "fixed_half_stack":metric(predict_combo(base,ens,np.ones_like(base),.5),target),
           "fixed_half_chemical_shrink":metric(predict_combo(base,ens,
                                     gate(pi,ci,po,co,pfeat,bits,"chemical_tanimoto"),.5),target)}
        # Non-destructive local prediction cache; NEVER push pair-level arrays.
        # Includes measured truth; do not allow into PUBLIC git.
        cache=OUT/("fold_"+str(fold)+"_predictions.npz")
        np.savez_compressed(cache,truth=target,chemical=base,seed_predictions=outer_preds,
                            ensemble=ens,selected_stack=stack)
        target_deltas=[]
        for j in range(len(po)):
            truth=target[j]
            baseline_rmse=float(np.sqrt(np.mean((base[j]-truth)**2)))
            ens_rmse=float(np.sqrt(np.mean((ens[j]-truth)**2)))
            stack_rmse=float(np.sqrt(np.mean((stack[j]-truth)**2)))
            target_deltas.append({"protein_index_in_prior_24":int(po[j]),
                "baseline_rmse":baseline_rmse,"ensemble_minus_baseline":ens_rmse-baseline_rmse,
                "stack_minus_baseline":stack_rmse-baseline_rmse})
            all_per_protein.append({"fold":fold,"ens_delta":ens_rmse-baseline_rmse,
                                    "stack_delta":stack_rmse-baseline_rmse})
        record={"fold":fold,"train_proteins":16,"train_compounds":36,
            "heldout_proteins":8,"heldout_compounds":18,
            "chemical_prior_ridge":ridge,"inner_selection":selection,
            "inner_validation_summaries":inner_details,
            "inner_grid":grid,"metrics":m,"replay_against_original":replay,
            "per_protein_paired_delta":target_deltas,
            "prediction_cache_local_only":cache.name,
            "seed_training_histories":outer_history}
        outcome["folds"].append(record)
        dump(OUT/"PROGRESS.json",outcome)
        print(json.dumps({"fold":fold,"status":"OUTER_FOLD_DONE",
            "prior":m["chemical_only"]["macro_rmse"],
            "seed_mean":m["anchored_seed_mean_rmse"],
            "ensemble":m["anchored_prediction_ensemble"]["macro_rmse"],
            "stack":m["inner_selected_stack"]["macro_rmse"],
            "inner_choice":selection}),flush=True)

    aggregate={}
    for key in ("chemical_only","anchored_prediction_ensemble","inner_selected_stack",
                "fixed_half_stack","fixed_half_chemical_shrink"):
        vals=[f["metrics"][key]["macro_rmse"] for f in outcome["folds"]]
        aggregate[key]={"target_macro_rmse":float(np.mean(vals)),
            "per_fold_macro_rmse":[float(v) for v in vals]}
    aggregate["anchored_seed_mean_rmse"]={
        "target_macro_rmse":float(np.mean([f["metrics"]["anchored_seed_mean_rmse"] for f in outcome["folds"]]))}
    for col in ("ens_delta","stack_delta"):
        aggregate["paired_"+col]=paired_bootstrap(
            [r[col] for r in all_per_protein],
            [r["fold"] for r in all_per_protein])
    aggregate["original_prior_old_mean"]=float(np.mean([
        f["models"]["chemical_profile"]["double_cold"]["macro_rmse"] for f in old["folds"]]))
    aggregate["original_anchored_mean"]=float(np.mean([
        f["models"]["anchored_neural_seed"+str(seed)]["double_cold"]["macro_rmse"]
        for f in old["folds"] for seed in SEEDS]))
    outcome["aggregates"]=aggregate
    outcome["status"]="COMPLETE_EXPLORATORY_NESTED_ENSEMBLE_STACK_SHRINK_NOT_INDEPENDENT_TEST"
    outcome["elapsed_seconds"]=round(time.monotonic()-start,2)
    dump(OUT/"RESULTS.json",outcome)
    print(json.dumps({"status":outcome["status"],"elapsed_seconds":outcome["elapsed_seconds"],
      "prior":aggregate["chemical_only"]["target_macro_rmse"],
      "seed_mean":aggregate["anchored_seed_mean_rmse"]["target_macro_rmse"],
      "ensemble":aggregate["anchored_prediction_ensemble"]["target_macro_rmse"],
      "stack":aggregate["inner_selected_stack"]["target_macro_rmse"],
      "stack_delta_ci":aggregate["paired_stack_delta"]}),flush=True)


if __name__=="__main__":
    with gpu_lease(OLD):
        main()
