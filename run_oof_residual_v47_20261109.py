"""v4.7 deterministic, scaffold-excluded cross-fitted chemical residual experiment.

Scientific status: EXPLORATORY; the same DAVIS TRAIN-only outer folds were
previously inspected by v4.3-v4.6. Never claim independent superiority.

Comparison: anchored neural residual labels against in-sample fitted prior
(control) versus a TRAIN-ONLY, scaffold/neighbor-group EXCLUDED 4-way
cross-fitted prior (treatment). Both use same fixed epoch 5, same three seeds,
same initialization and full-fit test prior. Weight selection uses 4 inner
protein/chemical grouped folds on the OUTER TRAIN rectangle only.
No similarity gates, no retrospective outer-fold model selection.
"""
from __future__ import annotations
import os
# Must be set BEFORE importing torch or initializing CUDA.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG",":4096:8")!=":4096:8":
    raise RuntimeError("Determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ.setdefault("PYTHONHASHSEED","0")
import copy
import hashlib
import json
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
sys.path.insert(0,str(HERE)) # avoid accidental same-named modules in OLD
import profile_model
import repaired_model
import run_profile_study
from profile_model import AffinityProfile, tanimoto
from repaired_model import RepairedSequenceFoundation
from run_profile_study import partition_groups, inner_select, metric, forward
from gpu_lease import gpu_lease

for module in (profile_model,repaired_model,run_profile_study):
    if Path(module.__file__).resolve().parent!=HERE.resolve():
        raise RuntimeError("Wrong module origin: "+str(module.__file__))

PREV=HERE/"interaction_study_20261109"
OUT=PREV/"oof_residual_4inner_deterministic_20261109"
SEEDS=(2931,2932,2933)
EPOCHS=5
WEIGHTS=(0.0,0.25,0.5,0.75,1.0)
ARMS=("insample","oof")
INNER_FOLDS=4
CROSS_FIT_FOLDS=4
PROTO="v47_nested4_4way_cluster_oof_prior_fixed5_deterministic_v1"

def configure_determinism():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError("This exact matched training requires a CUDA GPU")
    return {"torch_version":torch.__version__,
            "cuda_runtime":torch.version.cuda,
            "device":torch.cuda.get_device_name(0),
            "cublas_workspace_config":os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic":torch.backends.cudnn.deterministic,
            "cudnn_benchmark":torch.backends.cudnn.benchmark,
            "tf32_matmul":torch.backends.cuda.matmul.allow_tf32}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):h.update(buf)
    return h.hexdigest()

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(temp,path)

def local_groups(global_groups,cids):
    lookup={v:i for i,v in enumerate(cids)}
    ans=[[lookup[i] for i in group if i in lookup] for group in global_groups]
    return [sorted(g) for g in ans if g]

def fitted_prior(pi,ci,features,bits,y,groups):
    """All label fitting and ridge choice use *only* pi x ci."""
    grouped=local_groups(groups,ci)
    decisions,_=inner_select(features[pi],bits[ci],
                             y[np.ix_(pi,ci)],grouped)
    ridge=float(decisions["chemical"]["chemical_ridge"])
    m=AffinityProfile(ridge,10.0,mode="chemical")
    m.fit(features[pi],bits[ci],y[np.ix_(pi,ci)])
    return m.chemical(bits),ridge

def crossfit_prior(pi,ci,features,bits,y,groups,salt,prior_builder=None,with_audit=True):
    """LOCO-group OOF predictions for EVERY train compound, no self-target labels.

    Every compound/scaffold neighbor group gets a prediction from a chemical
    model with its whole group removed. Ridge tuning is also rerun on each
    reduced TRAIN rectangle, excluding all held-out labels.

    prior_builder is injectable solely for testing exclusion boundaries.
    """
    ci=list(ci)
    pieces=local_groups(groups,ci)
    if len(pieces)<CROSS_FIT_FOLDS:
        raise RuntimeError("Insufficient disjoint chemical groups for 4-way OOF")
    folds=partition_groups(pieces,CROSS_FIT_FOLDS,salt)
    seen=set()
    pred=np.full(len(ci),np.nan,dtype=np.float64)
    audit=[]
    for k,subset_local in enumerate(folds):
        hold=[ci[j] for j in subset_local]
        if not hold:raise RuntimeError("Empty chemical crossfit fold")
        fit=[c for c in ci if c not in set(hold)]
        if set(hold)&set(fit):raise RuntimeError("Chemical group leakage")
        if not fit:raise RuntimeError("Empty crossfit prior fit")
        # Group separation must hold even for neighbor connectivity.
        local_ref={i:k for k,group in enumerate(groups) for i in group}
        if {local_ref[j] for j in hold}&{local_ref[j] for j in fit}:
            raise RuntimeError("Scaffold/near-neighbor group crosses crossfit boundary")
        builder=prior_builder or fitted_prior
        p,ridge=builder(pi,fit,features,bits,y,groups)
        if len(p)!=len(bits):raise RuntimeError("Invalid prior length")
        for c in hold:
            if c in seen:raise RuntimeError("Duplicate OOF heldout compound")
            pred[ci.index(c)]=p[c]
            seen.add(c)
        audit.append({"crossfit_fold":k,"train_compounds":len(fit),
              "heldout_compounds":len(hold),"train_proteins":len(pi),
              "prior_ridge":ridge,
              "heldout_neighbor_components":len({local_ref[c] for c in hold})})
    if seen!=set(ci) or not np.isfinite(pred).all():
        raise RuntimeError("Cross-fitting incomplete or nonfinite")
    if with_audit:return pred,audit
    return pred

def training_pair(pi,ci,queries_pi,queries_ci,sm,bits,y,emb,full_prior,oof_train_prior):
    """Train SAME neural model for both residual targets with SAME init/seeds."""
    scale=float(y[np.ix_(pi,ci)].std())
    if not (scale>0):raise RuntimeError("Nonpositive train-only label scale")
    x=torch.tensor(bits,dtype=torch.float32,device="cuda")
    mols=[sm[j] for j in ci]
    out={arm:[] for arm in ARMS}
    training_info={arm:[] for arm in ARMS}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        initial=RepairedSequenceFoundation().cuda()
        weights={k:v.detach().clone() for k,v in initial.state_dict().items()}
        del initial
        for arm in ARMS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model=RepairedSequenceFoundation().cuda()
            model.load_state_dict(weights,strict=True)
            nn.init.zeros_(model.pair_head[-1].weight)
            nn.init.zeros_(model.pair_head[-1].bias)
            optim=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
            offsets=full_prior[ci] if arm=="insample" else oof_train_prior
            losses=[]
            for epoch in range(1,EPOCHS+1):
                model.train()
                vals=[]
                for r in np.random.default_rng(seed*100+epoch).permutation(len(pi)):
                    pid=pi[r]
                    optim.zero_grad(set_to_none=True)
                    preds=forward(model,model.compound(mols,x[ci]),emb[pid])
                    target=torch.tensor((y[pid,ci]-offsets)/scale,device="cuda",
                                        dtype=torch.float32)
                    loss=(preds-target).square().mean()
                    if not torch.isfinite(loss):raise RuntimeError("Training became nonfinite")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optim.step()
                    vals.append(float(loss.detach()))
                losses.append(float(np.mean(vals)))
            model.eval()
            with torch.inference_mode():
                cx=model.compound(sm,x)
                prediction=np.stack([forward(model,cx,emb[p]).float().cpu().numpy()
                                    for p in queries_pi]).astype(np.float64)
                prediction=prediction*scale+full_prior[None,:]
            out[arm].append(prediction[:,queries_ci])
            training_info[arm].append({"seed":seed,
                         "train_loss_start":losses[0],
                         "train_loss_end":losses[-1]})
            del model,optim
            torch.cuda.empty_cache()
        del weights
    return {arm:np.stack(out[arm]) for arm in ARMS},training_info

def choose_weight(preds):
    # Four INNER fold cases contain only inner validation observations from
    # each outer TRAIN rectangle.
    candidates=[]
    for w in WEIGHTS:
        s=[metric(c["prior"]+w*(c["ensemble"]-c["prior"]),c["truth"])["macro_rmse"]
           for c in preds]
        candidates.append({"weight":w,"inner_macro_rmse":float(np.mean(s)),
                           "per_inner_rmse":[float(z) for z in s]})
    selected=min(candidates,key=lambda x:(x["inner_macro_rmse"],x["weight"]))
    return selected,candidates

def main():
    env=configure_determinism()
    if OUT.exists():
        raise RuntimeError("Refusing to reuse/overwrite existing experiment directory")
    old=json.loads((PREV/"study_results.json").read_text(encoding="utf-8"))
    if old.get("status")!="HISTORICAL_TRAIN_ONLY_NESTED_CROSS_VALIDATION_NOT_PROSPECTIVE_TEST":
        raise RuntimeError("Previous split identity missing")
    trainrows=OLD/"data/engineering_pilot/rows.json"
    allrows=json.loads(trainrows.read_text(encoding="utf-8"))
    train=[r for r in allrows if r["split"]=="train"]
    pids=old["protein_hashes"];cids=old["compound_hashes"]
    if len(train)!=1296 or len(pids)!=24 or len(cids)!=54:raise RuntimeError("Changed data shape")
    ytable={(r["sequence"],r["compound"]):float(r["y"]) for r in train}
    y=np.array([[ytable[p,c] for c in cids] for p in pids],dtype=np.float64)
    smiles=json.loads((OLD/"data/staging/compounds.json").read_text())
    sm=[smiles[c] for c in cids]
    fpspath=OLD/"data/staging/morgan_radius2_2048_chiral.npz"
    with np.load(fpspath,allow_pickle=False) as z:
        fp=dict(zip(z["ids"].astype(str),z["bits"]))
    bits=np.stack([fp[c] for c in cids]).astype(np.float64)
    emb=[]
    for p in pids:
        obj=torch.load(OLD/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(p+".pt"),
                       map_location="cpu",weights_only=True)
        if obj["sequence_hash"]!=p:raise RuntimeError("ESM mismatch")
        emb.append(obj["embeddings"].float().cuda())
    features=np.stack([e.mean(0).cpu().numpy() for e in emb])
    groups=old["chemical_groups"]
    result={"status":"EXPLORATORY_NOT_PROSPECTIVE_RUNNING",
       "version":PROTO,
       "reason":"Test in-sample chemical self-similarity bias on TRAIN residual targets",
       "protocol_sha256":sha(HERE/"run_oof_residual_v47_20261109.py"),
       "legacy_protocol_sha256":sha(PREV/"study_results.json"),
       "train_file_sha256":sha(trainrows),
       "fingerprints_sha256":sha(fpspath),
       "determinism":env,"seed_choices":list(SEEDS),
       "four_inner_folds":INNER_FOLDS,"four_scaffold_oof_prior_folds":CROSS_FIT_FOLDS,
       "same_train_budget":EPOCHS,"stack_weights":list(WEIGHTS),
       "residual_fit_arms":list(ARMS),
       "outer_folds_previously_inspected":True,
       "protein_family_holdout":False,
       "no_bindingdb_or_original_dev":True,
       "no_published_perceiver_baseline":True,
       "limitations":[
         "Old three outer folds already seen many times: exploratory engineering only",
         "24 proteins x 54 compounds with protein identities but not families disjoint",
         "Crossfitted compound prior sees ~75% of training chemistry; final inference prior sees 100%",
         "Using observed outer labels only for scoring, not replay gating, training or model selection",
         "Deterministic algorithms enforced; no post hoc tolerance permitted",
         "Chosen blend weights have just four inner folds and will be noisy"
       ],"folds":[]}
    OUT.mkdir(parents=True)
    write_json(OUT/"PROTOCOL.json",result)
    started=time.monotonic()
    for fold in range(3):
        outer_pi=sorted(set(range(24))-set(old["outer_protein_folds"][fold]))
        outer_ci=sorted(set(range(54))-set(old["outer_chemical_folds"][fold]))
        hold_pi=sorted(old["outer_protein_folds"][fold])
        hold_ci=sorted(old["outer_chemical_folds"][fold])
        if (len(outer_pi),len(outer_ci),len(hold_pi),len(hold_ci))!=(16,36,8,18):
            raise RuntimeError("Outer folds changed")
        fullprior,ridge=fitted_prior(outer_pi,outer_ci,features,bits,y,groups)
        oof,audit=crossfit_prior(outer_pi,outer_ci,features,bits,y,groups,
                          "v47_outer_scaffold_crossfit_"+str(fold))
        train_truth=y[np.ix_(outer_pi,outer_ci)]
        in_res=train_truth-fullprior[outer_ci][None,:]
        oof_res=train_truth-oof[None,:]
        inflation={"in_sample_training_rmse":float(np.sqrt(np.mean(in_res*in_res))),
            "oof_training_rmse":float(np.sqrt(np.mean(oof_res*oof_res))),
            "in_sample_target_mean_abs_residual":float(np.mean(np.abs(in_res))),
            "oof_target_mean_abs_residual":float(np.mean(np.abs(oof_res)))}
        print(json.dumps({"fold":fold,"phase":"PRIOR_DIAGNOSIS",
            "in_sample":inflation["in_sample_training_rmse"],
            "oof":inflation["oof_training_rmse"]}),flush=True)

        # Four paired heldout protein-and-chemical groups INSIDE outer TRAIN only.
        pf=partition_groups([[i] for i in range(len(outer_pi))],
                            INNER_FOLDS,"v47_four_inner_protein_"+str(fold))
        cf=partition_groups(local_groups(groups,outer_ci),INNER_FOLDS,
                            "v47_four_inner_chemical_"+str(fold))
        inner={arm:[] for arm in ARMS}
        inner_stats=[]
        for k in range(INNER_FOLDS):
            ptest=[outer_pi[i] for i in pf[k]]
            ctest=[outer_ci[i] for i in cf[k]]
            pexclude=set(ptest);cexclude=set(ctest)
            pfit=[p for p in outer_pi if p not in pexclude]
            cfit=[c for c in outer_ci if c not in cexclude]
            if not ptest or not ctest or len(pfit)!=12 or set(cfit)&cexclude:
                raise RuntimeError("Four-fold inner partition broken")
            inner_prior,inner_ridge=fitted_prior(pfit,cfit,features,bits,y,groups)
            inner_oof,inner_audit=crossfit_prior(pfit,cfit,features,bits,y,groups,
                "v47_inner_crossfit_outer_"+str(fold)+"_inner_"+str(k))
            pred,history=training_pair(pfit,cfit,ptest,ctest,sm,bits,y,emb,
                                      inner_prior,inner_oof)
            truth=y[np.ix_(ptest,ctest)]
            base=np.broadcast_to(inner_prior[ctest],truth.shape).copy()
            for arm in ARMS:
                inner[arm].append({"prior":base,"ensemble":pred[arm].mean(0),
                                   "truth":truth})
            inner_stats.append({"inner_fold":k,"train_proteins":len(pfit),
                "train_compounds":len(cfit),"heldout_proteins":len(ptest),
                "heldout_compounds":len(ctest),"ridge":inner_ridge,
                "chemical_rmse":metric(base,truth)["macro_rmse"],
                "insample_neural_rmse":metric(pred["insample"].mean(0),truth)["macro_rmse"],
                "oof_neural_rmse":metric(pred["oof"].mean(0),truth)["macro_rmse"]})
            print(json.dumps({"fold":fold,"inner":k,"phase":"INNER_COMPLETE",
                 "n_train":(len(pfit),len(cfit)),
                 "baseline":inner_stats[-1]["chemical_rmse"],
                 "oof":inner_stats[-1]["oof_neural_rmse"]}),flush=True)
        chosen={}
        for arm in ARMS:
            selected,grid=choose_weight(inner[arm])
            chosen[arm]={"selection":selected,"grid":grid}

        # Outer holdout labels never used to choose weights or accept reruns.
        predictions,train_audit=training_pair(outer_pi,outer_ci,list(range(24)),
                               list(range(54)),sm,bits,y,emb,fullprior,oof)
        regimes={"double_cold":(hold_pi,hold_ci),
                 "cold_compound":(outer_pi,hold_ci),
                 "cold_protein":(hold_pi,outer_ci)}
        scores={}
        for name,(pidx,cidx) in regimes.items():
            truth=y[np.ix_(pidx,cidx)]
            base=np.broadcast_to(fullprior[cidx],truth.shape).copy()
            scores[name]={"chemical_only":metric(base,truth)}
            for arm in ARMS:
                allpred=predictions[arm][:,pidx,:][:,:,cidx]
                ens=allpred.mean(0)
                selected_w=chosen[arm]["selection"]["weight"]
                scores[name][arm+"_ensemble"]=metric(ens,truth)
                scores[name][arm+"_fixed_half"]=metric(base+.5*(ens-base),truth)
                scores[name][arm+"_inner_selected"]=metric(base+selected_w*(ens-base),truth)
                scores[name][arm+"_seed_mean_rmse"]=float(np.mean([metric(x,truth)["macro_rmse"] for x in allpred]))
            if name=="double_cold":
                cache=OUT/("fold_"+str(fold)+"_local_predictions.npz")
                np.savez_compressed(cache,truth=truth,chemical_only=base,
                     in_sample_seeds=predictions["insample"][:,pidx,:][:,:,cidx],
                     oof_seeds=predictions["oof"][:,pidx,:][:,:,cidx])
        record={"fold":fold,"outer_fit_proteins":len(outer_pi),
                 "outer_fit_compounds":len(outer_ci),"prior_ridge":ridge,
                 "prior_residual_diagnosis":inflation,
                 "outer_crossfit_prior_audit":audit,
                 "inner_validation":inner_stats,
                 "inner_selection":chosen,
                 "metrics":scores,
                 "local_pair_cache":"fold_"+str(fold)+"_local_predictions.npz"}
        result["folds"].append(record)
        write_json(OUT/"PROGRESS.json",result)
        print(json.dumps({"fold":fold,"phase":"OUTER_COMPLETE",
              "base":scores["double_cold"]["chemical_only"]["macro_rmse"],
              "insample":scores["double_cold"]["insample_inner_selected"]["macro_rmse"],
              "oof":scores["double_cold"]["oof_inner_selected"]["macro_rmse"],
              "cold_compound_oof":scores["cold_compound"]["oof_inner_selected"]["macro_rmse"],
              "weights":{arm:chosen[arm]["selection"]["weight"] for arm in ARMS}}),flush=True)

    aggregates={}
    for regime in ("double_cold","cold_compound","cold_protein"):
        keys=[k for k in result["folds"][0]["metrics"][regime] if not k.endswith("_seed_mean_rmse")]
        aggregates[regime]={}
        for k in keys:
            vals=[f["metrics"][regime][k]["macro_rmse"] for f in result["folds"]]
            aggregates[regime][k]={"target_macro_rmse":float(np.mean(vals)),
                    "per_fold":[float(v) for v in vals]}
    # Bootstrap unique heldout proteins on double-cold only, never 3-fold bootstrap.
    diffs=[]
    for f in result["folds"]:
        with np.load(OUT/f["local_pair_cache"],allow_pickle=False) as z:
            truth=z["truth"];base=z["chemical_only"]
            arm_preds={arm:z[("oof" if arm=="oof" else "in_sample")+"_seeds"].mean(0)
                       for arm in ARMS}
        for j in range(truth.shape[0]):
            br=float(np.sqrt(np.mean((base[j]-truth[j])**2)))
            dat={"baseline":br}
            for arm in ARMS:
                w=f["inner_selection"][arm]["selection"]["weight"]
                pred=base[j]+w*(arm_preds[arm][j]-base[j])
                dat[arm]=float(np.sqrt(np.mean((pred-truth[j])**2))-br)
            diffs.append(dat)
    assert len(diffs)==24
    rng=np.random.default_rng(1747)
    resamples=rng.integers(0,24,size=(10000,24))
    aggregates["paired_target_uncertainty"]={}
    for arm in ARMS:
        x=np.array([row[arm] for row in diffs])
        ci=np.percentile(x[resamples].mean(1),[2.5,97.5])
        aggregates["paired_target_uncertainty"][arm]={
            "mean_delta_rmse":float(x.mean()),
            "descriptive_bootstrap_interval_95":[float(ci[0]),float(ci[1])],
            "n_unique_heldout_proteins":24,
            "repeated_prior_outer_inspection":True}
    result["aggregates"]=aggregates
    result["status"]="COMPLETE_EXPLORATORY_REUSED_OUTER_SPLITS"
    result["elapsed_seconds"]=round(time.monotonic()-started,2)
    write_json(OUT/"RESULTS.json",result)
    print(json.dumps({"status":result["status"],"elapsed_s":result["elapsed_seconds"],
         "double_cold":{k:round(v["target_macro_rmse"],6) for k,v in aggregates["double_cold"].items()},
         "cold_compound":{k:round(v["target_macro_rmse"],6) for k,v in aggregates["cold_compound"].items()}}),flush=True)

if __name__=="__main__":
    with gpu_lease(OLD):
        main()
