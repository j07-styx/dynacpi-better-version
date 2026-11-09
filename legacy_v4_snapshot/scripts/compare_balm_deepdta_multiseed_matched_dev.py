"""Matched DeepDTA vs locked CPI, ESM-memory CPI and PerceiverCPI multi-seed.

Only the repeatedly inspected BALM inner-dev split; NO prospective TEST.
DeepDTA and PerceiverCPI are matched by frozen train/dev rows and seed IDs,
not by GPU compute or architecture input length.
"""
import hashlib,json,glob
from pathlib import Path
import numpy as np

from dynacpi.balm_public_data_dev import ROOT,get_train_dev
from scripts.compare_balm_perceiver_vs_ensemble_dev import (
    group_stats,paired_bootstrap,row_hash
)

DIR=ROOT/"research"
DEEP=DIR/"balm_deepdta_multiseed_dev_20261108"
PER=DIR/"balm_perceiver_multiseed_dev_20261108"
ENS=DIR/"balm_ensemble_dev_predictions_20261108.json"
MEM=DIR/"balm_memory_vs_perceiver_matched_innerdev_20261108.json"
RULES={10:[2103,2601,2602,2603,2604,2605],30:[2601,2602]}
DRAWS=5000

def metrics_per_model(seed_sse,counts):
    n=counts.astype(float)
    return {
        "per_seed_pooled_rmse":[float(x) for x in np.sqrt(seed_sse.sum(axis=1)/n.sum())],
        "per_seed_macro_rmse":[float(x) for x in np.sqrt(seed_sse/n).mean(axis=1)],
        "mean_of_seed_pooled_rmse":float(np.sqrt(seed_sse.sum(axis=1)/n.sum()).mean()),
        "mean_of_seed_macro_rmse":float(np.sqrt(seed_sse/n).mean(axis=1).mean())
    }

def paired_seed_families(a,b,n,draws=DRAWS,seed=20261108):
    """Bootstrap same groups for both seed families; delta A minus B."""
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    n=np.asarray(n,dtype=np.float64)
    assert a.ndim==2 and b.ndim==2 and a.shape[1]==b.shape[1]==len(n)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.min(n)>0
    total=n.sum()
    def score(sse):
        return np.array([
            np.sqrt(sse.sum(axis=1)/total).mean(),
            np.sqrt(sse/n).mean(axis=1).mean()
        ])
    point=score(a)-score(b)
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(n),size=(draws,len(n)))
    counts=n[ix].sum(axis=1)
    a_p=np.sqrt(a[:,ix].sum(axis=2)/counts).mean(axis=0)
    b_p=np.sqrt(b[:,ix].sum(axis=2)/counts).mean(axis=0)
    a_m=np.sqrt(a/n)[:,ix].mean(axis=2).mean(axis=0)
    b_m=np.sqrt(b/n)[:,ix].mean(axis=2).mean(axis=0)
    return {
        "delta_a_minus_b_pooled":float(point[0]),
        "delta_a_minus_b_macro":float(point[1]),
        "ci95_pooled":[float(x) for x in np.quantile(a_p-b_p,[.025,.975])],
        "ci95_macro":[float(x) for x in np.quantile(a_m-b_m,[.025,.975])],
        "bootstrap_draws":draws,
    }

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,choices=(10,30),required=True)
    args=p.parse_args()
    E=args.epochs
    OUT=DIR/f"balm_deepdta_vs_perceiver_and_cpi_paired_dev_20261108_ep{E}.json"
    if OUT.exists():raise FileExistsError(OUT)
    rows=get_train_dev()["inner_dev"]
    assert len(rows)==3891
    y=np.asarray([r["label"] for r in rows],dtype=np.float64)
    sha=row_hash(rows)
    group_keys=sorted({r["group"] for r in rows})
    groupidx={g:i for i,g in enumerate(group_keys)}
    gi=np.asarray([groupidx[r["group"]] for r in rows],dtype=np.int32)
    counts=np.bincount(gi,minlength=len(group_keys)).astype(float)
    assert len(group_keys)==173

    locked=json.loads(ENS.read_text(encoding="utf8"))
    mem=json.loads(MEM.read_text(encoding="utf8"))
    assert locked["dev_row_sha256"]==mem["dev_row_sha256"]==sha
    selected=locked["selected_by_train_cv_from_existing_report"]
    assert selected=="ridge_interaction_blend_0.25"
    baselines={
        "train_mean":np.full(len(y),locked["train_mean"]),
        "locked_ridge_interactions":np.asarray(locked["predictions"][selected]),
        "esm_memory_joint8_k64":np.asarray(mem["predictions_by_model"]["memory_joint_esm8_k64"]),
        "esm_memory_chemical_only_k64":np.asarray(mem["predictions_by_model"]["memory_chemical_only_k64"]),
    }
    individual={}
    for name,pred in baselines.items():
        assert np.isfinite(pred).all() and len(pred)==len(y)
        individual[name]=group_stats((pred-y)**2,gi,len(group_keys))
    deep=[];per=[];hist=[]
    for seed in RULES[E]:
        d=json.loads((DEEP/f"deepdta_seed{seed}_ep{E}.json").read_text(encoding="utf8"))
        q=json.loads((PER/f"perceiver_seed{seed}_ep{E}.json").read_text(encoding="utf8"))
        assert d["dev_row_sha256"]==q["dev_row_sha256"]==sha
        assert d["epochs"]==q["epochs"]==E and d["seed"]==q["seed"]==seed
        assert len(d["dev_predictions_by_epoch"])==len(q["dev_predictions_by_epoch"])==E
        pred_deep=np.asarray(d["dev_predictions_by_epoch"][-1])
        pred_per=np.asarray(q["dev_predictions_by_epoch"][-1])
        assert len(pred_deep)==len(pred_per)==3891
        deep.append(group_stats((pred_deep-y)**2,gi,len(group_keys))[0])
        per.append(group_stats((pred_per-y)**2,gi,len(group_keys))[0])
        best=min(d["history"],key=lambda r:r["pooled_dev_rmse"])
        hist.append({
            "seed":seed,
            "deepdta_final_pooled":d["history"][-1]["pooled_dev_rmse"],
            "deepdta_final_macro":d["history"][-1]["macro_group_dev_rmse"],
            "deepdta_best_dev_epoch":best["epoch"],
            "deepdta_best_dev_pooled":best["pooled_dev_rmse"],
            "perceiver_final_pooled":q["history"][-1]["pooled_dev_rmse"],
            "perceiver_final_macro":q["history"][-1]["macro_protein_group_dev_rmse"],
        })
    deep=np.asarray(deep);per=np.asarray(per)
    # Assert each method's group counts are equal, no row leaks.
    for name,(s,n) in individual.items():
        np.testing.assert_array_equal(n,counts)
    comps={}
    for name,(s,n) in individual.items():
        comps[f"{name}_minus_deepdta_seedmean_FINAL"]=paired_bootstrap(
            s,n,deep,np.repeat(n[None,:],len(deep),axis=0),
            draws=DRAWS,seed=20261108)
    deep_average_pred=np.mean([
        np.asarray(json.loads((DEEP/f"deepdta_seed{seed}_ep{E}.json").read_text())["dev_predictions_by_epoch"][-1])
        for seed in RULES[E]
    ],axis=0)
    avg_sse,avg_n=group_stats(np.square(deep_average_pred-y),gi,len(group_keys))
    comps["locked_ridge_interactions_minus_deepdta_AVERAGED_PREDICTIONS"]=paired_bootstrap(
        *individual["locked_ridge_interactions"],np.expand_dims(avg_sse,0),
        np.expand_dims(avg_n,0),draws=DRAWS,seed=20261108)
    comps["deepdta_seedmean_minus_perceiver_seedmean_FINAL"]=paired_seed_families(deep,per,counts)
    # Favor DeepDTA by choosing its best DEV epoch *inside each seed*; label-leaky and optimistic.
    best=[]
    for seed in RULES[E]:
        d=json.loads((DEEP/f"deepdta_seed{seed}_ep{E}.json").read_text())
        i=int(np.argmin([h["pooled_dev_rmse"] for h in d["history"]]))
        best.append(group_stats(
            np.square(np.asarray(d["dev_predictions_by_epoch"][i])-y),
            gi,len(group_keys))[0])
    best=np.asarray(best)
    comps["locked_ridge_interactions_minus_deepdta_BEST_DEV_EPOCH_OPTIMISTIC"]=paired_bootstrap(
        *individual["locked_ridge_interactions"],
        best,np.repeat(counts[None,:],len(best),axis=0),
        draws=DRAWS,seed=20261108)
    result={
        "status":"BALM_INNER_DEV_MATCHED_MULTI_SEED_DEEPDTA_PERCEIVER_CPI_NO_TEST",
        "epochs":E,"seeds":RULES[E],"dev_rows":len(rows),
        "dev_original_protein_groups":len(group_keys),
        "dev_row_sha256":sha,
        "seed_family_metrics":{
            "deepdta":metrics_per_model(deep,counts),
            "perceivercpi":metrics_per_model(per,counts),
        },
        "individual_seed_metrics":hist,
        "comparisons":comps,
        "source_models":{
            "locked":str(ENS.relative_to(ROOT)),
            "memory":str(MEM.relative_to(ROOT)),
            "deepdta":str(DEEP.relative_to(ROOT)),
            "perceiver":str(PER.relative_to(ROOT)),
        },
        "warnings":[
            "Frozen BALM inner-development repeatedly examined; not prospective test.",
            "DeepDTA-inspired CNN differs from the original paper training protocol.",
            "Final epoch is the primary comparison; best DEV epoch is optimistic, selected using validation labels.",
            "Each model has different parameter count/input lengths and unequal training compute; not paper-equivalent.",
            "173 protein groups use original approximate 3-mer grouping and no Murcko scaffold exclusion.",
            "BindingDB assay/construct provenance unavailable, repeated identical pair affinity disagreements.",
            "Bootstrap of development protein groups does not account for adaptive model selection.",
            "Scientific cohort gate remains blocked; no test evaluated.",
        ],
    }
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print("DONE",json.dumps({
        "epochs":E,"seed_family_metrics":result["seed_family_metrics"],
        "locked_vs_deepdta_FINAL":comps["locked_ridge_interactions_minus_deepdta_seedmean_FINAL"],
        "locked_vs_deepdta_BEST":comps["locked_ridge_interactions_minus_deepdta_BEST_DEV_EPOCH_OPTIMISTIC"],
        "deepdta_vs_perceiver":comps["deepdta_seedmean_minus_perceiver_seedmean_FINAL"],
    }),flush=True)

if __name__=="__main__":main()
