"""Paired protein-group bootstrap: locked BALM ensemble vs matched multi-seed PerceiverCPI.

Inputs (all inner-dev, no reserved test):
  research/balm_ensemble_dev_predictions_20261108.json
  research/balm_perceiver_multiseed_dev_20261108/perceiver_seed*_ep*.json
Epoch rules are fixed here BEFORE looking at results:
  PRIMARY   : final epoch of every seed (no dev-based selection).
  OPTIMISTIC: per-seed epoch with lowest dev pooled RMSE (favours PerceiverCPI).
Delta = ensemble - perceiver; NEGATIVE favours the ensemble. Groups are resampled with
replacement (5000 draws, seed 20261108); seeds are averaged inside each draw.
Usage: python -m scripts.compare_balm_perceiver_vs_ensemble_dev
"""
import json,glob,hashlib
import numpy as np
from dynacpi.balm_public_data_dev import get_train_dev,ROOT

ENS=ROOT/"research/balm_ensemble_dev_predictions_20261108.json"
PDIR=ROOT/"research/balm_perceiver_multiseed_dev_20261108"
OUT=ROOT/"research/balm_perceiver_vs_ensemble_paired_dev_20261108.json"
DRAWS=5000

def group_stats(sqerr,gidx,ng):
    sse=np.bincount(gidx,weights=sqerr,minlength=ng)
    n=np.bincount(gidx,minlength=ng).astype(float)
    return sse,n

def metrics_from(sse,n,w=None):
    """Pooled and group-macro RMSE from per-group SSE/count (optionally resample weights)."""
    if w is not None:sse=sse*w;n=n*w
    keep=n>0
    return float(np.sqrt(sse.sum()/n.sum())),float(np.mean(np.sqrt(sse[keep]/n[keep])))

def paired_bootstrap(ens_sse,ens_n,per_sse,per_n,draws=DRAWS,seed=20261108):
    """ens_*: [G]; per_*: [S,G] (S seeds averaged inside each draw). Returns point and CI deltas."""
    ng=len(ens_sse);rng=np.random.default_rng(seed)
    def both(w):
        e=metrics_from(ens_sse,ens_n,w)
        p=np.mean([metrics_from(per_sse[s],per_n[s],w) for s in range(len(per_sse))],axis=0)
        return np.array(e),p
    e0,p0=both(None)
    deltas=[]
    for _ in range(draws):
        w=np.bincount(rng.integers(0,ng,ng),minlength=ng).astype(float)
        e,p=both(w);deltas.append(e-p)
    d=np.asarray(deltas)
    lo=np.percentile(d,2.5,axis=0);hi=np.percentile(d,97.5,axis=0)
    return {"ensemble":{"pooled":e0[0],"macro":e0[1]},"perceiver_mean_over_seeds":{"pooled":p0[0],"macro":p0[1]},
            "delta_ens_minus_perceiver":{"pooled":float(e0[0]-p0[0]),"macro":float(e0[1]-p0[1])},
            "ci95_pooled":[float(lo[0]),float(hi[0])],"ci95_macro":[float(lo[1]),float(hi[1])],
            "frac_draws_ensemble_better_pooled":float(np.mean(d[:,0]<0)),
            "frac_draws_ensemble_better_macro":float(np.mean(d[:,1]<0))}

def row_hash(rows):
    h=hashlib.sha256()
    for r in rows:h.update(f"{r['protein']}|{r['smiles']}|{r['label']}\n".encode())
    return h.hexdigest()

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--epochs",type=int,default=10)
    EP=ap.parse_args().epochs
    OUT=ROOT/(f"research/balm_perceiver_vs_ensemble_paired_dev_20261108.json" if EP==10
              else f"research/balm_perceiver_vs_ensemble_paired_dev_20261108_ep{EP}.json")
    if OUT.exists():raise FileExistsError(OUT)
    dev=get_train_dev()["inner_dev"]
    y=np.asarray([r["label"] for r in dev],dtype=float)
    groups=sorted({r["group"] for r in dev});gi={g:i for i,g in enumerate(groups)}
    gidx=np.asarray([gi[r["group"]] for r in dev]);ng=len(groups)
    ens=json.loads(ENS.read_text());assert ens["dev_row_sha256"]==row_hash(dev)
    sel=ens["selected_by_train_cv_from_existing_report"]
    runs=[json.loads(open(p).read()) for p in sorted(glob.glob(str(PDIR/f"perceiver_seed*_ep{EP}.json")))]
    assert runs and all(r["dev_row_sha256"]==ens["dev_row_sha256"] for r in runs)
    seeds=[r["seed"] for r in runs]
    ens_sse={}
    for name in ("ridge",sel):
        pred=np.asarray(ens["predictions"][name]);ens_sse[name]=group_stats((pred-y)**2,gidx,ng)
    final=[group_stats((np.asarray(r["dev_predictions_by_epoch"][-1])-y)**2,gidx,ng) for r in runs]
    best_ep=[int(np.argmin([h["pooled_dev_rmse"] for h in r["history"]])) for r in runs]
    best=[group_stats((np.asarray(r["dev_predictions_by_epoch"][e])-y)**2,gidx,ng) for r,e in zip(runs,best_ep)]
    avg_pred=np.mean([np.asarray(r["dev_predictions_by_epoch"][-1]) for r in runs],axis=0)
    seedavg=[group_stats((avg_pred-y)**2,gidx,ng)]
    def pack(lst):return np.array([a for a,_ in lst]),np.array([b for _,b in lst])
    per_seed_final=[{"seed":r["seed"],"final_epoch":r["history"][-1],"best_epoch_by_dev":int(e)+1,
                     "best_epoch_pooled":r["history"][e]["pooled_dev_rmse"]} for r,e in zip(runs,best_ep)]
    comps={}
    for name in ("ridge",sel):
        es,en=ens_sse[name]
        comps[f"{name}_vs_perceiver_FINAL_EPOCH_PRIMARY"]=paired_bootstrap(es,en,*pack(final))
        comps[f"{name}_vs_perceiver_BEST_DEV_EPOCH_OPTIMISTIC"]=paired_bootstrap(es,en,*pack(best))
        comps[f"{name}_vs_perceiver_SEED_AVERAGED_PREDICTIONS_FINAL"]=paired_bootstrap(es,en,*pack(seedavg))
    result={"status":"BALM_FROZEN_INNER_DEV_PAIRED_COMPARISON_NO_TEST","seeds":seeds,"perceiver_epochs":EP,
            "dev_rows":len(dev),"dev_groups":ng,"bootstrap_draws":DRAWS,
            "ensemble_locked_choice":sel,"per_seed":per_seed_final,"comparisons":comps,
            "interpretation_rules":["Negative delta favours the ensemble.",
              "PRIMARY uses the final epoch of each seed; no dev-based epoch selection.",
              "BEST_DEV_EPOCH is optimistic for PerceiverCPI.",
              "Dev split was inspected repeatedly; this is NOT a test-set result.",
              "Different model families/compute: compare as benchmark systems, not as a mechanism claim."]}
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for k,v in comps.items():
        print(k,json.dumps({"d_pooled":round(v["delta_ens_minus_perceiver"]["pooled"],4),"ci_pooled":[round(x,4) for x in v["ci95_pooled"]],
              "d_macro":round(v["delta_ens_minus_perceiver"]["macro"],4),"ci_macro":[round(x,4) for x in v["ci95_macro"]]}))
    print("SAVED",OUT)

if __name__=="__main__":main()
