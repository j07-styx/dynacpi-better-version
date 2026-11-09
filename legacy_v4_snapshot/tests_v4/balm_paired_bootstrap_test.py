"""Engineering test for the paired group bootstrap used in the Perceiver-vs-ensemble comparison."""
import numpy as np
from scripts.compare_balm_perceiver_vs_ensemble_dev import group_stats,metrics_from,paired_bootstrap

def make(ng=40,per=12,seed=0):
    rng=np.random.default_rng(seed)
    gidx=np.repeat(np.arange(ng),per)
    y=rng.normal(size=ng*per)
    return rng,gidx,y

def test_identical_models_have_zero_delta():
    rng,gidx,y=make()
    pred=y+rng.normal(scale=.5,size=len(y))
    s,n=group_stats((pred-y)**2,gidx,40)
    out=paired_bootstrap(s,n,np.array([s]),np.array([n]),draws=300)
    assert abs(out["delta_ens_minus_perceiver"]["pooled"])<1e-12
    assert out["ci95_pooled"]==[0.0,0.0] and out["ci95_macro"]==[0.0,0.0]

def test_clearly_better_model_has_negative_ci():
    rng,gidx,y=make(seed=1)
    good=y+rng.normal(scale=.2,size=len(y));bad=y+rng.normal(scale=1.,size=len(y))
    gs,gn=group_stats((good-y)**2,gidx,40);bs,bn=group_stats((bad-y)**2,gidx,40)
    out=paired_bootstrap(gs,gn,np.array([bs]),np.array([bn]),draws=500)
    assert out["ci95_pooled"][1]<0 and out["ci95_macro"][1]<0
    assert out["frac_draws_ensemble_better_pooled"]>.99

def test_metrics_match_direct_computation():
    rng,gidx,y=make(seed=2)
    pred=y+rng.normal(size=len(y));e=(pred-y)**2
    s,n=group_stats(e,gidx,40);pooled,macro=metrics_from(s,n)
    assert abs(pooled-np.sqrt(e.mean()))<1e-12
    assert abs(macro-np.mean([np.sqrt(e[gidx==g].mean()) for g in range(40)]))<1e-12

def test_bootstrap_is_reproducible():
    rng,gidx,y=make(seed=3)
    a=y+rng.normal(size=len(y));b=y+rng.normal(size=len(y))
    sa,na=group_stats((a-y)**2,gidx,40);sb,nb=group_stats((b-y)**2,gidx,40)
    r1=paired_bootstrap(sa,na,np.array([sb]),np.array([nb]),draws=200)
    r2=paired_bootstrap(sa,na,np.array([sb]),np.array([nb]),draws=200)
    assert r1==r2

if __name__=="__main__":
    for f in (test_identical_models_have_zero_delta,test_clearly_better_model_has_negative_ci,
              test_metrics_match_direct_computation,test_bootstrap_is_reproducible):
        f();print("PASS",f.__name__)
