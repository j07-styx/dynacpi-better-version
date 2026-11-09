"""Compare frozen ESM2 35M versus 8M and their L2-normalized fusion.

All 639 pretrained representations from original BALM TRAIN proteins only.
Cross-validation uses training-only Smith-Waterman protein clusters plus
strict scaffold exclusion; no original inner-dev or prospective test labels.
"""
import argparse
import hashlib
import json
import math
import time

import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from dynacpi.balm_public_data_dev import ROOT, get_train_only
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import (
    cache_feature_map,fit_predict as ridge_predict,bootstrap
)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import (
    HIST_PARAMS,data_features,
)
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,metric,
)
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import (
    scaffold_key,scaffold_filter,
)

RULES=("identity30_cover80","identity40_cover80")
SOURCE35=ROOT/"data/reference/balm_esm2_35m_train_20261108/esm2_t12_35m_mean_layer12_w1022.npz"
MANIFEST35=ROOT/"research/balm_esm2_t12_35m_train_only_feature_manifest_dev.json"
MODELS=("train_mean","ridge_8m","ridge_35m","ridge_8m_35m_fusion",
        "nested_8m","nested_35m","nested_8m_35m_fusion")

def load_35m():
    if not SOURCE35.exists() or not MANIFEST35.exists():
        raise FileNotFoundError("Missing 35M TRAIN-only feature cache/manifest")
    manifest=json.loads(MANIFEST35.read_text(encoding="utf8"))
    assert manifest["upstream_pretrained"]=="esm2_t12_35M_UR50D"
    assert manifest["protein_embeddings"]==639
    assert manifest["embedding_dimensions"]==480
    assert hashlib.sha256(SOURCE35.read_bytes()).hexdigest()==manifest["cache_sha256"]
    with np.load(SOURCE35,allow_pickle=False) as cached:
        keys=cached["sequence_sha256"]
        matrix=cached["embedding"].astype(np.float32)
    assert len(keys)==639 and matrix.shape==(639,480)
    assert np.isfinite(matrix).all()
    length=np.linalg.norm(matrix,axis=1,keepdims=True)
    assert np.all(length>0)
    matrix/=length
    return {str(key):emb for key,emb in zip(keys,matrix)},manifest

def fused_features(features8,features35):
    assert set(features8)==set(features35) and len(features8)==639
    scale=1/math.sqrt(2.)
    fused={}
    for key in features8:
        a=features8[key];b=features35[key]
        assert a.shape==(320,) and b.shape==(480,)
        fused[key]=np.concatenate([a,b])*scale
    return fused

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--group-rule",required=True,choices=RULES)
    args=p.parse_args()
    rule=args.group_rule
    OUT=ROOT/"research"/f"balm_esm2_8m_vs_35m_fusion_sw_{rule}_scaffold_cold_traincv_dev.json"
    if OUT.exists():raise FileExistsError(OUT)
    t=time.time()
    old=json.loads((ROOT/"research"/
        f"balm_esm8m_nested_sw_{rule}_murcko_scaffold_cold_traincv_dev.json").read_text())
    alignment=json.loads((ROOT/"research/balm_train_only_smith_waterman_protein_group_audit_20261108.json").read_text())
    emb8,manifest8=cache_feature_map()
    emb35,manifest35=load_35m()
    fused=fused_features(emb8,emb35)
    raw=get_train_only()
    assert len(raw)==15538
    hashes={r["protein"]:hashlib.sha256(r["protein"].encode()).hexdigest() for r in raw}
    assigned=alignment["sequence_sha256_to_alignment_groups"][rule]
    assert set(hashes.values())==set(emb35)==set(emb8)==set(assigned)
    train=[dict(r,group=assigned[hashes[r["protein"]]]) for r in raw]
    groups=np.asarray([r["group"] for r in train])
    scaffold={s:scaffold_key(s) for s in set(r["smiles"] for r in train)}
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fp_cache={};hcache={};chem_cache={};prot_cache={}
    sq={name:[] for name in MODELS}
    counts=[];folds=[];seen=set()
    for fold,(itra,iva) in enumerate(GroupKFold(4).split(groups,groups=groups),1):
        held=[train[int(i)] for i in iva]
        fit=collapse_training(scaffold_filter([train[int(i)] for i in itra],held,scaffold))
        fold_old=old["outer_folds"][fold-1]
        assert len(fit)==fold_old["fit_unique_pairs"]
        assert len(held)==fold_old["heldout_pairs"]
        group_set=set(r["group"] for r in held)
        assert not group_set & seen
        seen|=group_set
        assert not group_set & {r["group"] for r in fit}
        assert not {scaffold[r["smiles"]] for r in held} & {scaffold[r["smiles"]] for r in fit}
        beta=float(fold_old["inherited_beta_selected_inside_this_outer_fold"])
        ridge8=ridge_predict(fit,held,generator,fp_cache,emb8,hcache)
        ridge35=ridge_predict(fit,held,generator,fp_cache,emb35,hcache)
        ridge_fusion=ridge_predict(fit,held,generator,fp_cache,fused,hcache)
        chemical,protein=data_features(fit,chem_cache,prot_cache)
        cv,pv=data_features(held,chem_cache,prot_cache)
        booster=HistGradientBoostingRegressor(**HIST_PARAMS)
        booster.fit(np.column_stack([chemical,protein]),
                    np.asarray([r["label"] for r in fit],dtype=np.float64))
        bpred=booster.predict(np.column_stack([cv,pv]))
        mu=float(np.mean([r["label"] for r in fit]))
        y=np.asarray([r["label"] for r in held],dtype=np.float64)
        g=np.asarray([r["group"] for r in held])
        pred={"train_mean":np.full(len(held),mu)}
        for name,model in (("8m",ridge8),
                           ("35m",ridge35),("8m_35m_fusion",ridge_fusion)):
            ridge=model["chemical_kmer_esm8m"]
            pred["ridge_"+name]=ridge
            mix=.75*ridge+.25*bpred
            pred["nested_"+name]=mu+beta*(mix-mu)
        assert set(pred)==set(MODELS)
        observed={name:metric(y,values,g) for name,values in pred.items()}
        for metric_name in ("pooled_rmse","component_macro_rmse"):
            assert abs(observed["nested_8m"][metric_name]-
                fold_old["outer_metrics"]["nested_kmer_esm"][metric_name])<3e-5
        for group in sorted(group_set):
            mask=g==group
            counts.append(int(mask.sum()))
            for name in MODELS:
                sq[name].append(float(np.square(y[mask]-pred[name][mask]).sum()))
        fold_rec={
            "fold":fold,"fit_unique_pairs":len(fit),"heldout_rows":len(held),
            "heldout_alignment_components":len(group_set),"scaffold_overlap":0,
            "inherited_beta":beta,"scores":observed,
        }
        folds.append(fold_rec)
        print("FOLD",fold,json.dumps(fold_rec),flush=True)
    assert seen==set(groups)
    assert sum(counts)==len(raw) and len(counts)==alignment["rules"][rule]["connected_components"]
    summary=bootstrap(sq,counts,references=("train_mean","nested_8m","nested_35m"))
    output={
        "status":"EXPLORATORY_FROZEN_TRAIN_ONLY_ESM2_35M_VS_8M_AND_FUSION_SW_SCAFFOLD_COLD",
        "protein_alignment_rule":rule,
        "training_rows":len(train),
        "alignment_groups":len(seen),
        "strict_scaffold_exclusion":"RDKit Bemis Murcko including common acyclic group",
        "encoder_8m":"ESM2 t6 8M layer6 pooled 320-d frozen",
        "encoder_35m":"ESM2 t12 35M layer12 pooled 480-d frozen",
        "encoder_8m_sha256":manifest8["cache_sha256"],
        "encoder_35m_sha256":manifest35["cache_sha256"],
        "fusion_rule":"L2 normalize each embedding, concatenate divided by sqrt(2), retain fixed existing Ridge protein feature coefficient 0.25",
        "architecture":"75pct Morgan and protein k-mer plus ESM Ridge, 25pct unchanged joint chemical and AA composition booster, inherited training-nested beta",
        "models":MODELS,
        "outer_folds":folds,
        "out_of_fold":summary,
        "seconds":round(time.time()-t,2),
        "caveats":[
            "Original frozen TRAIN only; no original development/test labels or sequences used.",
            "All architecture variants motivated by repeated exploratory analysis, not independent confirmation.",
            "Protein groups are alignment identity/coverage proxies, not independently certified homology.",
            "Murcko scaffold exclusion does not remove all structural analogs.",
            "BALM source lacks experimental assay DOI/method/construct; conflicting repeated labels.",
            "Both encoders truncate 85 long TRAIN sequences to first 1022 amino acids.",
            "All 5000 group bootstrap confidence intervals are descriptive, not test-set significance.",
        ],
    }
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print("FINAL",json.dumps({
        "rule":rule,
        "pooled":summary["pooled_rmse"],
        "macro":summary["component_macro_rmse"],
        "vs_8m":{
            k:v for k,v in summary["bootstrap"]["comparisons"].items()
            if k.endswith("_minus_nested_8m")
        },
        "duration":output["seconds"],
    }),flush=True)

if __name__=="__main__":main()
