"""Compare chemical/protein similarity affinity memory under cold group CV.

Four protein Smith-Waterman alignment-group folds and strict Murcko scaffold
exclusion in each outer fit. No original frozen dev or reserved test access.
"""
import argparse,hashlib,json,time

import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from dynacpi.balm_public_data_dev import ROOT,get_train_only
from dynacpi.cpi_similarity_memory_dev import (
    MEMORY_SPECS,neighbor_weighted_labels
)
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import (
    cache_feature_map,fit_predict as ridge_predict,bootstrap
)
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,fingerprint_rows,metric
)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import (
    HIST_PARAMS,data_features
)
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import (
    scaffold_key,scaffold_filter
)

RULES=("identity30_cover80","identity40_cover80")
NAMES=("train_mean","nested_esm8_reference",)+tuple(
    "memory_"+spec.name for spec in MEMORY_SPECS)+tuple(
    "ensemble_plus_memory_"+spec.name for spec in MEMORY_SPECS)

def memory_similarity(tr,va,gen,cache,esm_features,hcache):
    """Cache repeat chemical/protein vectors; similarity must not use labels."""
    ctr=fingerprint_rows(tr,gen,cache)
    cva=fingerprint_rows(va,gen,cache)
    cosine=(cva@ctr.T).toarray().astype(np.float32)
    def protein(rows):
        for r in rows:
            s=r["protein"]
            if s not in hcache:
                hcache[s]=hashlib.sha256(s.encode()).hexdigest()
        return np.stack([esm_features[hcache[r["protein"]]] for r in rows])
    p_tr=protein(tr)
    p_va=protein(va)
    pcos=p_va@p_tr.T
    np.clip(cosine,0,1,out=cosine)
    np.clip(pcos,-1,1,out=pcos)
    return cosine,pcos

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--group-rule",choices=RULES,required=True)
    p.add_argument("--paired-controls",action="store_true")
    args=p.parse_args()
    rule=args.group_rule
    suffix="_paired_controls" if args.paired_controls else ""
    OUT=ROOT/"research"/f"balm_joint_similarity_memory_sw_{rule}_scaffold_cold_traincv_dev{suffix}.json"
    if OUT.exists():raise FileExistsError(OUT)
    start=time.time()
    old=json.loads((ROOT/"research"/
       f"balm_esm8m_nested_sw_{rule}_murcko_scaffold_cold_traincv_dev.json").read_text())
    audit=json.loads((ROOT/"research/balm_train_only_smith_waterman_protein_group_audit_20261108.json").read_text())
    esm,manifest=cache_feature_map()
    raw=get_train_only()
    assert len(raw)==15538
    seqhash={r["protein"]:hashlib.sha256(r["protein"].encode()).hexdigest() for r in raw}
    assignment=audit["sequence_sha256_to_alignment_groups"][rule]
    rows=[dict(r,group=assignment[seqhash[r["protein"]]]) for r in raw]
    group=np.asarray([r["group"] for r in rows])
    scaffold={s:scaffold_key(s) for s in set(r["smiles"] for r in rows)}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={};hcache={};ccache={};pcache={}
    sumsq={name:[] for name in NAMES};counts=[];folds=[];seen=set()
    for fold,(itra,iva) in enumerate(GroupKFold(4).split(group,groups=group),1):
        held=[rows[int(i)] for i in iva]
        fit=collapse_training(scaffold_filter([rows[int(i)] for i in itra],held,scaffold))
        heldgroups=set(r["group"] for r in held)
        assert not heldgroups&seen
        seen|=heldgroups
        assert not heldgroups&{r["group"] for r in fit}
        assert not {scaffold[r["smiles"]] for r in fit}&{scaffold[r["smiles"]] for r in held}
        reference=old["outer_folds"][fold-1]
        assert len(fit)==reference["fit_unique_pairs"]
        assert len(held)==reference["heldout_pairs"]
        beta=float(reference["inherited_beta_selected_inside_this_outer_fold"])
        ytrain=np.asarray([r["label"] for r in fit],dtype=np.float64)
        mu=float(ytrain.mean())
        yvalid=np.asarray([r["label"] for r in held],dtype=np.float64)
        gvalid=np.asarray([r["group"] for r in held])
        ridge=ridge_predict(fit,held,gen,cache,esm,hcache)["chemical_kmer_esm8m"]
        c,p=data_features(fit,ccache,pcache)
        cv,pv=data_features(held,ccache,pcache)
        booster=HistGradientBoostingRegressor(**HIST_PARAMS)
        booster.fit(np.column_stack([c,p]),ytrain)
        boost=booster.predict(np.column_stack([cv,pv]))
        original_mix=.75*ridge+.25*boost
        baseline=mu+beta*(original_mix-mu)
        chemcos,protcos=memory_similarity(fit,held,gen,cache,esm,hcache)
        pred={
            "train_mean":np.full(len(held),mu),
            "nested_esm8_reference":baseline,
        }
        for spec in MEMORY_SPECS:
            nearest=neighbor_weighted_labels(chemcos,protcos,ytrain,mu,spec)
            # Predeclared conservative shrink of noisy-neighbor predictor.
            memory=mu+.5*(nearest-mu)
            pred["memory_"+spec.name]=memory
            # New three-arm predictor: 75% original ensemble, 25% memory.
            pred["ensemble_plus_memory_"+spec.name]=.75*baseline+.25*memory
        assert set(pred)==set(NAMES)
        scored={name:metric(yvalid,val,gvalid) for name,val in pred.items()}
        for metric_name in ("pooled_rmse","component_macro_rmse"):
            assert abs(scored["nested_esm8_reference"][metric_name]-
                reference["outer_metrics"]["nested_kmer_esm"][metric_name])<3e-5
        for heldgroup in sorted(heldgroups):
            mask=(gvalid==heldgroup)
            counts.append(int(mask.sum()))
            for name in NAMES:
                sumsq[name].append(float(np.square(yvalid[mask]-pred[name][mask]).sum()))
        item={"fold":fold,"fit_unique_pairs":len(fit),"heldout_original_rows":len(held),
              "heldout_alignment_components":len(heldgroups),"scaffold_overlap":0,
              "inherited_beta":beta,"scores":scored}
        folds.append(item)
        print("FOLD",fold,json.dumps(item),flush=True)
    assert sum(counts)==len(raw)
    assert len(counts)==audit["rules"][rule]["connected_components"]
    refs=("train_mean","nested_esm8_reference")
    if args.paired_controls:
        refs+=("ensemble_plus_memory_chemical_only_k64","ensemble_plus_memory_joint_esm8_k64")
    overall=bootstrap(sumsq,counts,references=refs)
    result={
        "status":"BALM_TRAIN_ONLY_DUAL_SIMILARITY_MEMORY_ALIGNMENT_AND_SCAFFOLD_COLD_EXPLORATORY",
        "protein_alignment_rule":rule,
        "train_original_rows":len(raw),
        "total_heldout_alignment_components":len(counts),
        "memory_specs":[vars(s) for s in MEMORY_SPECS],
        "similarity":"Normalized chiral Morgan2048 bit cosine; normalized frozen ESM2 8M protein cosine; topk similarity-weighted training labels.",
        "weighting":"Neighbor-weighted prediction shrunk halfway toward fit-fold training mean, then 25pct blended with existing 75pct ridge/25pct descriptor booster plus inherited beta.",
        "esm_cache_sha256":manifest["cache_sha256"],
        "folds":folds,
        "out_of_fold":overall,
        "paired_chemical_controls":bool(args.paired_controls),
        "seconds":round(time.time()-start,2),
        "caveats":[
            "Only frozen TRAIN records, never original inner-development or reserved-test labels/inputs.",
            "All hyperparameters are exploratory, not independent verification.",
            "Neighborhood memory uses only fold-training labels; no validation labels fitted.",
            "Alignment-groups are homology proxies, Murcko exclusion does not guarantee chemical dissimilarity.",
            "BALM source lacks assay DOI/method/construct, repeated identical inputs can disagree.",
        ],
    }
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print("FINAL",json.dumps({
        "rule":rule,"pooled":overall["pooled_rmse"],
        "macro":overall["component_macro_rmse"],
        "paired_vs_reference":{
            k:v for k,v in overall["bootstrap"]["comparisons"].items()
            if k.endswith("_minus_nested_esm8_reference")
        },"seconds":result["seconds"],
    }),flush=True)

if __name__=="__main__":main()

