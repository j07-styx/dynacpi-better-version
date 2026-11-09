"""Three deranged-protein ESM2 negative controls on strict BALM scaffold-cold CV.

Keep chemical inputs, true protein k-mer features, boosting branch, targets,
scaffolds, train-only inherited betas identical. Derange ONLY frozen ESM2 protein
embeddings by a global deterministic bijection across the 639 TRAIN proteins.
No frozen development or reserved test labels or features accessed.
"""
import hashlib,json,time
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold

from dynacpi.balm_public_data_dev import ROOT,get_train_only
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import (
    cache_feature_map,fit_predict as ridge_compare,bootstrap
)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import (
    fit_predict as joint_compare
)
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import collapse_training,metric
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import scaffold_key,scaffold_filter

OUT=ROOT/"research/balm_esm8m_protein_derangement_negative_controls_scaffold_cv_dev.json"
ORIGINAL=ROOT/"research/balm_esm8m_augmented_joint_ensemble_scaffold_cold_traincv_dev.json"
SEEDS=(2103,2104,2105)
NAMES=("train_mean","nested_standard","real_esm") + tuple("deranged_esm_"+str(s) for s in SEEDS)

def shuffled_feature_map(features,seed):
    keys=sorted(features)
    rng=np.random.default_rng(seed)
    perm=np.arange(len(keys))
    for trial in range(25):
        perm=rng.permutation(len(keys))
        if np.all(perm!=np.arange(len(keys))):break
    else:
        raise AssertionError("Cannot generate a derangement.")
    shuffled={key:features[keys[int(perm[i])]] for i,key in enumerate(keys)}
    assert all(not np.array_equal(shuffled[key],features[key]) for key in keys)
    digest=hashlib.sha256("".join(keys[int(i)] for i in perm).encode()).hexdigest()
    return shuffled,digest

def main():
    if OUT.exists():raise FileExistsError(OUT)
    t=time.time()
    embeddings,meta=cache_feature_map()
    deranged={}
    permutation_hashes={}
    for seed in SEEDS:
        deranged[seed],permutation_hashes[str(seed)]=shuffled_feature_map(embeddings,seed)
    prior=json.loads(ORIGINAL.read_text(encoding="utf-8"))
    raw=get_train_only()
    assert len(raw)==15538
    groups=np.asarray([r["group"] for r in raw])
    scaffold={s:scaffold_key(s) for s in sorted(set(r["smiles"] for r in raw))}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={};hcache={};dcache={};pcache={}
    fold_scores=[];component_sq={n:[] for n in NAMES};component_count=[]
    for fold,(it,iv) in enumerate(GroupKFold(n_splits=4).split(groups,groups=groups),1):
        held=[raw[int(i)] for i in iv]
        tr=collapse_training(scaffold_filter([raw[int(i)] for i in it],held,scaffold))
        old=prior["folds"][fold-1]
        assert old["fold"]==fold and old["outer_train_unique"]==len(tr)
        assert len(held)==old["holdout_original_pairs"]
        assert not {scaffold[x["smiles"]] for x in tr}&{scaffold[x["smiles"]] for x in held}
        beta=float(old["inherited_nested_beta_from_train_cv"])
        joint=joint_compare(tr,held,gen,cache,dcache,pcache)
        true=ridge_compare(tr,held,gen,cache,embeddings,hcache)
        np.testing.assert_allclose(joint["ridge"],true["chemical_plus_kmer"],rtol=1e-5,atol=1e-4)
        y=np.asarray([r["label"] for r in held],dtype=np.float64)
        g=np.asarray([r["group"] for r in held])
        mean=float(np.mean([r["label"] for r in tr]))
        booster=joint["boost_joint"]
        standard=.75*true["chemical_plus_kmer"]+.25*booster
        real=.75*true["chemical_kmer_esm8m"]+.25*booster
        pred={
            "train_mean":np.full(len(held),mean),
            "nested_standard":mean+beta*(standard-mean),
            "real_esm":mean+beta*(real-mean),
        }
        for seed in SEEDS:
            ctrl=ridge_compare(tr,held,gen,cache,deranged[seed],hcache)
            deranged_pred=.75*ctrl["chemical_kmer_esm8m"]+.25*booster
            pred["deranged_esm_"+str(seed)]=mean+beta*(deranged_pred-mean)
        for held_group in sorted(set(g)):
            mask=g==held_group
            component_count.append(int(mask.sum()))
            for name in NAMES:
                component_sq[name].append(float(np.sum(np.square(y[mask]-pred[name][mask]))))
        metrics={name:metric(y,values,g) for name,values in pred.items()}
        for key,prior_key in (("nested_standard","nested_standard"),
                              ("real_esm","nested_esm8m_with_inherited_beta")):
            for m in ("pooled_rmse","component_macro_rmse"):
                assert abs(metrics[key][m]-old["metrics"][prior_key][m])<2e-5
        report={"fold":fold,"outer_train_unique":len(tr),"holdout_rows":len(held),
                "heldout_components":len(set(g)),"scaffold_overlap_count":0,
                "inherited_nested_beta":beta,"metrics":metrics}
        fold_scores.append(report)
        print("FOLD",fold,json.dumps(report),flush=True)
    assert len(component_count)==519 and sum(component_count)==len(raw)
    oof=bootstrap(component_sq,component_count,references=("train_mean","nested_standard","real_esm"))
    mean_scores={name:{
        m:float(np.mean([fold["metrics"][name][m] for fold in fold_scores]))
        for m in ("pooled_rmse","component_macro_rmse")
    } for name in NAMES}
    output={
        "status":"EXPLORATORY_BALM_TRAIN_ONLY_SCAFFOLD_COLD_FROZEN_ESM_PROTEIN_DERANGEMENT_NEGATIVE_CONTROLS",
        "control_definition":"Random global one-to-one reassignment of ESM2 protein vectors across all 639 TRAIN sequence hashes; true chemical and protein k-mer features and joint booster unmodified",
        "derangement_seeds":SEEDS,"derangement_mapping_sha256":permutation_hashes,
        "source_esm_feature_cache_sha256":meta["cache_sha256"],
        "original_esm_ensemble_report":ORIGINAL.name,
        "folds":fold_scores,
        "fold_mean":mean_scores,
        "out_of_fold":oof,"seconds":round(time.time()-t,2),
        "caveats":[
           "Three negative-control permutations do not approximate the full permutation null distribution.",
           "Same repeated TRAIN-only experimental folds and previously explored architecture.",
           "Protein 3-mer group proxy not validated homology; source assay provenance absent.",
           "No frozen development or prospective reserved test outcomes used or evaluated.",
           "ESM2 8M N-terminal window 1022 truncates 85 of 639 training proteins.",
        ],
    }
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print("FINAL",json.dumps({
        "pooled":oof["pooled_rmse"],
        "macro":oof["component_macro_rmse"],
        "paired_to_real":{k:v for k,v in oof["bootstrap"]["comparisons"].items() if k.endswith("_minus_real_esm")},
        "seconds":output["seconds"]
    }),flush=True)

if __name__=="__main__":main()
