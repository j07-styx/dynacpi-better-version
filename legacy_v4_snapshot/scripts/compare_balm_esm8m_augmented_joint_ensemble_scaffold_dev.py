"""Frozen ESM2+protein-kmer Ridge inside fixed 75/25 joint CPI ensemble.

Use the exact four strict scaffold/protein cold TRAIN folds. Carry over betas
chosen by prior nested inner TRAIN-only folds; do not retune on outer outcomes.
Compare matched ordinary joint booster while varying only the Ridge features.
No frozen development or reserved test scoring.
"""
import json,time,numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold

from dynacpi.balm_public_data_dev import ROOT,get_train_only
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import (
    cache_feature_map,fit_predict as ridge_comparison,bootstrap
)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import (
    fit_predict as joint_predict)
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import (
    scaffold_key,scaffold_filter)
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,metric)

OUT=ROOT/"research/balm_esm8m_augmented_joint_ensemble_scaffold_cold_traincv_dev.json"
PRIOR=ROOT/"research/balm_bindingdb_nested_scaffold_cold_mean_shrinkage_traincv_dev.json"
NAMES=("train_mean","ridge_kmer","ridge_kmer_esm8m",
       "joint_standard","joint_esm8m",
       "nested_standard","nested_esm8m_with_inherited_beta")

def main():
    if OUT.exists():raise FileExistsError(OUT)
    t=time.time()
    features,meta=cache_feature_map()
    earlier=json.loads(PRIOR.read_text(encoding="utf-8"))
    assert len(earlier["outer_folds"])==4
    raw=get_train_only();assert len(raw)==15538
    group=np.asarray([r["group"] for r in raw])
    scaffold={s:scaffold_key(s) for s in sorted(set(r["smiles"] for r in raw))}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fp={};hcache={};cd={};pd={}
    component_sum_sq={n:[] for n in NAMES};component_n=[];folds=[]
    for i,(itr,iva) in enumerate(GroupKFold(n_splits=4).split(group,groups=group),1):
        held=[raw[int(j)] for j in iva]
        fit=collapse_training(scaffold_filter([raw[int(j)] for j in itr],held,scaffold))
        control=earlier["outer_folds"][i-1]
        assert control["outer_fold"]==i and control["outer_fit_unique_pairs"]==len(fit)
        assert control["heldout_rows"]==len(held)
        assert not {r["group"] for r in fit}&{r["group"] for r in held}
        assert not {scaffold[r["smiles"]] for r in fit}&{scaffold[r["smiles"]] for r in held}
        beta=float(control["inner_selected_beta"])
        ridge=ridge_comparison(fit,held,gen,fp,features,hcache)
        joint=joint_predict(fit,held,gen,fp,cd,pd)
        # Both code paths fit the exact same classical k-mer Ridge; verify.
        np.testing.assert_allclose(
            ridge["chemical_plus_kmer"],joint["ridge"],atol=1e-4,rtol=1e-5)
        traditional=ridge["chemical_plus_kmer"]
        enriched=ridge["chemical_kmer_esm8m"]
        booster=joint["boost_joint"]
        mu=float(np.mean([r["label"] for r in fit]))
        y=np.asarray([r["label"] for r in held],dtype=np.float64)
        g=np.asarray([r["group"] for r in held])
        traditional_joint=.75*traditional+.25*booster
        enriched_joint=.75*enriched+.25*booster
        predictions={
            "train_mean":np.full_like(y,mu),
            "ridge_kmer":traditional,
            "ridge_kmer_esm8m":enriched,
            "joint_standard":traditional_joint,
            "joint_esm8m":enriched_joint,
            "nested_standard":mu+beta*(traditional_joint-mu),
            "nested_esm8m_with_inherited_beta":mu+beta*(enriched_joint-mu),
        }
        for held_group in sorted(set(g)):
            mask=(g==held_group)
            component_n.append(int(mask.sum()))
            for name in NAMES:
                component_sum_sq[name].append(float(np.sum(np.square(y[mask]-predictions[name][mask]))))
        scored={n:metric(y,predictions[n],g) for n in NAMES}
        item={
            "fold":i,"outer_train_unique":len(fit),
            "holdout_original_pairs":len(held),
            "heldout_components":len(set(g)),
            "murcko_overlap_count":len({scaffold[r["smiles"]] for r in fit}&{scaffold[r["smiles"]] for r in held}),
            "inherited_nested_beta_from_train_cv":beta,
            "metrics":scored,
        }
        folds.append(item)
        # Preserve independent reference experiment point estimates as check.
        for metric_name in ("pooled_rmse","component_macro_rmse"):
            assert abs(scored["nested_standard"][metric_name]-
                       control["outer_metrics"]["nested_selected_shrinkage"][metric_name]) < 2e-5
        print("FOLD",i,json.dumps(item),flush=True)
    assert sum(component_n)==len(raw) and len(component_n)==519
    metric_summary=bootstrap(component_sum_sq,component_n,references=("train_mean","ridge_kmer","nested_standard"))
    output={
        "status":"ESM8M_AUGMENTED_JOINT_ENSEMBLE_FROZEN_TRAIN_ONLY_SCAFFOLD_COLD_NO_DEV_OR_TEST",
        "feature_cache_sha256":meta["cache_sha256"],
        "source_prior_nested_report":PRIOR.name,
        "scaffold_policy":"all Murcko scaffolds withheld including one shared acyclic group",
        "inherited_betas":[f["inherited_nested_beta_from_train_cv"] for f in folds],
        "models":NAMES,
        "protocol":"Keep the ordinary joint shallow descriptor booster and its fixed 25% ensemble coefficient; replace only classical Morgan+protein-kmer Ridge with frozen ESM2-8M-augmented Ridge. Reuse each beta selected strictly inside original TRAIN folds.",
        "folds":folds,"out_of_fold":metric_summary,
        "seconds":round(time.time()-t,2),
        "caveats":[
            "No original frozen dev or prospective test outcomes used.",
            "Model choice motivated by previously inspected experiments, so these are exploratory.",
            "Frozen ESM8M only mean-pools N-terminal first 1022 residues; 85 proteins truncated.",
            "Approximate protein trigram graph, not true homology identity; source assay provenance missing.",
            "Four outer folds reuse the exact same original TRAIN records; no external dataset confirmation."
        ],
    }
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    compare=metric_summary["bootstrap"]["comparisons"]
    print("FINAL",json.dumps({
        "pooled":metric_summary["pooled_rmse"],
        "macro":metric_summary["component_macro_rmse"],
        "relative_to_standard":{
            "esm_augmented_nested_minus_prior_nested_pooled":
                metric_summary["pooled_rmse"]["nested_esm8m_with_inherited_beta"]-
                metric_summary["pooled_rmse"]["nested_standard"],
            "esm_augmented_nested_minus_prior_nested_macro":
                metric_summary["component_macro_rmse"]["nested_esm8m_with_inherited_beta"]-
                metric_summary["component_macro_rmse"]["nested_standard"],
        },
        "duration":output["seconds"],
    }),flush=True)

if __name__=="__main__":main()

