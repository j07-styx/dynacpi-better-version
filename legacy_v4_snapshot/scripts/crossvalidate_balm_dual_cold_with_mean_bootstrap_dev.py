"""BALM TRAIN-only paired protein/canonical-molecule cold GroupKFold challenge.

Outer fold target groups are disjoint. Training compounds with ANY canonical
isomeric SMILES equal to a heldout compound are excluded before fitting.
No frozen inner-dev or reserved test outcomes are used.
"""
import json,time,math
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold

from dynacpi.balm_public_data_dev import get_train_only,ROOT
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,metric)
from scripts.ablate_balm_bindingdb_ensemble_interaction_controls_dev import evaluate
from scripts.train_balm_bindingdb_chemical_domain_gated_ensemble_dev import nearest_chemical_cosine

OUT=ROOT/"research/balm_bindingdb_train_only_dual_cold_group_cv_with_mean_bootstrap.json"
MODES=(
    "train_mean","ridge","ridge_plus_chemical_boost","ridge_plus_joint_boost",
    "ridge_plus_explicit_interaction_boost",
)

def canonical_smiles(s):
    mol=Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError("Unparseable SMILES")
    return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)

def oof_bootstrap(component_sum_sq,component_counts,replicates=5000,seed=20261108):
    names=list(component_sum_sq)
    n=np.asarray(component_counts,dtype=np.int32)
    s=np.asarray([component_sum_sq[key] for key in names],dtype=np.float64)
    assert s.shape[1]==len(n)
    pooled=np.sqrt(s.sum(axis=1)/n.sum())
    macro=np.sqrt(s/n[None,:]).mean(axis=1)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(n),size=(replicates,len(n)),dtype=np.int32)
    bpool=np.sqrt(s[:,idx].sum(axis=2)/n[idx].sum(axis=1)[None,:])
    bmacro=np.sqrt(s/n[None,:])[:,idx].mean(axis=2)
    controls={}
    for reference in ("ridge","train_mean","ridge_plus_joint_boost"):
        b=names.index(reference)
        for a,candidate in enumerate(names):
            if a==b:continue
            diff=bpool[a]-bpool[b]
            mdiff=bmacro[a]-bmacro[b]
            controls[candidate+"_minus_"+reference]={
                "pooled_delta":float(pooled[a]-pooled[b]),
                "pooled_95pct_group_bootstrap_CI":[float(x) for x in np.quantile(diff,[.025,.975])],
                "macro_delta":float(macro[a]-macro[b]),
                "macro_95pct_group_bootstrap_CI":[float(x) for x in np.quantile(mdiff,[.025,.975])],
                "heldout_component_wins":int(np.sum(s[a]/n<s[b]/n)),
            }
    return {
        "macro_component_rmse":{name:float(macro[i]) for i,name in enumerate(names)},
        "pooled_rmse":{name:float(pooled[i]) for i,name in enumerate(names)},
        "paired_bootstrap":{"replicates":replicates,"seed":seed,"controls":controls}
    }

def main():
    if OUT.exists():raise FileExistsError(OUT)
    t=time.time()
    all_data=get_train_dev()
    raw=all_data["train"]
    assert len(raw)==15538
    groups=np.asarray([r["group"] for r in raw])
    all_groups=set(groups)
    # Canonicalization happens solely on frozen TRAIN. This is separate from
    # validation outcome values and never touches reserved test.
    smile_to_canonical={s:canonical_smiles(s) for s in sorted(set(r["smiles"] for r in raw))}
    keys=[smile_to_canonical[r["smiles"]] for r in raw]
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fpcache={};ccache={};pcache={}
    fold_data=[];component_sums={name:[] for name in MODES};component_counts=[];seen_groups=set()
    total_train_pairs=[]
    for fold,(itra,iva) in enumerate(GroupKFold(n_splits=4).split(groups,groups=groups),1):
        heldout_canonical={keys[int(i)] for i in iva}
        tr=[raw[int(i)] for i in itra if keys[int(i)] not in heldout_canonical]
        va=[raw[int(i)] for i in iva]
        tr=collapse_training(tr)
        assert len(tr)>1000,(fold,len(tr))
        assert set(r["group"] for r in tr).isdisjoint({r["group"] for r in va})
        train_can={smile_to_canonical[r["smiles"]] for r in tr}
        val_can={smile_to_canonical[r["smiles"]] for r in va}
        assert not train_can.intersection(val_can)
        unique_groups=set(r["group"] for r in va)
        assert seen_groups.isdisjoint(unique_groups)
        seen_groups.update(unique_groups)
        pred=evaluate(tr,va,gen,fpcache,ccache,pcache)
        sim=nearest_chemical_cosine(tr,va,gen,fpcache)
        y=np.asarray([r["label"] for r in va],dtype=np.float64)
        g=np.asarray([r["group"] for r in va])
        mu=np.mean([r["label"] for r in tr])
        pred["train_mean"]=np.full_like(y,mu)
        per_model={"train_mean":metric(y,np.full_like(y,mu),g)}
        per_model.update({name:metric(y,p,g) for name,p in pred.items()})
        for group in sorted(unique_groups):
            mask=(g==group)
            component_counts.append(int(mask.sum()))
            for name in MODES:
                component_sums[name].append(float(np.sum((y[mask]-pred[name][mask])**2)))
        entry={
            "fold":fold,"original_train_fold_rows":len(itra),
            "dual_cold_train_unique_rows":len(tr),
            "removed_training_rows":len(itra)-len([raw[int(i)] for i in itra if keys[int(i)] not in heldout_canonical]),
            "holdout_rows":len(va),
            "holdout_protein_components":len(unique_groups),
            "holdout_unique_canonical_compounds":len(val_can),
            "train_unique_canonical_compounds":len(train_can),
            "canonical_compound_overlap_count":len(train_can&val_can),
            "max_morgan_cosine_mean":float(sim.mean()),
            "fraction_max_morgan_cosine_ge_0p999":float(np.mean(sim>=.999)),
            "metrics":per_model,
        }
        fold_data.append(entry)
        total_train_pairs.append(len(tr))
        print("FOLD",fold,json.dumps(entry),flush=True)
    assert seen_groups==all_groups
    assert sum(component_counts)==len(raw)
    oof=oof_bootstrap(component_sums,component_counts)
    averages={
        name:{metric_name:float(np.mean([f["metrics"][name][metric_name] for f in fold_data]))
              for metric_name in ("pooled_rmse","component_macro_rmse")}
        for name in ("train_mean",)+MODES
    }
    output={
        "status":"BALM_TRAIN_ONLY_PROTEIN_AND_CANONICAL_COMPOUND_COLD_CV_NO_FROZEN_DEV_OR_TEST",
        "restriction":"Each fold holds out protein graph components and removes all training records sharing any isomeric canonical RDKit SMILES with heldout records.",
        "train_original_rows":len(raw),"unique_train_components":len(all_groups),
        "cv_folds":fold_data,"fold_mean_metrics":averages,
        "pooled_out_of_fold_across_nonoverlapping_4_folds":oof,
        "seconds":round(time.time()-t,2),
        "caveats":[
            "Applies only within original frozen TRAIN; does not inspect frozen dev or reserved test.",
            "This removes exact canonical molecules; related scaffolds and similar chemistry may remain.",
            "Protein groups are approximate trigram graph components, not verified sequence-identity homology.",
            "No assay source DOI/method in BALM; same exact inputs can have conflicting labels.",
            "Paired group bootstrap assumes group sampling despite group size variation.",
            "Model configurations were already explored on another dev split; use for stress testing, not unbiased proof.",
        ]
    }
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf8")
    print("FINAL",json.dumps({
        "fold_mean":averages,
        "oof":oof,
        "mean_retained_train_unique":float(np.mean(total_train_pairs)),
        "total_seconds":output["seconds"],
    }),flush=True)

if __name__=="__main__":main()


