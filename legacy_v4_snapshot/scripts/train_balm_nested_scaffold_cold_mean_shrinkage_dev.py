"""Nested, TRAIN-only protein group and Murcko scaffold cold test for mean-shrunk CPI.

Outer 4-fold protein group holdout; remove all Bemis-Murcko scaffold groups
from fitting for every fold. Inside each outer train, use distinct 3-fold
protein+Murcko-scaffold-cold splits to choose a shrinkage coefficient. The
preexisting frozen inner-dev and prospective test never supply labels.
"""
import json,time
import numpy as np
from sklearn.model_selection import GroupKFold
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from dynacpi.balm_public_data_dev import get_train_only,ROOT
from scripts.crossvalidate_balm_dual_cold_with_mean_bootstrap_dev import (
    canonical_smiles,oof_bootstrap
)
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,metric
)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import fit_predict

OUT=ROOT/"research/balm_bindingdb_nested_scaffold_cold_mean_shrinkage_traincv_dev.json"
BETAS=(0.,.25,.5,.75,1.)

def scaffold_filter(train_rows,validation_rows,can):
    """Remove all scaffold groups seen in validation, including acyclic group."""
    withheld={can[r["smiles"]] for r in validation_rows}
    result=[r for r in train_rows if can[r["smiles"]] not in withheld]
    assert not {can[r["smiles"]] for r in result}&withheld
    return result

def scaffold_key(smiles):
    """Strict Bemis-Murcko core. All molecules without ring cores share one key."""
    molecule=Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES")
    core=MurckoScaffold.GetScaffoldForMol(molecule)
    key=Chem.MolToSmiles(core,canonical=True,isomericSmiles=True)
    return key if key else "ACYCLIC_SINGLE_GROUP"


def predict_components(fit,va,gen,cache_chem,cache_desc,cache_prot):
    preds=fit_predict(fit,va,gen,cache_chem,cache_desc,cache_prot)
    ridge=preds["ridge"]
    combo=preds["ridge_interaction_blend_0.25"]
    # IMPORTANT: replace explicit interaction blend with ordinary joint
    # chemical/protein booster selected by prior matched controls.
    combo=.75*ridge+.25*preds["boost_joint"]
    mu=float(np.mean([r["label"] for r in fit]))
    yy=np.asarray([r["label"] for r in va],dtype=np.float64)
    return mu,yy,ridge,combo

def select_beta(rows,can,gen,chem_cache,desc_cache,prot_cache):
    """Inner dual-cold heldout labels used ONLY for choosing outer model beta."""
    group=np.asarray([r["group"] for r in rows])
    subfold=[]
    for i,(itra,iva) in enumerate(GroupKFold(n_splits=3).split(group,groups=group),1):
        hold=[rows[int(i)] for i in iva]
        fit=collapse_training(scaffold_filter([rows[int(i)] for i in itra],hold,can))
        assert len(fit)>300,(i,len(fit))
        assert {r["group"] for r in fit}.isdisjoint({r["group"] for r in hold})
        mu,y,_,ensemble=predict_components(fit,hold,gen,chem_cache,desc_cache,prot_cache)
        valid_groups=[r["group"] for r in hold]
        scores={str(beta):metric(y,mu+beta*(ensemble-mu),valid_groups)
                for beta in BETAS}
        subfold.append({"fold":i,"fit_rows":len(fit),"holdout_rows":len(hold),
                        "holdout_components":len(set(valid_groups)),
                        "scores":scores})
    mean={
        str(beta):{
          k:float(np.mean([f["scores"][str(beta)][k] for f in subfold]))
          for k in ("pooled_rmse","component_macro_rmse")
        } for beta in BETAS
    }
    baseline=mean["0.0"]
    eligible=[
        beta for beta in BETAS
        if mean[str(beta)]["pooled_rmse"]<=baseline["pooled_rmse"] and
           mean[str(beta)]["component_macro_rmse"]<=baseline["component_macro_rmse"]
    ]
    score=lambda b:.5*(mean[str(b)]["pooled_rmse"]/baseline["pooled_rmse"]+
                       mean[str(b)]["component_macro_rmse"]/baseline["component_macro_rmse"])
    selected=min(eligible,key=lambda b:(score(b),b)) if eligible else 0.
    return selected,subfold,mean

def main():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.time()
    raw=get_train_only()
    assert len(raw)==15538
    groups=np.array([r["group"] for r in raw])
    can={sm:scaffold_key(sm) for sm in sorted(set(r["smiles"] for r in raw))}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache_chem={};cache_desc={};cache_prot={}
    folds=[];component_counts=[]
    names=("train_mean","ridge","ridge_plus_joint_boost","nested_selected_shrinkage")
    component_sums={name:[] for name in names}
    seen=set()
    for idx,(itr,iva) in enumerate(GroupKFold(n_splits=4).split(groups,groups=groups),1):
        heldout=[raw[int(i)] for i in iva]
        subtrain=scaffold_filter([raw[int(i)] for i in itr],heldout,can)
        tr=collapse_training(subtrain)
        unique_groups=set(r["group"] for r in heldout)
        assert len(tr)>=3000
        assert {r["group"] for r in tr}.isdisjoint(unique_groups)
        assert not seen&unique_groups
        seen.update(unique_groups)
        selected,innerfolds,inneravg=select_beta(
            tr,can,gen,cache_chem,cache_desc,cache_prot)
        mu,y,ridge,ensemble=predict_components(
            tr,heldout,gen,cache_chem,cache_desc,cache_prot)
        out={
            "train_mean":np.full_like(y,mu),
            "ridge":ridge,
            "ridge_plus_joint_boost":ensemble,
            "nested_selected_shrinkage":mu+selected*(ensemble-mu)
        }
        heldout_groups=np.asarray([r["group"] for r in heldout])
        for protein_group in sorted(unique_groups):
            mask=(heldout_groups==protein_group)
            component_counts.append(int(mask.sum()))
            for name in names:
                component_sums[name].append(float(np.sum(np.square(y[mask]-out[name][mask]))))
        record={
            "outer_fold":idx,"outer_fit_unique_pairs":len(tr),
            "heldout_rows":len(heldout),"heldout_components":len(unique_groups),
            "heldout_murcko_scaffold_overlap_count":len(
                {can[r["smiles"]] for r in tr}&
                {can[r["smiles"]] for r in heldout}),
            "inner_dual_cold_folds":innerfolds,
            "inner_mean_metrics":inneravg,
            "inner_selected_beta":selected,
            "outer_metrics":{name:metric(y,out[name],heldout_groups)
                             for name in names},
        }
        assert record["heldout_murcko_scaffold_overlap_count"]==0
        folds.append(record)
        print("OUTER_FOLD",idx,json.dumps({
            "n_outer_train":len(tr),"n_val":len(heldout),
            "selected_beta":selected,
            "inner_mean":inneravg,
            "outer_metrics":record["outer_metrics"]
        }),flush=True)
    assert seen==set(groups) and sum(component_counts)==len(raw)
    results=oof_bootstrap(component_sums,component_counts,replicates=5000,seed=20261108)
    mean_fold={
        name:{
            k:float(np.mean([f["outer_metrics"][name][k] for f in folds]))
            for k in ("pooled_rmse","component_macro_rmse")
        } for name in names
    }
    output={
        "status":"TRAIN_ONLY_NESTED_PROTEIN_AND_STRICT_MURCKO_SCAFFOLD_COLD_CV_NO_DEV_OR_TEST",
        "architecture":"75pct Ridge +25pct joint molecular/protein gradient boosting, then train-only nested selected shrinkage to fold train mean",
        "scaffold_policy":"RDKit Bemis-Murcko scaffold; one shared ACYCLIC_SINGLE_GROUP for empty cores; strict exclusion in inner and outer folds",
        "predeclared_betas":BETAS,
        "selection":"GroupKFold(3) within each outer fit, both macro and pooled <= inner mean, min balanced relative pooled/macro",
        "original_train_rows":len(raw),"outer_fold_count":len(folds),
        "murcko_scaffold_group_count_in_train":len(set(can.values())),
        "outer_folds":folds,"mean_outer_fold_metrics":mean_fold,
        "outer_oof":results,"seconds":round(time.time()-start,2),
        "caveats":[
            "Inner/outer group splits block approximate protein trigram graph components, not verified homology.",
            "All training rows sharing heldout Bemis-Murcko scaffold are removed, including a common acyclic group; other analogs may remain.",
            "Train cohorts contain incomplete assay provenance and conflicting same-input labels.",
            "Both architecture and choice of candidate betas follow previous development analysis; exploratory.",
            "Neither frozen inner-development nor prospective test labels or predictions accessed.",
        ],
    }
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print("FINAL",json.dumps({
        "nested_betas":[f["inner_selected_beta"] for f in folds],
        "fold_mean":mean_fold,
        "pooled":results["pooled_rmse"],
        "macro":results["macro_component_rmse"],
        "paired":results["paired_bootstrap"]["controls"],
        "seconds":output["seconds"]
    }),flush=True)

if __name__=="__main__": main()




