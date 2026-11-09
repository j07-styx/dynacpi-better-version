"""Compare robust aggregation and discordant-input filtering on TRAIN-only BALM.

Under Smith-Waterman protein group and strict Murcko scaffold cold folds, no
original frozen dev or reserved test labels, sequences or predictions used.
"""
import argparse,hashlib,json,time
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold

from dynacpi.balm_public_data_dev import ROOT,get_train_only
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import (
    scaffold_key,scaffold_filter)
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,metric)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import (
    fit_predict as joint_predict)
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import (
    cache_feature_map,fit_predict as ridge_predict,bootstrap)

RULES=("identity40_cover80","identity30_cover80")
MODES=("original_mean","original_ensemble","median_mean","median_ensemble",
       "exclude_mean","exclude_ge1_ensemble")

def grouped(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["protein"],r["smiles"])].append(r)
    return groups

def median_collapse(rows):
    data=[]
    for records in grouped(rows).values():
        r=dict(records[0])
        r["label"]=float(np.median([item["label"] for item in records]))
        data.append(r)
    return data

def exclude_conflicting_exact_inputs(rows,canonical_cache,threshold=1.):
    """Only training labels select exclusions. Stereoisomers remain distinct."""
    for r in rows:
        assert r["smiles"] in canonical_cache
    grouped_canon=defaultdict(list)
    for r in rows:
        key=(r["protein"],canonical_cache[r["smiles"]])
        grouped_canon[key].append(r["label"])
    excluded={
        key for key,labels in grouped_canon.items()
        if len(labels)>1 and np.ptp(labels)>=threshold
    }
    filtered=[
        r for r in rows
        if (r["protein"],canonical_cache[r["smiles"]]) not in excluded
    ]
    assert len(filtered)+sum(1 for r in rows
        if (r["protein"],canonical_cache[r["smiles"]]) in excluded)==len(rows)
    return filtered,{
        "discarded_canonical_exact_input_groups":len(excluded),
        "discarded_original_training_rows":len(rows)-len(filtered),
        "retained_training_rows":len(filtered),
        "conflict_log_affinity_span_threshold":threshold,
    }

def predict_esm_joint(training,validation,gen,fp,chemcache,protc,hcache,feature):
    regr=ridge_predict(training,validation,gen,fp,feature,hcache)
    boost=joint_predict(training,validation,gen,fp,chemcache,protc)
    np.testing.assert_allclose(regr["chemical_plus_kmer"],boost["ridge"],
                               atol=1e-4,rtol=1e-5)
    return .75*regr["chemical_kmer_esm8m"]+.25*boost["boost_joint"]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--group-rule",required=True,choices=RULES)
    args=parser.parse_args()
    rule=args.group_rule
    OUT=ROOT/"research"/("balm_sw_"+rule+"_affinity_disagreement_training_ablation_dev.json")
    if OUT.exists():raise FileExistsError(OUT)
    start=time.time()
    audit=json.loads((ROOT/"research/balm_train_only_smith_waterman_protein_group_audit_20261108.json").read_text())
    previous=json.loads((ROOT/"research"/("balm_esm8m_nested_sw_"+rule+"_murcko_scaffold_cold_traincv_dev.json")).read_text())
    features,manifest=cache_feature_map()
    raw=get_train_only()
    sh={s:hashlib.sha256(s.encode()).hexdigest() for s in set(r["protein"] for r in raw)}
    mapping=audit["sequence_sha256_to_alignment_groups"][rule]
    rows=[dict(r,group=mapping[sh[r["protein"]]]) for r in raw]
    groups=np.asarray([r["group"] for r in rows])
    scaffold={s:scaffold_key(s) for s in set(r["smiles"] for r in rows)}
    canonical={}
    for s in scaffold:
        mol=Chem.MolFromSmiles(s)
        assert mol is not None
        canonical[s]=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fp={};ccache={};pcache={};hcache={}
    sums={name:[] for name in MODES};sizes=[];folds=[]
    for fold,(itr,iva) in enumerate(GroupKFold(4).split(groups,groups=groups),1):
        held=[rows[int(i)] for i in iva]
        base=scaffold_filter([rows[int(i)] for i in itr],held,scaffold)
        assert not {r["group"] for r in base}&{r["group"] for r in held}
        assert not {scaffold[r["smiles"]] for r in base}&{scaffold[r["smiles"]] for r in held}
        fit_original=collapse_training(base)
        fit_median=median_collapse(base)
        clean,removed=exclude_conflicting_exact_inputs(base,canonical,threshold=1.)
        fit_filtered=collapse_training(clean)
        assert len(fit_original)==len(fit_median)
        assert len(fit_filtered)<=len(fit_original)
        assert len(fit_filtered)>1000
        beta=float(previous["outer_folds"][fold-1]["inherited_beta_selected_inside_this_outer_fold"])
        y=np.asarray([r["label"] for r in held],dtype=np.float64)
        held_groups=np.asarray([r["group"] for r in held])
        output={}
        for prefix,train in (
            ("original",fit_original),("median",fit_median),("exclude",fit_filtered)
        ):
            mean=float(np.mean([r["label"] for r in train]))
            pred=predict_esm_joint(train,held,gen,fp,ccache,pcache,hcache,features)
            output[prefix+"_mean"]=np.full(len(y),mean)
            output[{"original":"original_ensemble","median":"median_ensemble","exclude":"exclude_ge1_ensemble"}[prefix]]=mean+beta*(pred-mean)
        for group in sorted(set(held_groups)):
            mask=(held_groups==group)
            sizes.append(int(mask.sum()))
            for name in MODES:
                sums[name].append(float(np.sum(np.square(y[mask]-output[name][mask]))))
        metrics={name:metric(y,v,held_groups) for name,v in output.items()}
        expected=previous["outer_folds"][fold-1]["outer_metrics"]["nested_kmer_esm"]
        for m in ("pooled_rmse","component_macro_rmse"):
            assert abs(metrics["original_ensemble"][m]-expected[m])<3e-5
        item={
            "fold":fold,"training_original_pairs":len(base),
            "training_unique_original":len(fit_original),
            "training_unique_median":len(fit_median),
            "training_unique_after_conflict_exclusion":len(fit_filtered),
            "removed_groups":removed,
            "heldout_original_pairs":len(held),
            "heldout_alignment_groups":len(set(held_groups)),
            "scaffold_overlap":0,
            "inherited_beta":beta,
            "metrics":metrics,
        }
        folds.append(item)
        print("FOLD",fold,json.dumps(item),flush=True)
    assert sum(sizes)==len(raw) and len(sizes)==audit["rules"][rule]["connected_components"]
    stats=bootstrap(sums,sizes,replicates=5000,seed=20261108,
                    references=("original_ensemble","original_mean"))
    result={
        "status":"EXPLORATORY_BALM_TRAIN_ONLY_SW_PROTEIN_AND_SCAFFOLD_COLD_LABEL_NOISE_ABLATION",
        "alignment_group_rule":rule,
        "cohort":"frozen BALM TRAIN only",
        "affinity_disagreement_filter":"remove entire exact protein+canonical-isomeric-SMILES groups from each outer training fold when repeated labels have span >=1.0 p-affinity units; no heldout labels used",
        "median_aggregation":"median on repeated exact protein+raw SMILES training pairs; no exclusion",
        "selection":"thresholds and treatments fixed after independent TRAIN-only label distribution audit, exploratory",
        "esm_feature_cache_sha256":manifest["cache_sha256"],
        "folds":folds,
        "out_of_fold":stats,
        "seconds":round(time.time()-start,1),
        "warnings":[
            "Conflicting repeated labels may represent different legitimate assay conditions; no claim they are errors.",
            "Original source assay DOI, technique, experimental condition and construct absent.",
            "No data change applied to reserved test; all training exclusions are within fit fold only.",
            "Original inner-development or reserved prospective test inputs, labels and predictions not accessed.",
            "Protein grouping is an alignment-derived homology PROXY, not verified homology.",
            "Data filtering and median are motivated by explored TRAIN labels; model benchmarking still exploratory.",
        ],
    }
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print("FINAL",json.dumps({
        "pooled":stats["pooled_rmse"],
        "macro":stats["component_macro_rmse"],
        "median_vs_original":stats["bootstrap"]["comparisons"]["median_ensemble_minus_original_ensemble"],
        "exclude_vs_original":stats["bootstrap"]["comparisons"]["exclude_ge1_ensemble_minus_original_ensemble"],
        "seconds":result["seconds"],
    }),flush=True)

if __name__=="__main__":main()
