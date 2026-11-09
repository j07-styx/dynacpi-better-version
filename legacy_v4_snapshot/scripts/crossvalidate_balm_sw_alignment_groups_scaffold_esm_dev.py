"""Nested CPI experiment under SW-alignment protein groups + strict Murcko cold.

The graph uses TRAIN protein sequences only. No frozen development or
prospective test records, predictions or labels are used.
"""
import argparse,hashlib,json,time
from pathlib import Path

import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold

from dynacpi.balm_public_data_dev import ROOT,get_train_only
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import (
    scaffold_key,scaffold_filter,select_beta)
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,metric)
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import (
    cache_feature_map,fit_predict as ridge_predict,bootstrap)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import (
    fit_predict as joint_predict)

ALLOWED=("identity30_cover80","identity40_cover80")
MODELS=("train_mean","ridge_kmer","ridge_kmer_plus_esm",
        "joint_kmer","joint_kmer_esm","nested_kmer","nested_kmer_esm")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--group-rule",required=True,choices=ALLOWED)
    args=ap.parse_args()
    rule=args.group_rule
    output=ROOT/"research"/("balm_esm8m_nested_sw_"+rule+"_murcko_scaffold_cold_traincv_dev.json")
    if output.exists():raise FileExistsError(output)
    start=time.time()
    audit=json.loads((ROOT/"research/balm_train_only_smith_waterman_protein_group_audit_20261108.json").read_text(encoding="utf8"))
    sourcegroups=audit["sequence_sha256_to_alignment_groups"][rule]
    embs,feature_manifest=cache_feature_map()
    original=get_train_only()
    assert len(original)==15538
    seq2sha={r["protein"]:hashlib.sha256(r["protein"].encode()).hexdigest() for r in original}
    assert set(seq2sha.values())==set(sourcegroups)==set(embs)
    records=[dict(r,group=sourcegroups[seq2sha[r["protein"]]]) for r in original]
    groups=np.asarray([r["group"] for r in records])
    count_groups=len(set(groups))
    assert count_groups==audit["rules"][rule]["connected_components"]
    scaffold={s:scaffold_key(s) for s in set(r["smiles"] for r in records)}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={};hashes={};chem_desc={};prot_desc={}
    component_sq={name:[] for name in MODELS};component_counts=[]
    fold_reports=[];seen=set()
    for fold,(itr,iva) in enumerate(GroupKFold(4).split(groups,groups=groups),1):
        va=[records[int(i)] for i in iva]
        tr=collapse_training(scaffold_filter([records[int(i)] for i in itr],va,scaffold))
        held_groups=set(r["group"] for r in va)
        assert not held_groups&set(r["group"] for r in tr)
        assert not held_groups&seen
        assert not {scaffold[r["smiles"]] for r in tr}&{scaffold[r["smiles"]] for r in va}
        assert len(tr)>1000
        seen.update(held_groups)

        beta,innerfolds,inneravg=select_beta(tr,scaffold,gen,cache,chem_desc,prot_desc)
        traditional=joint_predict(tr,va,gen,cache,chem_desc,prot_desc)
        evolved=ridge_predict(tr,va,gen,cache,embs,hashes)
        np.testing.assert_allclose(traditional["ridge"],evolved["chemical_plus_kmer"],
                                   rtol=1e-5,atol=1e-4)
        y=np.asarray([r["label"] for r in va],dtype=np.float64)
        va_groups=np.asarray([r["group"] for r in va])
        mean=float(np.mean([r["label"] for r in tr]))
        joint_kmer=.75*traditional["ridge"]+.25*traditional["boost_joint"]
        joint_esm=.75*evolved["chemical_kmer_esm8m"]+.25*traditional["boost_joint"]
        outputs={
            "train_mean":np.full_like(y,mean),
            "ridge_kmer":traditional["ridge"],
            "ridge_kmer_plus_esm":evolved["chemical_kmer_esm8m"],
            "joint_kmer":joint_kmer,
            "joint_kmer_esm":joint_esm,
            "nested_kmer":mean+beta*(joint_kmer-mean),
            "nested_kmer_esm":mean+beta*(joint_esm-mean),
        }
        for held_group in sorted(held_groups):
            mask=(va_groups==held_group)
            component_counts.append(int(mask.sum()))
            for model in MODELS:
                component_sq[model].append(float(np.sum(np.square(y[mask]-outputs[model][mask]))))
        metrics={name:metric(y,outputs[name],va_groups) for name in MODELS}
        item={
            "outer_fold":fold,
            "fit_unique_pairs":len(tr),
            "heldout_pairs":len(va),
            "heldout_alignment_groups":len(held_groups),
            "murcko_scaffold_overlap":0,
            "inherited_beta_selected_inside_this_outer_fold":beta,
            "inner_folds":innerfolds,
            "inner_mean_metrics":inneravg,
            "outer_metrics":metrics,
        }
        fold_reports.append(item)
        print("OUTER",fold,json.dumps({
            "n_train":len(tr),"n_val":len(va),"heldout_groups":len(held_groups),
            "beta":beta,"metrics":metrics,
        }),flush=True)
    assert seen==set(groups)
    assert sum(component_counts)==len(original) and len(component_counts)==count_groups
    outof=bootstrap(component_sq,component_counts,replicates=5000,seed=20261108,
                    references=("train_mean","joint_kmer","nested_kmer"))
    final={
        "status":"EXPLORATORY_BALM_TRAIN_ONLY_ALIGNMENT_GROUP_AND_MURCKO_SCAFFOLD_COLD_NESTED_CV_NO_DEV_OR_TEST",
        "protein_group_rule":rule,
        "alignment_rule":audit["rules"][rule],
        "protein_alignment_input":"all 639 frozen TRAIN sequences, exhaustive Smith-Waterman + BLOSUM62",
        "original_frozen_train_records":len(original),
        "heldout_alignment_group_count":count_groups,
        "heldout_murcko_exclusion":"RDKit Bemis-Murcko scaffold identity; acyclic compounds in one common group, applied outer and inner",
        "nested_beta_candidates":[0.,.25,.5,.75,1.],
        "nested_selector":"Inner GroupKFold(3), same alignment group and strict Murcko exclusion, classical joint booster selects beta; held beta transferred unchanged to ESM branch",
        "frozen_esm_feature_cache_sha256":feature_manifest["cache_sha256"],
        "models":MODELS,
        "outer_folds":fold_reports,
        "out_of_fold":outof,
        "seconds":round(time.time()-start,1),
        "caveats":[
            "Protein groups are local Smith-Waterman alignment-identity/coverage proxies, not verified biological homology.",
            "Shorter protein coverage can accept a shared domain within longer proteins.",
            "These groupings created using TRAIN sequences only, not new official benchmark split.",
            "No frozen inner-development or reserved prospective test outcomes or sequences used.",
            "BALM original assay provenance and repeated-input label disagreement remain unresolved.",
            "Architecture and selection choices were developed using previously inspected development metrics; results exploratory.",
            "ESM2 features are frozen mean representations from first 1022 AA (85 TRAIN proteins truncated).",
        ]
    }
    output.write_text(json.dumps(final,indent=2)+"\n",encoding="utf8")
    pairs=outof["bootstrap"]["comparisons"]
    print("FINAL",json.dumps({
        "rule":rule,"betas":[f["inherited_beta_selected_inside_this_outer_fold"] for f in fold_reports],
        "pooled":outof["pooled_rmse"],"macro":outof["component_macro_rmse"],
        "nested_esm_vs_mean":pairs["nested_kmer_esm_minus_train_mean"],
        "nested_esm_vs_classic":pairs["nested_kmer_esm_minus_nested_kmer"],
        "seconds":final["seconds"],
    }),flush=True)

if __name__=="__main__":main()
