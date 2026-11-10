"""Pre-register label-BLIND DAVIS partitions for cross-split robustness.

Produces read-only split index manifests and TRAIN/VALIDATION CSVs for the official
PerceiverCPI code. Does NOT export any new test labels or evaluate models.
Evaluation variants and seeds are fixed in this source and public protocol,
not chosen using held-out performance.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

HERE=Path(__file__).resolve().parent
SOURCE=HERE/"full_davis_demo_20261109"
PREP=SOURCE/"prepared_features.npz"
OUT=HERE/"generalization_suite_20261110"
MODEL_LABELS="pKd = 9 - log10(Kd [nM])"
SPECS=[
    ("protein_strict_20261113","protein",0.35,20261113),
    ("protein_strict_20261114","protein",0.35,20261114),
    ("compound_scaffold_20261115","compound",None,20261115),
    ("double_cold_20261116","double",0.35,20261116)
]
TRAINING_SEEDS=[2931,2932]
NO_OUTCOME_SELECTION=True

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sequence_components(seqs,threshold):
    tf=TfidfVectorizer(analyzer="char",ngram_range=(3,3),dtype=np.float32,lowercase=False,norm="l2")
    x=tf.fit_transform(seqs);sims=(x@x.T).toarray()
    parent=list(range(len(seqs)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(seqs)):
        for j in range(i):
            if seqs[i]==seqs[j] or sims[i,j]>=threshold:
                parent[root(j)]=root(i)
    ids=[root(i) for i in range(len(seqs))]
    mapping={g:k for k,g in enumerate(sorted(set(ids)))}
    return np.array([mapping[g] for g in ids],dtype=int),sims

def scaffold_clusters(smiles):
    scaffold_names=[]
    for smi in smiles:
        mol=Chem.MolFromSmiles(str(smi))
        if mol is None:raise ValueError("Invalid ligand SMILES")
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffold_names.append(scaffold or "NO_RING")
    mapping={s:i for i,s in enumerate(sorted(set(scaffold_names)))}
    return np.asarray([mapping[x] for x in scaffold_names],dtype=int),scaffold_names

def groupsplit(groups,seed):
    idx=np.arange(len(groups))
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.18,random_state=seed).split(idx,groups=groups))
    tr_idx,validx=next(GroupShuffleSplit(n_splits=1,test_size=.18,random_state=seed+1).split(a,groups=groups[a]))
    tr=np.sort(a[tr_idx]);v=np.sort(a[validx]);t=np.sort(b)
    assert all(len(x)>0 for x in [tr,v,t])
    assert len(np.union1d(np.union1d(tr,v),t))==len(groups)
    assert not set(groups[tr])&set(groups[v])
    assert not set(groups[tr])&set(groups[t])
    assert not set(groups[v])&set(groups[t])
    return tr,v,t

def emit_train_and_val(dest, train_proteins,val_proteins,train_compounds,val_compounds,seqs,smiles,y):
    def out_csv(filename,ppi,cci):
        path=dest/filename
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["smiles","sequence","label"])
            for i in ppi:
                for j in cci:
                    w.writerow([str(smiles[int(j)]),seqs[int(i)],repr(float(y[int(i),int(j)]))])
        return path
    train=out_csv("train.csv",train_proteins,train_compounds)
    val=out_csv("validation.csv",val_proteins,val_compounds)
    # Upstream train.py demands a test file; must be byte-identical validation.
    proxy=dest/"validation_proxy_test.csv"
    proxy.write_bytes(val.read_bytes())
    return {x.name:sha(x) for x in [train,val,proxy]}

def main():
    if OUT.exists():raise RuntimeError("Refusing to overwrite pre-registered suite")
    source_manifest=json.loads((SOURCE/"split_manifest.json").read_text(encoding="utf-8"))
    if sha(PREP)!=source_manifest["prepared_features_sha256"]:
        raise RuntimeError("Original source feature checksum does not match")
    if sha(SOURCE/"source"/"Y")!=source_manifest["raw_Y_sha256"]:
        raise RuntimeError("Original DAVIS data source checksum does not match")
    sequences=list(json.loads((SOURCE/"source"/"proteins.txt").read_text(encoding="utf-8")).values())
    with np.load(PREP,allow_pickle=False) as f:
        y=f["y"].astype(np.float64)
        smiles=f["smiles"].astype(str)
        group_old=f["groups"].astype(int)
    if y.shape!=(442,68) or len(sequences)!=442:raise RuntimeError("Unexpected DAVIS dimensions")
    strict_group,sims=sequence_components(sequences,.35)
    scaffold_group,scaffolds=scaffold_clusters(smiles)
    split_index=[]
    allp=np.arange(442)
    allc=np.arange(68)
    # Create a parent directory before emitting test labels; none are ever saved.
    OUT.mkdir(parents=True,exist_ok=False)
    for name,variant,threshold,seed in SPECS:
        dest=OUT/name
        dest.mkdir()
        if variant in ("protein","double"):
            tp,vp,sp=groupsplit(strict_group,seed)
        else:
            tp=vp=sp=allp
        if variant in ("compound","double"):
            tc,vc,sc=groupsplit(scaffold_group,seed+100)
        else:
            tc=vc=sc=allc
        if variant=="protein":
            assert not set(strict_group[tp])&set(strict_group[sp])
        if variant=="compound":
            assert not set(scaffold_group[tc])&set(scaffold_group[sc])
        if variant=="double":
            assert not set(strict_group[tp])&set(strict_group[sp])
            assert not set(scaffold_group[tc])&set(scaffold_group[sc])
        # Train and validation labels ONLY. Test indices are frozen but not exported to CSV.
        checks=emit_train_and_val(dest,tp,vp,tc,vc,sequences,smiles,y)
        o={
          "scenario":name,"category":variant,"seed":seed,
          "n_3mer_sequence_groups_at_0_35":int(len(set(strict_group))),
          "n_murcko_scaffold_groups":int(len(set(scaffold_group))),
          "training_seeds":TRAINING_SEEDS,
          "perceiver_epochs":30,
          "fixed_v5_architecture":"centered_esm_censored",
          "metric_priority":["target_macro_rmse","pooled_rmse","non_floor_rmse"],
          "protein_similarity":"TF-IDF 3-mer connected groups at cosine >=0.35; NOT curated families",
          "compound_grouping":"RDKit Bemis-Murcko scaffold groups",
          "identical_protein_group_ids_disjoint":True if variant in ("protein","double") else None,
          "identical_murcko_scaffolds_disjoint":True if variant in ("compound","double") else None,
          "floor_labels_fraction_not_used_for_split":True,
          "test_labels_not_exported":True,
          "selection_locked_before_test":True,
          "label_units":MODEL_LABELS,
          "source_Y_sha256":sha(SOURCE/"source"/"Y"),
          "source_features_sha256":sha(PREP),
          "original_v5_split_manifest_sha256":sha(SOURCE/"split_manifest.json"),
          "indices":{
            "protein_train":list(map(int,tp)),"protein_val":list(map(int,vp)),"protein_test":list(map(int,sp)),
            "compound_train":list(map(int,tc)),"compound_val":list(map(int,vc)),"compound_test":list(map(int,sc))
          },
          "train_pairs":int(len(tp)*len(tc)),
          "validation_pairs":int(len(vp)*len(vc)),
          "test_pairs_not_exported":int(len(sp)*len(sc)),
          "input_csv_sha256":checks,
          "execution_status":"PREDECLARED_SPLIT_NO_MODELS_EVALUATED",
          "caveat":"This suite reuses the publicly explored DAVIS benchmark and is not an external prospective test."
        }
        path=dest/"SPLIT_LOCKED.json"
        path.write_text(json.dumps(o,indent=2),encoding="utf-8")
        split_index.append({k:o[k] for k in ["scenario","category","seed","train_pairs","validation_pairs","test_pairs_not_exported"]})
        split_index[-1]["split_manifest_sha256"]=sha(path)
    summary={
      "created_utc_via_local_run_not_used_to_select":True,
      "run_label":"v1 preregistered expanded DAVIS robustness evaluation",
      "scenarios":split_index,
      "original_davis_source_manifest_sha256":sha(SOURCE/"split_manifest.json"),
      "n_scaffold_groups":len(set(scaffold_group)),
      "n_strict_sequence_groups":len(set(strict_group)),
      "max_old_p0_90_between_target_coherence_not_family_qualified":True,
      "holdout_label_exports":0,
      "public_manifest_no_outcome_information":True,
      "do_not_select_favorable_results":True,
      "summary":"Three new evaluation types: strict protein-group, scaffold cold, double cold. Each split is trained independently, scored only after validation model freeze. Unlike original 0.9 threshold, 0.35 threshold merges more nonidentical related sequences. Model selection fixed; published PerceiverCPI fold protocol still distinct."
    }
    (OUT/"SUITE_LOCKED.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
