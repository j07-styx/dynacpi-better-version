"""Prepare the public DeepDTA full Davis benchmark for isolated demo training.

This lane does NOT use quarantined legacy staging labels, BindingDB, old DEV or
protected test. Source order (68 ligands x 442 protein entries) is canonical
DeepDTA. Repeated protein sequences are grouped in train/val/test.
"""
import hashlib
import os
import importlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

HERE=Path(__file__).resolve().parent
OLD=Path(os.environ.get("DYNACPI_LEGACY_ROOT",str(Path.home()/"dynacpi_v4")))
OUT=HERE/"full_davis_demo_20261109"
SRC=OUT/"source"
DATA=OUT/"prepared_features.npz"
SPLIT=OUT/"split_manifest.json"
Y_SHA256="9855c4f234ec6dd18295b0e0cd6ee54214cafa908c6eb1d447b47d454fc0294a"
SPLIT_SEED=20261109
CLUSTER_MIN_KMER_COS=0.90

class SafeNumpyUnpickler(pickle.Unpickler):
    ALLOWED={
        ("numpy.core.multiarray","_reconstruct"),
        ("numpy","ndarray"),("numpy","dtype"),
    }
    def find_class(self,module,name):
        if (module,name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Rejected pickle global {module}.{name}")
        return getattr(importlib.import_module(module),name)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cluster_proteins(seqs):
    # 3mer TF-IDF cosine is a *PROXY* for sequence similarity, not a verified
    # kinase-family or sequence-identity clustering. Report this caveat.
    tf=TfidfVectorizer(analyzer="char",ngram_range=(3,3),dtype=np.float32,
                       lowercase=False,norm="l2")
    x=tf.fit_transform(seqs)
    sims=(x@x.T).toarray()
    n=len(seqs); parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i in range(n):
        for j in range(i):
            if seqs[i]==seqs[j] or sims[i,j]>=CLUSTER_MIN_KMER_COS:
                parent[root(j)]=root(i)
    ids=[root(i) for i in range(n)]
    d={g:i for i,g in enumerate(sorted(set(ids)))}
    return np.array([d[i] for i in ids],dtype=np.int32),sims

def main():
    if DATA.exists() or SPLIT.exists():
        raise RuntimeError("Existing full-DAVIS prepared artifacts: refusing overwrite")
    if sha(SRC/"Y").lower()!=Y_SHA256:
        raise RuntimeError("DeepDTA Y downloaded file hash changed")
    lig=json.loads((SRC/"ligands_can.txt").read_text(encoding="utf-8"))
    pro=json.loads((SRC/"proteins.txt").read_text(encoding="utf-8"))
    molecules=list(lig.values()); names=list(pro.keys()); seqs=list(pro.values())
    if (len(molecules),len(names))!=(68,442):
        raise RuntimeError("Unexpected DAVIS canonical reference dimensions")
    with (SRC/"Y").open("rb") as fh:
        matrix=np.asarray(SafeNumpyUnpickler(fh,encoding="latin1").load(),dtype=np.float64)
    if matrix.shape!=(68,442) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Unexpected DAVIS matrix shape/values: {matrix.shape}")
    if matrix.min()<=0 or matrix.max()>1e8:
        raise RuntimeError("DAVIS Kd nM range invalid")
    y=9-np.log10(matrix).T # 442 proteins x 68 chemical pKd labels
    censored=np.isclose(matrix,10000.0).T
    if y.min()<4 or y.max()>13:
        raise RuntimeError("pKd label convention wrong")

    fpgen=GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fingerprints=[]
    canonical=[]
    for smi in molecules:
        mol=Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError("Invalid canonical SMILES in public DAVIS")
        canonical.append(Chem.MolToSmiles(mol,isomericSmiles=True))
        arr=np.zeros((2048,),dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol),arr)
        fingerprints.append(arr)
    bits=np.stack(fingerprints).astype(np.float32)

    feats_by_seq={}
    cache=OLD/"data/esm2_t33_650M_layer33_w1000_o128_fp16"
    for seq in set(seqs):
        pid=hashlib.sha256(seq.encode()).hexdigest()
        path=cache/(pid+".pt")
        if not path.is_file():
            raise RuntimeError("Missing ESM protein embedding: "+pid)
        obj=torch.load(path,map_location="cpu",weights_only=True)
        if obj.get("sequence_hash")!=pid:
            raise RuntimeError("Mismatched ESM protein hash: "+pid)
        emb=obj["embeddings"].float()
        if emb.ndim!=2 or emb.shape[1]!=1280:
            raise RuntimeError("Unexpected ESM residue feature dimensions")
        feats_by_seq[seq]=emb.mean(0).numpy()
    prot_features=np.stack([feats_by_seq[s] for s in seqs]).astype(np.float32)

    groups,similarities=cluster_proteins(seqs)
    splitter=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SPLIT_SEED)
    trainval,test=next(splitter.split(prot_features,groups=groups))
    split2=GroupShuffleSplit(n_splits=1,test_size=.1765,random_state=SPLIT_SEED+1)
    ixtr,ixval=next(split2.split(prot_features[trainval],groups=groups[trainval]))
    train=np.sort(trainval[ixtr]);val=np.sort(trainval[ixval]);test=np.sort(test)
    assert len(set(train)&set(val))==len(set(train)&set(test))==len(set(val)&set(test))==0
    assert len(train)+len(val)+len(test)==442
    for a,b in ((train,val),(train,test),(val,test)):
        assert not set(groups[a])&set(groups[b])
    # Exact identity check independent from approximate groups
    for a,b in ((train,val),(train,test),(val,test)):
        assert not set(seqs[i] for i in a)&set(seqs[i] for i in b)

    manifest={
      "status":"EXPLORATORY_PUBLIC_DEEPDTA_DAVIS_PROTEIN_GROUPED_DEMO",
      "source_url":"https://github.com/hkmztrk/DeepDTA/tree/master/data/davis",
      "data_origin":"public DeepDTA DAVIS Y/ligands_can.txt/proteins.txt",
      "raw_Y_sha256":sha(SRC/"Y"),"ligands_sha256":sha(SRC/"ligands_can.txt"),
      "proteins_sha256":sha(SRC/"proteins.txt"),
      "label":"pKd=9-log10(Kd in nM)",
      "floor_flag":"Kd exactly 10000 nM, heuristic floor/censor marker, no row-level assay confirmation",
      "n_protein_reference_entries":442,
      "n_unique_sequences":len(set(seqs)),
      "n_compounds":68,"n_measured_pairs":int(np.isfinite(y).sum()),
      "fraction_floor":float(censored.mean()),
      "n_groups":int(len(np.unique(groups))),
      "sequence_grouping":"identical full sequences OR connected TF-IDF character 3-mer cosine>=0.90",
      "not_verified_kinase_family_disjoint":True,
      "max_3mer_cosine_train_test":float(similarities[np.ix_(train,test)].max()),
      "max_3mer_cosine_train_val":float(similarities[np.ix_(train,val)].max()),
      "seed":SPLIT_SEED,"indices":{"train":train.tolist(),"val":val.tolist(),"test":test.tolist()},
      "n_proteins":{"train":len(train),"val":len(val),"test":len(test)},
      "n_pairs":{"train":len(train)*68,"val":len(val)*68,"test":len(test)*68},
      "caveats":[
        "Public Davis benchmark labels and provisional pKd 5 floor flags are not independently assay-reviewed.",
        "This is a NEW benchmark lane separate from quarantined staging and blocked BindingDB splits.",
        "Protein groups from 3-mer TF-IDF cosine only approximate sequence homology, not curated kinase families.",
        "All 68 compounds are present in each split: this measures cold-protein, NOT double-cold.",
        "Do not compare to published PerceiverCPI numbers absent an identical matched training/evaluation protocol.",
        "Censor-aware loss is an experimental assumption, not ground-truth measurement annotation."
      ]}
    OUT.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(DATA,pfeat=prot_features,bits=bits,y=y.astype(np.float32),
                        censored=censored.astype(np.bool_),
                        train=train,val=val,test=test,groups=groups,
                        names=np.array(names),smiles=np.array(canonical))
    manifest["prepared_features_sha256"]=sha(DATA)
    SPLIT.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"status":"READY_FULL_DAVIS","proteins":442,"unique_proteins":len(set(seqs)),
                      "compounds":68,"pairs":int(y.size),
                      "floor_fraction":float(censored.mean()),"groups":int(len(np.unique(groups))),
                      "train":len(train),"val":len(val),"test":len(test),
                      "max_train_test_sim":manifest["max_3mer_cosine_train_test"],
                      "protein_features":list(prot_features.shape),
                      "bits":list(bits.shape),
                      "min_pKd":float(y.min()),"max_pKd":float(y.max()),
                      "manifest":str(SPLIT)},indent=2),flush=True)

if __name__=="__main__":
    main()
