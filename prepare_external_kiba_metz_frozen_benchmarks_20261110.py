"""Build PUBLIC, immutable external KIBA/Metz ML benchmark splits.

No model tuning or test evaluation. ESM-2 features are read from exact
sequence-hash cached public protein embeddings on the authorized PC.
KIBA labels are original DeepDTA COMPOSITE KIBA scores, NOT pKd.
Metz labels are PADME filtered numeric affinity values, NOT recoded KIBA.
Source files, dataset variants and deterministic splits are SHA-locked.
New train+validation CSVs ONLY; all held-out labels stay sealed for testing.

External test protocol intended for comparing DynaCPI against the original
PerceiverCPI on identical pair rows. No claim to duplicate their paper folds.
"""
from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from prepare_full_davis_v5 import SafeNumpyUnpickler

BASE=Path(__file__).resolve().parent/"external_benchmarks_20261110"
CACHE=Path.home()/"dynacpi_v4"/"data"/"esm2_t33_650M_layer33_w1000_o128_fp16"
SOURCES={
 "kiba":{"root":BASE/"DeepDTA_public"/"data"/"kiba",
        "repository":BASE/"DeepDTA_public",
        "src_sha":"a546a8433a6822e958f36171c4356ad6f414d623",
        "labels":"KIBA composite score, untransformed DeepDTA Y (NO pKd; high=stronger association)",
        "expected_entries":(229,2111,118254)},
 "metz":{"root":BASE/"PADME_public"/"metz_data",
        "repository":BASE/"PADME_public",
        "src_sha":"e01c592cc06c4de04b3ed6db35da5af5ff7f863f",
        "labels":"PADME Metz numeric reported affinities (not transformed to KIBA or DAVIS pKd)",
        "expected_entries":(170,1423,35259)}
}
SPLITS={
  "protein_cold":20261117,
  "compound_cold":20261118,
  "double_cold":20261119
}
PROTEIN_GROUP_COSINE=0.35
TRAINING_SEEDS=[2931,2932]
OUT=BASE/"locked_external_protocol_v1"

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def read_kiba(root):
 molecules=list(json.loads((root/"ligands_can.txt").read_text(encoding="utf-8")).values())
 sequences=list(json.loads((root/"proteins.txt").read_text(encoding="utf-8")).values())
 with (root/"Y").open("rb") as fh:
  labels=np.asarray(SafeNumpyUnpickler(fh,encoding="latin1").load(),dtype=np.float64)
 if labels.shape!=(len(molecules),len(sequences)):
  raise RuntimeError("DeepDTA KIBA Y shape does not match original source order")
 c,p=np.where(np.isfinite(labels))
 values=labels[c,p].astype(np.float64)
 return sequences,molecules,p.astype(np.int32),c.astype(np.int32),values,["Y","ligands_can.txt","proteins.txt"]

def read_metz(root):
 drugs=pd.read_csv(root/"drug_info.csv")
 proteins=pd.read_csv(root/"prot_info.csv")
 pairs=pd.read_csv(root/"triplet_split_warm_filtered.csv")
 drugs=drugs.sort_values("index")
 proteins=proteins.sort_values("index")
 if drugs["index"].tolist()!=list(range(1,1424)):
  raise RuntimeError("PADME drug index not contiguous 1..1423")
 if proteins["index"].tolist()!=list(range(1,171)):
  raise RuntimeError("PADME protein index not contiguous 1..170")
 if pairs.duplicated(["drug","target"]).any():
  raise RuntimeError("Metz duplicate pair in frozen filtered version")
 if pairs[["drug","target","value","fold"]].isna().any().any():
  raise RuntimeError("Metz filtered file has missing value")
 if not (pairs.drug.between(1,1423).all() and pairs.target.between(1,170).all()):
  raise RuntimeError("PADME interaction IDs out of range")
 if sorted(set(pairs.fold))!=[1,2,3,4,5]:
  raise RuntimeError("PADME warm split markers unexpectedly changed")
 return (proteins.sequence.astype(str).tolist(),drugs.smiles.astype(str).tolist(),
         (pairs.target.to_numpy(dtype=np.int32)-1),
         (pairs.drug.to_numpy(dtype=np.int32)-1),
         pairs.value.to_numpy(dtype=np.float64),
         ["drug_info.csv","prot_info.csv","triplet_split_warm_filtered.csv","restructured_unique.csv"])

def protein_clusters(seqs):
 vec=TfidfVectorizer(analyzer="char",ngram_range=(3,3),
                      lowercase=False,norm="l2",dtype=np.float32)
 x=vec.fit_transform(seqs);sim=(x@x.T).toarray()
 parent=list(range(len(seqs)))
 def root(i):
  while parent[i]!=i:
   parent[i]=parent[parent[i]];i=parent[i]
  return i
 for i in range(len(seqs)):
  for j in range(i):
   if seqs[i]==seqs[j] or sim[i,j]>=PROTEIN_GROUP_COSINE:
    parent[root(i)]=root(j)
 idx=np.array([root(i) for i in range(len(seqs))])
 mapping={u:k for k,u in enumerate(np.unique(idx).tolist())}
 clusters=np.array([mapping[x] for x in idx],dtype=np.int32)
 return clusters,float(sim[~np.eye(len(sim),dtype=bool)].max())

def scaffold_clusters(smiles):
 clusters=[]
 for smi in smiles:
  mol=Chem.MolFromSmiles(smi)
  if mol is None:raise RuntimeError("Invalid source molecular notation")
  clusters.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "NO_RING")
 mapping={x:k for k,x in enumerate(sorted(set(clusters)))}
 return np.array([mapping[x] for x in clusters],dtype=np.int32)

def roles(group_ids,seed):
 arr=np.arange(len(group_ids),dtype=np.int32)
 trv,test=next(GroupShuffleSplit(n_splits=1,test_size=.18,random_state=seed).split(arr,groups=group_ids))
 tr_rel,val_rel=next(GroupShuffleSplit(n_splits=1,test_size=.18,random_state=seed+1).split(trv,groups=group_ids[trv]))
 train=np.sort(trv[tr_rel]);validation=np.sort(trv[val_rel]);test=np.sort(test)
 for a,b in [(train,validation),(train,test),(validation,test)]:
  if set(group_ids[a])&set(group_ids[b]):
   raise RuntimeError("Split has group overlap")
 if len(train)<3 or len(validation)<2 or len(test)<2:
  raise RuntimeError("Insufficient group-disjoint role sizes")
 return train,validation,test

def cache_features(seqs):
 mean=[]
 cache_shas=[]
 for seq in seqs:
  sid=hashlib.sha256(seq.encode("utf-8")).hexdigest()
  path=CACHE/(sid+".pt")
  if not path.is_file():raise RuntimeError(f"ESM2 embedding missing for {sid}")
  ck=torch.load(path,map_location="cpu",weights_only=True)
  if ck["sequence_hash"]!=sid or ck["embeddings"].shape!=(len(seq),1280):
   raise RuntimeError("Frozen ESM2 cache does not match sequence or dimensions")
  mean.append(ck["embeddings"].float().mean(axis=0).numpy())
  cache_shas.append(sid)
 return np.stack(mean).astype(np.float32),cache_shas

def fingerprints(smiles):
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
 out=np.empty((len(smiles),2048),dtype=np.uint8)
 canonical=[]
 for i,smile in enumerate(smiles):
  mol=Chem.MolFromSmiles(smile)
  if mol is None:raise RuntimeError("Invalid published source SMILES at row "+str(i))
  canonical.append(Chem.MolToSmiles(mol,isomericSmiles=True))
  DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),out[i])
 return out,canonical

def write_training_csv(path,rows,seqs,smiles,prot_idx,chem_idx,y):
 import csv
 with path.open("w",newline="",encoding="utf-8") as fh:
  w=csv.writer(fh)
  w.writerow(["smiles","sequence","label"])
  for k in rows:
   w.writerow([smiles[int(chem_idx[k])],seqs[int(prot_idx[k])],repr(float(y[k]))])

def main():
 if OUT.exists():raise RuntimeError("External benchmark output already exists; never overwrite")
 if not CACHE.is_dir():raise RuntimeError("Prior ESM2 frozen cache is missing")
 torch.set_num_threads(2)
 OUT.mkdir(parents=True,exist_ok=False)
 final={}
 for name,info in SOURCES.items():
  root=info["root"]
  sha=subprocess_commit(info["repository"])
  if sha!=info["src_sha"]:raise RuntimeError("Public source commit unexpectedly changed")
  if name=="kiba":
   seqs,molecules,pids,cids,values,srcs=read_kiba(root)
  else:
   seqs,molecules,pids,cids,values,srcs=read_metz(root)
  if (len(seqs),len(molecules),len(values))!=tuple(info["expected_entries"]):
   raise RuntimeError("External source differs from original recorded cohort size")
  if not np.isfinite(values).all():
   raise RuntimeError("Non-finite original labeled affinity")
  pfeat,sequence_hashes=cache_features(seqs)
  bits,canonical=fingerprints(molecules)
  protein_groups,maxseq=protein_clusters(seqs)
  chemical_groups=scaffold_clusters(canonical)
  dirname=OUT/name
  dirname.mkdir()
  features=dirname/"source_features.npz"
  np.savez_compressed(features,protein_features=pfeat,
      compound_morgan_bits=bits,protein_index=pids,compound_index=cids,
      observed_score=values.astype(np.float32),
      chemical_smiles=np.asarray(canonical),
      protein_hashes=np.asarray(sequence_hashes))
  sourcechecks={path:file_sha(root/path) for path in srcs}
  manifest={
    "source_repo_sha":sha,
    "source_url":("https://github.com/hkmztrk/DeepDTA/tree/master/data/kiba" if name=="kiba" else
                  "https://github.com/simonfqy/PADME/tree/master/metz_data"),
    "raw_file_sha256s":sourcechecks,
    "features_sha256":file_sha(features),
    "label_definition":info["labels"],
    "not_same_as_perceiver_published_filtered_folds":True,
    "pretrained_features":"frozen pre-existing ESM2_t33_650M + radius2 Morgan 2048 chiral",
    "cache_exact_sequence_sha256s_verified":True,
    "dataset":name,"n_proteins":len(seqs),"n_compounds":len(molecules),
    "n_pairs_observed":int(len(values)),
    "missing_pairs_are_not_training_zeros":True,
    "n_protein_groups_3mer_035":len(np.unique(protein_groups)),
    "n_chemical_murcko_groups":len(np.unique(chemical_groups)),
    "max_protein_sim_nonself":maxseq,
    "no_label_derived_grouping":True,
    "no_test_labels_exported":True,
    "metric_priority":["target_macro_rmse_on_observed_pairs","pooled_rmse_on_observed_pairs",
                        "concordance_index_when_valid"],
    "seeds":[2931,2932],
    "training_budget":"frozen before ANY external test, no outcome-based tuning",
    "caveats":["KIBA is composite score, not DAVIS pKd",
               "Metz uses already-filtered PADME 35259-row numeric subset",
               "Raw DeepDTA KIBA 118254/2111 differs from published PerceiverCPI filtered 117657/2068",
               "Protein groups from character 3mer TFIDF are proxy NOT curated kinase families"]
  }
  splits=[]
  allp=np.arange(len(seqs),dtype=int)
  allc=np.arange(len(molecules),dtype=int)
  for scenario,seed in SPLITS.items():
   group_protein=scenario in ("protein_cold","double_cold")
   group_chem=scenario in ("compound_cold","double_cold")
   pt,pv,pe=roles(protein_groups,seed) if group_protein else (allp,allp,allp)
   ct,cv,ce=roles(chemical_groups,seed+100) if group_chem else (allc,allc,allc)
   train=np.flatnonzero(np.isin(pids,pt)&np.isin(cids,ct))
   val=np.flatnonzero(np.isin(pids,pv)&np.isin(cids,cv))
   test=np.flatnonzero(np.isin(pids,pe)&np.isin(cids,ce))
   if min(len(train),len(val),len(test))<100:
    raise RuntimeError(f"Sparse external split too small: {name}/{scenario}")
   if set(train)&set(val) or set(train)&set(test) or set(val)&set(test):
    raise RuntimeError("Leakage in external observed pair role splits")
   # The split files contain indexes but *NO hidden affinity labels*.
   scenario_folder=dirname/scenario
   scenario_folder.mkdir()
   csvtr=scenario_folder/"train.csv"
   csvval=scenario_folder/"validation.csv"
   write_training_csv(csvtr,train,seqs,canonical,pids,cids,values)
   write_training_csv(csvval,val,seqs,canonical,pids,cids,values)
   proxy=scenario_folder/"validation_proxy_test.csv"
   proxy.write_bytes(csvval.read_bytes())
   im={
     "dataset":name,"scenario":scenario,"seed":seed,
     "label_definition":manifest["label_definition"],
     "source_features_sha256":manifest["features_sha256"],
     "test_labels_not_exported":True,
     "observed_only_no_unobserved_negative_imputation":True,
     "protein_grouping":"connected TF-IDF 3-mer cosine >=0.35; not curated family",
     "chemical_grouping":"Bemis-Murcko scaffold; not fingerprint-distance exclusion",
     "protein_train":pt.tolist(),"protein_val":pv.tolist(),"protein_test":pe.tolist(),
     "compound_train":ct.tolist(),"compound_val":cv.tolist(),"compound_test":ce.tolist(),
     "pair_train_row_indices":train.tolist(),
     "pair_val_row_indices":val.tolist(),
     "pair_test_row_indices":test.tolist(),
     "n_pairs":{"train":len(train),"val":len(val),"test":len(test)},
     "n_observed_proteins":{"train":len(set(pids[train])),"val":len(set(pids[val])),"test":len(set(pids[test]))},
     "n_observed_compounds":{"train":len(set(cids[train])),"val":len(set(cids[val])),"test":len(set(cids[test]))},
     "train_protein_overlap_with_test":bool(set(pids[train])&set(pids[test])),
     "train_compound_overlap_with_test":bool(set(cids[train])&set(cids[test])),
     "csv_sha256":{"train.csv":file_sha(csvtr),"validation.csv":file_sha(csvval),
                   "validation_proxy_test.csv":file_sha(proxy)},
     "training_seeds":TRAINING_SEEDS,
     "allow_external_metric_comparison_only_with_same_pair_rows":True
   }
   path=scenario_folder/"SPLIT_LOCKED.json"
   path.write_text(json.dumps(im,indent=2),encoding="utf-8")
   splits.append({"name":scenario,"train_pairs":len(train),"val_pairs":len(val),"sealed_test_pairs":len(test),
                  "manifest_sha256":file_sha(path)})
  manifest["splits"]=splits
  (dirname/"SOURCE_AUDIT.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
  final[name]=manifest
  print(json.dumps({"dataset":name,"source":sha,"n_proteins":len(seqs),
       "n_compounds":len(molecules),"n_observed_pairs":len(values),
       "n_protein_groups":manifest["n_protein_groups_3mer_035"],
       "n_murcko_groups":manifest["n_chemical_murcko_groups"],
       "splits":splits,"test_labels_exported":False},indent=2),flush=True)
 (OUT/"SUITE_AUDIT.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
 print("EXTERNAL_SOURCE_AND_SPLIT_FREEZE_COMPLETE_NO_TEST_RESULTS",flush=True)

def subprocess_commit(root):
 import subprocess
 return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()

if __name__=="__main__":main()
