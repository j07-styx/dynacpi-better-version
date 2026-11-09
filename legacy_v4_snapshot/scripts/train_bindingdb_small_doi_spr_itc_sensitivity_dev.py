"""Small-coverage DOI-linked SPR/ITC sensitivity experiment, not publication QC.

Select one record per identical protein+compound with metadata-only decisions.
Compare fixed models by train-only protein-component CV; assess inner-dev once.
Prospective test is not materialized and must remain untouched.
"""
import hashlib,json,re,time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from train_bindingdb_descriptor_protein_boosting_dev import (
    molecule_features,protein_features,records)
from train_bindingdb_cold_protein_prior_dev import metric
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"data"/"audit"
OUT=ROOT/"research"/"bindingdb_doi_linked_spr_itc_small_cohort_dev.json"

def load():
 m=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
 tr=set(m["train_components"]);dv=set(m["inner_dev_components"]);te=set(m["test_components"])
 assert not tr&dv and not tr&te and not dv&te
 groups={}
 for row in records(A/"bindingdb_full_kd_protein_components.jsonl"):
  for sha in row["members"]:groups[sha]=row["component_id"]
 tasks={(r["sequence_sha256"],r["assay_key"])
        for r in records(A/"bindingdb_full_kd_task_summary.jsonl")
        if r["unique_canonical_compounds"]>=15}
 parts={"train":[],"inner_dev":[]}
 for r in records(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
  if (r["protein_seq_sha256"],r["assay_key"]) not in tasks or r["range_pKd"]>.1:continue
  g=groups[r["protein_seq_sha256"]]
  if g in tr:parts["train"].append(r)
  elif g in dv:parts["inner_dev"].append(r)
 wanted={(r["protein_seq_sha256"],r["assay_key"]) for rows in parts.values() for r in rows}
 assays={}
 for row in records(A/"bindingdb_full_kd_assay_contexts.jsonl"):
  key=row["sequence_sha256"],row["assay_key"]
  if key in wanted:assays[key]=row
 def check(r):
  entry=assays[r["protein_seq_sha256"],r["assay_key"]]
  desc=(entry["name"]+" "+entry["description"]).lower()
  physical=bool(re.search(r"\bspr\b|\bitc\b|surface plasmon resonance|isothermal titration",desc))
  wrong=bool(re.search(r"\bic50\b|\bki\b|\bcell\w*|thermal shift|\bdsf\b",desc))
  return physical and not wrong and entry["article_doi_count"]>=1
 clean={}
 for part,rows in parts.items():
  uniq={}
  for row in rows:
   if not check(row):continue
   key=(row["protein_seq_sha256"],row["compound_sha256"])
   if key not in uniq or row["assay_key"]<uniq[key]["assay_key"]:
    uniq[key]=row
  clean[part]=sorted(uniq.values(),key=lambda r:(r["protein_seq_sha256"],r["compound_sha256"]))
 expected=json.loads((ROOT/"research/bindingdb_frozen_spr_itc_tier_viability_dev.json").read_text())
 for part in clean:
  assert len(clean[part])==expected["partitions"][part]["method_article_no_mixed"]["unique_sequence_compound_pairs"]
 needed={r["protein_seq_sha256"] for rows in clean.values() for r in rows}
 seqs={}
 for r in records(A/"bindingdb_full_kd_candidates.jsonl"):
  seq=r["sequence"];sha=hashlib.sha256(seq.encode()).hexdigest()
  if sha in needed:seqs[sha]=seq
  if len(seqs)==len(needed):break
 assert len(seqs)==len(needed)
 return clean,seqs,groups

def build(items,seqs):
 cache={};protein={}
 X=np.zeros((len(items),42),dtype=np.float32)
 y=np.zeros(len(items),dtype=np.float64)
 for i,row in enumerate(items):
  sm=row["smiles_isomeric"];p=row["protein_seq_sha256"]
  if sm not in cache:cache[sm]=molecule_features(sm)
  if p not in protein:protein[p]=protein_features(seqs[p])
  X[i,:21]=cache[sm];X[i,21:]=protein[p]
  y[i]=row["median_pKd"]
 return X,y

def predict(name,xtr,ytr,gtr,xval):
 if name=="global_mean":
  return np.full(len(xval),float(ytr.mean()))
 if name=="component_balanced_mean":
  c=Counter(gtr);w=np.asarray([1./np.sqrt(c[g]) for g in gtr])
  return np.full(len(xval),float(np.average(ytr,weights=w)))
 if name=="chemical_ridge":
  ss=StandardScaler()
  tr=ss.fit_transform(xtr[:,:21]);va=ss.transform(xval[:,:21])
  model=Ridge(alpha=100.)
  model.fit(tr,ytr)
  return model.predict(va)
 if name=="chemical_boost" or name=="chemical_protein_boost":
  width=21 if name=="chemical_boost" else 42
  model=HistGradientBoostingRegressor(
       max_iter=120,learning_rate=.05,max_leaf_nodes=15,max_depth=3,
       min_samples_leaf=20,l2_regularization=15.,
       early_stopping=False,random_state=2103)
  model.fit(xtr[:,:width],ytr)
  return model.predict(xval[:,:width])
 raise ValueError(name)

def main():
 start=time.time()
 clean,seqs,groups=load()
 tr=clean["train"];dev=clean["inner_dev"]
 X,y=build(tr,seqs);V,z=build(dev,seqs)
 gt=np.asarray([groups[r["protein_seq_sha256"]] for r in tr])
 gv=np.asarray([groups[r["protein_seq_sha256"]] for r in dev])
 assert set(gt).isdisjoint(set(gv))
 names=["global_mean","component_balanced_mean","chemical_ridge",
        "chemical_boost","chemical_protein_boost"]
 folds={n:[] for n in names}
 splitter=GroupShuffleSplit(n_splits=3,test_size=.25,random_state=20261108)
 for i,(ii,jj) in enumerate(splitter.split(gt,groups=gt),1):
  for name in names:
   p=predict(name,X[ii],y[ii],gt[ii],X[jj])
   folds[name].append(metric(y[jj],p,gt[jj]))
  print(json.dumps({"fold":i,"train":len(ii),"heldout":len(jj),
                    "metrics":{name:folds[name][-1] for name in names}}),flush=True)
 baseline_pool=float(np.mean([m["pooled_rmse"] for m in folds["global_mean"]]))
 baseline_macro=float(np.mean([m["component_macro_rmse"] for m in folds["global_mean"]]))
 scores=[]
 for name in names:
  pool=float(np.mean([m["pooled_rmse"] for m in folds[name]]))
  macro=float(np.mean([m["component_macro_rmse"] for m in folds[name]]))
  scores.append({"name":name,"cv_pooled":pool,"cv_macro":macro,
                 "cv_joint":.5*(pool/baseline_pool+macro/baseline_macro)})
 scores.sort(key=lambda r:r["cv_joint"])
 eligible=[a for a in scores if a["cv_pooled"]<=baseline_pool and
           a["cv_macro"]<=baseline_macro]
 winner=eligible[0]["name"] if eligible else "global_mean"
 dev_baseline=metric(z,predict("global_mean",X,y,gt,V),gv)
 dev_winner=metric(z,predict(winner,X,y,gt,V),gv)
 out={
  "status":"SMALL_DOI_LINKED_SPR_ITC_COHORT_EXPLORATORY_NO_PROSPECTIVE_TEST",
  "metadata_filter":"SPR/ITC text, article DOI, no cellular/DSF/IC50/Ki mixed tokens; no Kd phrase requirement beyond original BindingDB numeric Kd field",
  "deduplication":"deterministic assay_key sort per canonical protein/compound, no labels used",
  "training_pairs":len(tr),"inner_dev_pairs":len(dev),
  "training_components":len(set(gt)),"dev_components":len(set(gv)),
  "train_assay_contexts_not_independently_verified":True,
  "model_candidates":names,
  "cv_folds":folds,"cv_ranking":scores,
  "cv_selected":winner,
  "dev_train_mean":dev_baseline,
  "dev_selected":dev_winner,
  "seconds":round(time.time()-start,1),
  "limitations":["Only a small minority of Kd records meet metadata filters; experiment not paper benchmark.",
                 "Assay and construct still unverified, Kd phrase not always stated in SPR/ITC description.",
                 "No assay key leakage into features; split remains protein-component disjoint.",
                 "Development has been inspected; no prospective test accessed."]}
 OUT.write_text(json.dumps(out,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k not in ("cv_folds","cv_ranking","model_candidates")},indent=2),flush=True)

if __name__=="__main__":
 main()
