"""Evaluate viability tiers for an assay-method-harmonized Kd cohort.

Only TRAIN and previously inspected INNER_DEV; no prospective-test data accessed.
No labels used to choose a tier or prioritize records.
"""
import json,re
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"data"/"audit"
OUT=ROOT/"research"/"bindingdb_frozen_spr_itc_tier_viability_dev.json"

def read(p):
 with p.open(encoding="utf8") as f:
  for s in f:yield json.loads(s)

def method(description):
 s=description.lower()
 if "surface plasmon resonance" in s or re.search(r"\bspr\b",s):return "SPR"
 if "isothermal titration" in s or re.search(r"\bitc\b",s):return "ITC"
 return None

def main():
 m=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
 train=set(m["train_components"]);dev=set(m["inner_dev_components"]);test=set(m["test_components"])
 assert not train&dev and not train&test and not dev&test
 comp={}
 for g in read(A/"bindingdb_full_kd_protein_components.jsonl"):
  for s in g["members"]:comp[s]=g["component_id"]
 eligible={(r["sequence_sha256"],r["assay_key"]) for r in
           read(A/"bindingdb_full_kd_task_summary.jsonl")
           if r["unique_canonical_compounds"]>=15}
 partitions={"train":[],"inner_dev":[]}
 for r in read(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
  if (r["protein_seq_sha256"],r["assay_key"]) not in eligible or r["range_pKd"]>.1:continue
  c=comp[r["protein_seq_sha256"]]
  if c in train:partitions["train"].append(r)
  elif c in dev:partitions["inner_dev"].append(r)
 assert len(partitions["train"])==m["train_pairs"]
 assert len(partitions["inner_dev"])==m["inner_dev_pairs"]
 wanted={(r["protein_seq_sha256"],r["assay_key"]) for part in partitions.values() for r in part}
 ctx={}
 for r in read(A/"bindingdb_full_kd_assay_contexts.jsonl"):
  key=(r["sequence_sha256"],r["assay_key"])
  if key in wanted:ctx[key]=r
 assert set(ctx)==wanted

 rules={
   "method_only":lambda c,s,mo:mo is not None,
   "method_plus_article":lambda c,s,mo:mo is not None and c["article_doi_count"]>0,
   "method_plus_endpoint":lambda c,s,mo:mo is not None and bool(re.search(r"\bkd\b|\bdissociation constant\b",s)),
   "method_plus_article_and_endpoint":lambda c,s,mo:mo is not None and c["article_doi_count"]>0 and bool(re.search(r"\bkd\b|\bdissociation constant\b",s)),
   "method_article_endpoint_no_mixed":lambda c,s,mo:mo is not None and c["article_doi_count"]>0 and bool(re.search(r"\bkd\b|\bdissociation constant\b",s)) and not re.search(r"\bic50\b|\bki\b|\bcell\w*|thermal shift|\bdsf\b",s),
   "method_article_no_mixed":lambda c,s,mo:mo is not None and c["article_doi_count"]>0 and not re.search(r"\bic50\b|\bki\b|\bcell\w*|thermal shift|\bdsf\b",s)
 }
 out={"status":"METADATA_ONLY_SPR_ITC_TIER_VIABILITY_TRAIN_AND_DEV","rules":list(rules),"partitions":{}}
 for name,rs in partitions.items():
  results={}
  for title,rule in rules.items():
   selected=[]
   for r in rs:
    c=ctx[r["protein_seq_sha256"],r["assay_key"]]
    s=(c["name"]+" "+c["description"]).lower()
    mo=method(s)
    if rule(c,s,mo):selected.append((r,mo))
   picked={}
   for r,mo in selected:
    key=(r["protein_seq_sha256"],r["compound_sha256"])
    if key not in picked or r["assay_key"]<picked[key][0]["assay_key"]:
     picked[key]=(r,mo)
   selected_unique=list(picked.values())
   counts=Counter(comp[r["protein_seq_sha256"]] for r,_ in selected_unique)
   methods=Counter(mo for _,mo in selected_unique)
   results[title]={
       "qualifying_rows":len(selected),
       "unique_sequence_compound_pairs":len(selected_unique),
       "protein_clusters":len(counts),
       "protein_sequences":len({r["protein_seq_sha256"] for r,_ in selected_unique}),
       "compounds":len({r["compound_sha256"] for r,_ in selected_unique}),
       "assay_keys":len({r["assay_key"] for r,_ in selected_unique}),
       "modalities":dict(methods),
       "largest_cluster_pair_fraction":max(counts.values())/len(selected_unique) if counts else None
   }
  out["partitions"][name]=results
  print(json.dumps({"partition":name,"viability":results},indent=2),flush=True)
 out["warnings"]=[
   "These are keyword-screened metadata tiers, not validated comparable Kd assays.",
   "Even the source Kd (nM) field may contain mixed assay mechanisms.",
   "Existing cohort's within-assay replicate QC used outcome labels historically.",
   "No outcome labels were examined by this viability screen; no prospective test was accessed."
 ]
 OUT.write_text(json.dumps(out,indent=2))

if __name__=="__main__":main()
