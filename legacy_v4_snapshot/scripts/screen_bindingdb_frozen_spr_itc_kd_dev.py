"""Metadata-only screen for assay-harmonized physical Kd candidates.

Hard rule: only explicit SPR/ITC assay texts with some article DOI
and an explicit Kd/dissociation-constant phrase; exclude cell and
thermal-shift assay descriptions. Does not read or score reserved test
examples; retained rows selected exclusively from source metadata.
"""
import json,re,hashlib
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"data"/"audit"
OUT=ROOT/"research"/"bindingdb_frozen_spr_itc_source_metadata_screen_dev.json"

def rows(path):
 with path.open(encoding="utf8") as handle:
  for line in handle:
   yield json.loads(line)

def classify(s):
 s=s.lower()
 if ("surface plasmon resonance" in s) or re.search(r"\bspr\b",s):return "SPR"
 if ("isothermal titration" in s) or re.search(r"\bitc\b",s):return "ITC"
 if ("thermal shift" in s) or re.search(r"\bdsf\b",s):return "DSF"
 if re.search(r"\bcell",s):return "CELL"
 return "OTHER"

def main():
 m=json.loads((ROOT/"research"/"bindingdb_frozen_component_split_manifest_dev.json").read_text())
 train=set(m["train_components"]);dev=set(m["inner_dev_components"]);test=set(m["test_components"])
 assert not train&dev and not train&test and not dev&test
 component={}
 for group in rows(A/"bindingdb_full_kd_protein_components.jsonl"):
  for seq in group["members"]:component[seq]=group["component_id"]
 tasks={(r["sequence_sha256"],r["assay_key"])
        for r in rows(A/"bindingdb_full_kd_task_summary.jsonl")
        if r["unique_canonical_compounds"]>=15}
 partitions={"train":[],"inner_dev":[]}
 for r in rows(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
  # This value was used to define original provisional cohort. No new dev label based selection.
  if (r["protein_seq_sha256"],r["assay_key"]) not in tasks or r["range_pKd"]>.1:
   continue
  comp=component[r["protein_seq_sha256"]]
  if comp in train:partitions["train"].append(r)
  elif comp in dev:partitions["inner_dev"].append(r)
 assert len(partitions["train"])==m["train_pairs"] and len(partitions["inner_dev"])==m["inner_dev_pairs"]
 wanted={(r["protein_seq_sha256"],r["assay_key"]) for part in partitions.values() for r in part}
 contexts={}
 for r in rows(A/"bindingdb_full_kd_assay_contexts.jsonl"):
  key=(r["sequence_sha256"],r["assay_key"])
  if key in wanted:contexts[key]=r
 assert set(contexts)==wanted
 out={"status":"PROVISIONAL_SPR_ITC_KD_METADATA_SCREEN_NO_TEST",
      "selection_rule":{
        "methods":"explicit SPR or ITC in assay name/description",
        "require":"article DOI count >= 1 and literal Kd or dissociation constant in assay text",
        "exclude":"cell/cellular, thermal shift and DSF, IC50 and Ki mentions",
        "deduplicate":"one assay per identical sequence+canonical compound, lexicographically lowest assay key; no labels used",
        "warning":"Keyword screen does not verify biophysical assay conditions or source paper correctness"
       },
      "partitions":{}}
 for part,records in partitions.items():
  modality=Counter()
  pass_records=[]
  reasons=Counter()
  for r in records:
   c=contexts[r["protein_seq_sha256"],r["assay_key"]]
   text=(c["name"]+" "+c["description"]).lower()
   method=classify(text);modality[method]+=1
   if method not in ("SPR","ITC"):reasons["non_SPR_ITC"]+=1;continue
   if c["article_doi_count"]<1:reasons["no_article_doi"]+=1;continue
   if not re.search(r"\bkd\b|\bdissociation constant\b",text):reasons["no_explicit_kd"]+=1;continue
   if re.search(r"\bic50\b|\bki\b|\bcell\w*|thermal shift|\bdsf\b",text):
    reasons["mixed_or_excluded_measurement"]+=1;continue
   pass_records.append(r)
  byinput=defaultdict(list)
  for r in pass_records:
   byinput[r["protein_seq_sha256"],r["compound_sha256"]].append(r)
  picked=[]
  for rr in byinput.values():
   rr.sort(key=lambda r:(r["assay_key"],r["reactant_ids"][0] if r["reactant_ids"] else ""))
   picked.append(rr[0])
  ncomp=len({component[r["protein_seq_sha256"]] for r in picked})
  unique_assays=len({r["assay_key"] for r in picked})
  bymod=Counter(classify(
    (contexts[r["protein_seq_sha256"],r["assay_key"]]["name"]+" "+
     contexts[r["protein_seq_sha256"],r["assay_key"]]["description"])
    ) for r in picked)
  summary={
   "original_rows":len(records),"raw_modalities":dict(modality),
   "raw_qualifying_rows":len(pass_records),
   "unique_input_qualifying_rows":len(picked),
   "qualifying_protein_components":ncomp,
   "qualifying_assay_keys":unique_assays,
   "unique_input_method_counts":dict(bymod),
   "rejected_counts":dict(reasons),
   "retained_task_keys":sorted({(r["protein_seq_sha256"],r["assay_key"]) for r in picked}),
   "selected_input_row_digest":hashlib.sha256("\n".join(sorted(
     r["protein_seq_sha256"]+"|"+r["compound_sha256"]+"|"+r["assay_key"] for r in picked
    )).encode()).hexdigest()
   }
  out["partitions"][part]=summary
  print(json.dumps({k:v for k,v in summary.items() if k!="retained_task_keys"},indent=2),flush=True)
 out["limitations"]=[
  "Metadata-only heuristic; not a publication-verified source-quality gate.",
  "The original input cohort was previously assembled using within-assay replicate-spread QC.",
  "Selection uses no dev outcome labels beyond that historical cohort definition.",
  "Reserved prospective test neither materialized nor measured; full test protocol must be locked separately.",
  "Some publication DOI assignments might be incomplete or misinterpreted."
 ]
 OUT.write_text(json.dumps(out,indent=2),encoding="utf8")

if __name__=="__main__":main()
