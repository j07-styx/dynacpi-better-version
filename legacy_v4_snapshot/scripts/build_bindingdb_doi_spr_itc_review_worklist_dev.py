"""Source-paper review queue for prospective SPR/ITC eligibility adjudication.

Only draws current TRAIN and INNER_DEV metadata; reserved TEST ignored.
Reproducible source-evidence worksheet, not a source-verification assertion.
"""
import csv,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"data"/"audit"
OUT=ROOT/"research"/"bindingdb_doi_spr_itc_source_review_worklist_dev.csv"
SUMMARY=ROOT/"research"/"bindingdb_doi_spr_itc_source_review_manifest_dev.json"

def read(p):
 with p.open(encoding="utf8") as f:
  for l in f:yield json.loads(l)
def main():
 m=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
 tr=set(m["train_components"]);dv=set(m["inner_dev_components"]);te=set(m["test_components"])
 assert not tr&dv and not tr&te and not dv&te
 comp={}
 for group in read(A/"bindingdb_full_kd_protein_components.jsonl"):
  for s in group["members"]:comp[s]=group["component_id"]
 taskmin={(x["sequence_sha256"],x["assay_key"]) for x in read(A/"bindingdb_full_kd_task_summary.jsonl") if x["unique_canonical_compounds"]>=15}
 rows={"train":[],"inner_dev":[]}
 for x in read(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
  if (x["protein_seq_sha256"],x["assay_key"]) not in taskmin or x["range_pKd"]>.1:continue
  c=comp[x["protein_seq_sha256"]]
  if c in tr:rows["train"].append(x)
  elif c in dv:rows["inner_dev"].append(x)
 need={(x["protein_seq_sha256"],x["assay_key"]) for part in rows.values() for x in part}
 assayctx={}
 for x in read(A/"bindingdb_full_kd_assay_contexts.jsonl"):
  key=(x["sequence_sha256"],x["assay_key"])
  if key in need:assayctx[key]=x
 assert need==set(assayctx)
 grouped=defaultdict(lambda:{"rows":[],"doi":Counter()})
 for part,records in rows.items():
  for x in records:
   a=assayctx[(x["protein_seq_sha256"],x["assay_key"])]
   text=(a["name"]+" "+a["description"]).lower()
   method="SPR" if re.search(r"\bspr\b|surface plasmon resonance",text) else "ITC" if re.search(r"\bitc\b|isothermal titration",text) else None
   if method is None or a["article_doi_count"]<1 or re.search(r"\bic50\b|\bki\b|\bcell\w*|thermal shift|\bdsf\b",text):
    continue
   grouped[(part,x["protein_seq_sha256"],x["assay_key"],method)]["rows"].append(x)
 neededids=set()
 for v in grouped.values():
  for r in v["rows"]:
   neededids.update(r["reactant_ids"])
 byid={}
 for r in read(A/"bindingdb_full_kd_candidates.jsonl"):
  rid=r["reactant_set_id"]
  if rid in neededids:
   byid[rid]=(r.get("article_doi") or "").strip()
 towrite=[]
 for (part,seq,assay,method),v in grouped.items():
  a=assayctx[(seq,assay)]
  ids=set(rid for x in v["rows"] for rid in x["reactant_ids"])
  dois=Counter(byid.get(rid,"") for rid in ids)
  uniquechem=len({r["compound_sha256"] for r in v["rows"]})
  desc=a["description"] or ""
  flags=[
   "free_text_method_not_verified",
   "source_full_text_not_checked",
   "construct_not_verified",
   "assay_ph_temperature_conditions_not_verified"
  ]
  if not re.search(r"\bkd\b|\bdissociation constant\b",(a["name"]+" "+desc).lower()):
   flags.append("explicit_kd_endpoint_missing_from_assay_description")
  if len([d for d in dois if d])>1:flags.append("multiple_article_doi")
  if "" in dois:flags.append("missing_doi_for_some_rows")
  towrite.append({
   "priority":1 if uniquechem>=80 else 2 if uniquechem>=30 else 3,
   "split":part,
   "protein_component_sha":comp[seq],
   "protein_sequence_sha":seq,
   "assay_key":assay,
   "assay_method_keyword":method,
   "assay_name":a["name"],
   "assay_description":desc,
   "candidate_compound_rows":len(v["rows"]),
   "unique_compounds":uniquechem,
   "article_dois":";".join(sorted(d for d in dois if d)),
   "record_doi_coverage":f"{sum(cnt for doi,cnt in dois.items() if doi)}/{sum(dois.values())}",
   "review_flags":";".join(flags),
   "review_status":"UNREVIEWED",
   "reviewer_evidence_url":"",
   "reviewer_endpoint_category":"",
   "reviewer_notes":""
  })
 towrite.sort(key=lambda x:(x["priority"],-x["unique_compounds"],x["split"],x["assay_key"]))
 assert towrite
 with OUT.open("w",encoding="utf-8-sig",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=list(towrite[0]))
  writer.writeheader();writer.writerows(towrite)
 bypart=Counter(x["split"] for x in towrite)
 manifest={
  "status":"METADATA_ONLY_UNREVIEWED_SOURCE_EVIDENCE_WORKLIST_NO_TEST",
  "assays_to_review":len(towrite),
  "assays_by_split":dict(bypart),
  "first_pass_priority_count":sum(x["priority"]==1 for x in towrite),
  "assays_missing_explicit_endpoint_description":sum(
      "explicit_kd_endpoint_missing" in x["review_flags"] for x in towrite),
  "assays_with_nonempty_source_doi":sum(bool(x["article_dois"]) for x in towrite),
  "csv_sha256":hashlib.sha256(OUT.read_bytes()).hexdigest(),
  "note":"This is NOT confirmation of comparable or exact assays. Explicit external publication+construct verification is required.",
  "reserved_test_used":False
 }
 SUMMARY.write_text(json.dumps(manifest,indent=2))
 print(json.dumps(manifest,indent=2),flush=True)
 print("TOP FIVE: "+json.dumps([
  {k:x[k] for k in ("split","assay_key","assay_method_keyword","unique_compounds","article_dois","review_flags")}
  for x in towrite[:5]],indent=2),flush=True)

if __name__=="__main__":main()
