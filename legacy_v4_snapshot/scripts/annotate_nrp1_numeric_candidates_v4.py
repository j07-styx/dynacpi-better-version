"""Attach six exact numeric publication candidates to the private review worksheet.
Does not certify chemical identity, source mapping, or affinity.
"""
import csv,json,os,tempfile,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
candidate=json.loads((audit/"nrp1_table3_numeric_candidates_LOCAL_PRIVATE.json").read_text(encoding="utf-8"))
assert candidate["exact_numeric_candidate_pairs"]==6
mapping={}
for p in candidate["paper_values"]:
 matches=p["exact_numeric_source_record_matches"]
 assert len(matches)==1
 rid=matches[0]
 assert rid not in mapping
 mapping[rid]=p["paper_compound_id"]
path=audit/"source_record_evidence_worklist_30_LOCAL_PRIVATE.csv"
with path.open(encoding="utf-8-sig",newline="") as f:
 reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
assert len(rows)==921
modified=0
for r in rows:
 rid=r["reactant_set_id"]
 if rid not in mapping:continue
 assert r["assay_key"]=="50002067_1" and r["review_decision"]=="UNREVIEWED"
 assert not r["original_record_mapping_verified"] and not r["direct_equilibrium_kd_verified"]
 label="PMID 29648813 Table 3; numeric-only candidate compound "+mapping[rid]+"; chemical identity NOT verified"
 if r["source_table_or_figure_locator"] not in ("",label):raise ValueError("Refusing overwrite of reviewer data")
 if not r["source_table_or_figure_locator"]:
  r["source_table_or_figure_locator"]=label;modified+=1
assert modified in (0,6),modified
if modified:
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix="nrp1_numeric_",suffix=".csv")
 try:
  with os.fdopen(fd,"w",encoding="utf-8",newline="") as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
assert sum("numeric-only candidate" in r["source_table_or_figure_locator"] for r in rows)==6
print(json.dumps({"status":"SIX_NUMERIC_CANDIDATES_ANNOTATED_UNVERIFIED","new_private_review_annotations":modified,"candidate_rows":6,"record_mapping_verified":0,"approved":0,"private_worklist_sha256":hashlib.sha256(path.read_bytes()).hexdigest()},indent=2))
