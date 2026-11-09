"""Compare six published NRP1 SPR Kd numbers with the 26 source-record Kd values.
Numeric equality is only a candidate, never a compound identity or approval.
"""
import csv,json,re,math
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
details=json.loads((root/"research/source_access_v4/PRIMARY_AFFINITY_TABLE_DETAIL.json").read_text(encoding="utf-8"))
nrp=next(x for x in details["publications"] if x["pmid"]=="29648813")
table=next(x for x in nrp["affinity_related_tables"] if x["id"]=="tbl3")
rows=table["visible_rows"]
assert "SPR" in " ".join(rows[0]) and "μM" in " ".join(rows[0])
published=[]
for r in rows[1:]:
 m=re.match(r"\s*(\d+(?:\.\d+)?)",r[2])
 assert m,r
 kd=float(m.group(1))
 published.append({"paper_compound_id":r[0],"paper_substituent":r[1],"paper_spr_kd_um":kd,"paper_spr_kd_nm":kd*1000,"paper_table":"tbl3"})
assert len(published)==6
with (audit/"source_record_evidence_worklist_30_LOCAL_PRIVATE.csv").open(encoding="utf-8-sig",newline="") as f:
 relevant=[r for r in csv.DictReader(f) if r["assay_key"]=="50002067_1"]
assert len(relevant)==26
ids={r["reactant_set_id"] for r in relevant}
source=[]
with (audit/"bindingdb_full_kd_source_provenance.jsonl").open(encoding="utf-8") as f:
 for line in f:
  item=json.loads(line)
  if str(item.get("BindingDB Reactant_set_id")) in ids:source.append(item)
assert len(source)==26
for p in published:
 p["exact_numeric_source_record_matches"]=[]
 p["within_5_percent_source_record_matches"]=[]
 for s in source:
  try:v=float(str(s.get("Kd (nM)","")).strip())
  except ValueError:continue
  if not math.isfinite(v) or v<=0:continue
  if math.isclose(v,p["paper_spr_kd_nm"],rel_tol=1e-9,abs_tol=1e-7):
   p["exact_numeric_source_record_matches"].append(str(s["BindingDB Reactant_set_id"]))
  if abs(v-p["paper_spr_kd_nm"])/p["paper_spr_kd_nm"]<=.05:
   p["within_5_percent_source_record_matches"].append(str(s["BindingDB Reactant_set_id"]))
out={"status":"NUMERIC_CANDIDATES_ONLY_NOT_CHEMICAL_IDENTITY","paper":"PMID 29648813 Table 3","paper_values":published,"bindingdb_assay_source_rows":26,"paper_values_count":6,"exact_numeric_candidate_pairs":sum(len(p["exact_numeric_source_record_matches"]) for p in published),"within_5_percent_candidate_pairs":sum(len(p["within_5_percent_source_record_matches"]) for p in published),"chemical_structure_matches_verified":0,"record_level_approved":0,"warning":"Equal Kd does not establish compound identity; source table may not cover all 26 records."}
private=audit/"nrp1_table3_numeric_candidates_LOCAL_PRIVATE.json"
private.write_text(json.dumps(out,indent=2),encoding="utf-8")
public={k:v for k,v in out.items() if k!="paper_values"}
public["paper_compound_kd_nm"]=[{"compound":p["paper_compound_id"],"kd_nm":p["paper_spr_kd_nm"],"numeric_candidate_count":len(p["exact_numeric_source_record_matches"])} for p in published]
(root/"research/source_access_v4/NRP1_TABLE3_NUMERIC_CROSSCHECK.json").write_text(json.dumps(public,indent=2),encoding="utf-8")
print(json.dumps(public,indent=2))
