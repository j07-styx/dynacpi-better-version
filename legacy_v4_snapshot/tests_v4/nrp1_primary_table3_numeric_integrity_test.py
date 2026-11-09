"""Fail-closed regression test of six NRP1 source Kd numeric candidates.
Rechecks PMID, DOI, Kd endpoint, numeric uniqueness, private annotations,
and absence of any scientific approval.
"""
import csv,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
p=json.loads((audit/"nrp1_table3_numeric_candidates_LOCAL_PRIVATE.json").read_text(encoding="utf-8"))
with (audit/"source_record_evidence_worklist_30_LOCAL_PRIVATE.csv").open(encoding="utf-8-sig",newline="") as f:
 rows={r["reactant_set_id"]:r for r in csv.DictReader(f)}
assert len(rows)==921
expected={}
for value in p["paper_values"]:
 matches=value["exact_numeric_source_record_matches"]
 assert len(matches)==1
 rid=matches[0]
 assert rid not in expected
 expected[rid]=value
assert len(expected)==6
observed={}
with (audit/"bindingdb_full_kd_source_provenance.jsonl").open(encoding="utf-8") as f:
 for line in f:
  r=json.loads(line)
  rid=str(r.get("BindingDB Reactant_set_id",""))
  if rid in expected:
   assert rid not in observed
   observed[rid]=r
assert set(observed)==set(expected)
for rid,pub in expected.items():
 src=observed[rid]; review=rows[rid]
 assert src["PMID"]=="29648813"
 assert src["Article DOI"].lower()=="10.1021/acs.jmedchem.8b00210"
 assert src["Ki (nM)"]=="" and src["IC50 (nM)"]==""
 assert math.isclose(float(src["Kd (nM)"]),pub["paper_spr_kd_nm"],rel_tol=1e-10)
 assert review["assay_key"]=="50002067_1"
 assert review["original_pmid"]=="29648813"
 assert "Table 3" in review["source_table_or_figure_locator"]
 assert "compound "+pub["paper_compound_id"]+";" in review["source_table_or_figure_locator"]
 assert "NOT verified" in review["source_table_or_figure_locator"]
 assert review["review_decision"]=="UNREVIEWED"
 assert not any(review[k] for k in ("original_record_mapping_verified","direct_equilibrium_kd_verified","physical_construct_verified","assay_conditions_compatible"))
print("PASS: 6 exact unique source Kd candidates, original DOI/PMID/endpoint verified; 0 approvals")
