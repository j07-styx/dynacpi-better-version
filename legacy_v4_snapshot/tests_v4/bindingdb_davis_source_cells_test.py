"""Regression checks: qualifiers and blanks must not become exact measurements."""
import sys,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from reconcile_bindingdb_davis_source_dev import compare,canonical
assert compare("",10000)=="PRIMARY_BLANK_NOT_NUMERIC"
assert compare("10",">10")=="QUALIFIED_OR_NONNUMERIC_RECORD"
assert compare("10","<10")=="QUALIFIED_OR_NONNUMERIC_RECORD"
assert compare("10","10.0")=="EXACT_NUMERIC_MATCH"
assert compare("10","11")=="NUMERIC_MISMATCH"
for value in ("NaN","Infinity","-1","0"):
    assert compare("10",value)=="INVALID_NUMERIC"
assert canonical("C[C@H](O)F")!=canonical("C[C@@H](O)F")
p=ROOT/"data/audit/davis_primary_20261107/bindingdb_source_cells_LOCAL_PRIVATE.json"
rows=json.loads(p.read_text())
report=json.loads((ROOT/"research/bindingdb_davis_source_cells_20261107.json").read_text())
assert len(rows)==63 and len({r["reactant_set_id"] for r in rows})==63
assert dict(Counter(r["status"] for r in rows))==report["counts"]
for r in rows:
    assert r["review_decision"]=="UNREVIEWED" and not r["physical_construct_verified"]
    if r["status"]=="EXACT_NUMERIC_MATCH":
        assert compare(r["primary_cell"],r["recorded_kd_nm"])=="EXACT_NUMERIC_MATCH"
        assert r["table_locator"] and r["primary_compound"]
print("PASS: blanks, inequalities, nonfinite values, stereochemistry, 63 source records and unchanged approval semantics")
