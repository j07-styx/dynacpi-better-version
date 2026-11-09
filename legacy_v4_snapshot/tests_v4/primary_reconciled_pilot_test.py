"""Check real subset lineage, blank semantics, exclusions and original preservation."""
import collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"data/engineering_pilot_primary_reconciled_20261107"
rows=json.loads((out/"rows.json").read_text());m=json.loads((out/"manifest.json").read_text())
rawpath=ROOT/"data/engineering_pilot/rows.json"
assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==m["source_rows_sha256"]
assert hashlib.sha256((out/"rows.json").read_bytes()).hexdigest()==m["new_rows_sha256"]
assert len(rows)==len({(r["sequence"],r["compound"]) for r in rows})==650
original={(r["sequence"],r["compound"]):r for r in json.loads(rawpath.read_text())}
for r in rows:
    old=original[(r["sequence"],r["compound"])]
    assert r["y"]==old["y"] and r["split"]==old["split"]
    if r["primary_label_kind"]=="screen_negative_or_weak_binding":
        assert r["numeric_pKd"] is None and r["y"]==5
    else:assert r["numeric_pKd"]==r["y"]
assert set(r["primary_compound"] for r in rows).isdisjoint({"JNJ-28312141","INCB18424","PKC-412","BIBF-1120 (derivative)"})
assert collections.Counter(r["split"] for r in rows)=={"train":450,"val":200}
assert m["cross_split_kinase_groups"]
print("PASS 650-row lineage, blank-label semantics, four exclusions, original hashes, group-overlap detection")
