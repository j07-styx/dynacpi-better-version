"""Offline integrity regression for source-method evidence. Does not access the network."""
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
q=root/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
p=root/"research/source_access_v4/open_access_method_evidence.json"
r=json.loads(p.read_text(encoding="utf-8"))
rows=list(csv.DictReader(q.open(encoding="utf-8-sig",newline="")))
assert r["status"]=="PASS_PRIMARY_METHODS_ONLY_NOT_RECORD_APPROVAL"
assert r["review_queue_sha256"]==hashlib.sha256(q.read_bytes()).hexdigest()
assert len(rows)==30 and all(x["curation_decision"]=="UNREVIEWED" for x in rows)
assert r["corroborated_assays"]==len(r["sources"])==2
assert r["source_records_in_assays"]==48
assert r["approved_records"]==0 and r["main_training_permitted"] is False
by={x["assay_key"]:x for x in rows}
for item in r["sources"]:
 assert by[item["assay_key"]]["pmids"]==item["pmid"]
 assert int(by[item["assay_key"]]["record_count"])==item["source_record_count"]
 assert item["source_level_method_corroborated"] is True
 assert item["record_level_compound_and_numeric_Kd_verified"] is False
 assert item["physical_construct_sequence_verified"] is False
 assert item["scientifically_approved"] is False
 assert len(item["source_xml_sha256"])==64
 assert all(item["primary_method_pattern_checks"].values())
print("PASS: 2 original-methods evidence entries, 48 source records; zero scientific approvals")
