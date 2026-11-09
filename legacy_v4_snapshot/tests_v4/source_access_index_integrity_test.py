"""Audit machine-generated primary-source access index WITHOUT approving assays.

Tests immutable real 30-task queue, structured status, section-IDs-only output,
safety of XML handling and both patent/publication missing-primary-source cases.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/audit_primary_source_access_v4.py"
spec=importlib.util.spec_from_file_location("source_access_audit",SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
queue=mod.QUEUE
before=queue.read_bytes()
rows=mod.read_queue()
assert len(rows)==30
assert all(x["curation_decision"]=="UNREVIEWED" for x in rows)
report=json.loads((ROOT/"research/source_access_v4/source_access_index.json").read_text(encoding="utf-8"))
assert report["status"]=="AUTOMATED_SOURCE_INDEX_ONLY_NOT_BIOLOGICAL_VALIDATION"
assert report["scientifically_approved_tasks"]==0
assert report["human_review_queue_unchanged"] is True
assert report["task_count"]==30 and report["unique_publications"]==20
assert report["review_file_sha256"]==hashlib.sha256(before).hexdigest()
assert len(report["tasks"])==30 and len(report["publications"])==20
assert sum(report["source_status_counts"].values())==30
assert sum(report["publication_access_counts"].values())==20
assert {x["task_id"] for x in report["tasks"]}=={x["task_id"] for x in rows}
for item in report["tasks"]:
    assert item["current_review_status"]=="UNREVIEWED"
    assert all(item[key] is False for key in
               ("verified_construct","verified_kd_semantics","verified_assay_compatibility"))
for item in report["publications"]:
    assert item["approval_status"]=="NOT_APPROVED_SOURCE_METADATA_ONLY"
    assert item["metadata_access"] in ("EXACT_PMID_MATCH","PMID_EXACT_MATCH_MISSING","FETCH_FAILED")
    assert not any(key in item for key in ("raw_fulltext","fulltext_xml","abstract_body","raw_paragraphs"))
    for section in item.get("section_headings_and_keyword_counts",[]):
        assert set(section)=={"section_id","section_heading","method_heading_hint","keyword_hit_counts"}
        assert len(section["section_heading"])<=110
xml=b"<article><body><sec id='methods'><title>Methods</title><p>Recombinant protein was measured with Kd.</p></sec></body></article>"
index=mod.extract_section_signals(xml)
assert index["relevant_sections_total"]==1
assert index["section_headings_and_keyword_counts"][0]["section_id"]=="methods"
assert index["section_headings_and_keyword_counts"][0]["keyword_hit_counts"]["KD"]==1
assert "Recombinant protein was measured" not in json.dumps(index)
for payload in (
    b"<!DOCTYPE article [<!ENTITY injected 'not allowed'>]><article/>",
    b"<!ENTITY injected 'not allowed'><article/>",
):
    try:mod.extract_section_signals(payload)
    except ValueError:pass
    else:raise AssertionError("Unsafe article XML DTD/entity not rejected")
assert before==queue.read_bytes(), "Review CSV was modified by integrity test"
worklist=json.loads((ROOT/"research/source_access_v4/worklist_manifest.json").read_text(encoding="utf-8"))
assert worklist["status"]=="READ_ONLY_SOURCE_WORKLIST_CREATED_NOT_APPROVED"
assert worklist["task_count"]==30 and worklist["scientific_approvals"]==0
assert worklist["canonical_review_file_unchanged"] is True
assert (ROOT/"research/source_access_v4/REVIEW_ACTIONS.md").is_file()
from sys import path as _path
_path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"
out={"status":"PASS_ENGINEERING_SOURCE_INDEX_ONLY",
     "review_queue_unchanged":True,"unique_source_pmids":20,
     "open_access_xml_indexed":report["publication_access_counts"].get("OPEN_ACCESS_SECTION_INDEXED",0),
     "scientifically_approved_tasks":0,
     "synthetic_xml_unsafe_entities_rejected":True,
     "raw_article_text_not_exported":True,
     "science_gate":"BLOCKED"}
(ROOT/"data/audit/source_access_index_integrity.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
