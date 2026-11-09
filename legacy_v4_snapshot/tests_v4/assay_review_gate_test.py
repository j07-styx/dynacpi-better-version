"""Fail-closed test of manual review approval controls; no true approvals fabricated."""
from datetime import date
from pathlib import Path
import copy
import json
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"scripts"))
from validate_assay_reviews import BASE,ENRICHED,read_rows,evaluate_reviews
base=read_rows(BASE)
reviewed=read_rows(ENRICHED)
assert len(base)==len(reviewed)==30
r=evaluate_reviews(base,reviewed)
assert r["counts"]=={"UNREVIEWED":30} and r["approved_tasks"]==0
unsafe=copy.deepcopy(reviewed)
unsafe[0]["curation_decision"]="APPROVE"
try:
    evaluate_reviews(base,unsafe)
except ValueError:pass
else:raise AssertionError("Unsubstantiated approval accepted")
tampered=copy.deepcopy(reviewed)
tampered[0]["assay_name"]="changed source annotation"
try:
    evaluate_reviews(base,tampered)
except ValueError as e:
    assert "Nonreview source metadata changed" in str(e)
else:
    raise AssertionError("Tampered source metadata accepted")
approved_example=copy.deepcopy(reviewed)
x=approved_example[0]
x["curation_decision"]="APPROVE"
x["construct_verified"]=x["assay_semantics_verified"]=x["records_compatible"]="YES"
x["evidence_reference"]="SYNTHETIC_TEST_ONLY"
x["reviewer_identity"]="UNIT_TEST_ONLY"
x["review_date"]=date.today().isoformat()
x["curation_reason"]="Synthetic unit-test record only, not scientific certification."
result=evaluate_reviews(base,approved_example)
assert result["approved_tasks"]==1 and result["counts"]["UNREVIEWED"]==29
# No real review rows or approval manifests were modified in this test.
report={"status":"PASS","current_unreviewed":30,"current_approved":0,
        "controls_tested":["reject unsubstantiated approval","reject source metadata tampering",
                           "test gated approval schema without changing real review sheet"],
        "scope":"Software validation only, not any assay verification"}
(root/"data/audit/bindingdb_assay_review_gate_test.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
