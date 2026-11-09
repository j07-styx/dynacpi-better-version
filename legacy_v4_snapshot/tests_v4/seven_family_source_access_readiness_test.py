"""Synthetic tests for seven-family source-access readiness projection.

Purpose: prevent cross-role source identity or approval leakage in public output,
and prevent an abstract/method locator from becoming a validated affinity row.
No raw BindingDB observations or reserved-test outcomes are read.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/build_seven_family_source_access_ledger_dev.py"
spec=importlib.util.spec_from_file_location("seven_family_source_access", SCRIPT)
model=importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def dummy_queue():
    families=[]
    for i,marker in enumerate(model.PRIMARY_SOURCE_NOTES):
        family_id=hashlib.sha256(f"fake-family-{i}".encode()).hexdigest()
        families.append({
            "source_family_sha256": family_id,
            "affected_provisional_role_component_counts": {"train": 1, "test": 1},
            "source_identifier_examples_only_not_exhaustive": [marker],
            "preexisting_ranked_source_leads": [],
            "review_status": "UNREVIEWED_METADATA_CONFLICT_NOT_APPROVED",
        })
    return {"status": "PRIVATE_HUMAN_REVIEW_QUEUE_UNAPPROVED", "families":families}


class SourceAccessEvidenceTests(unittest.TestCase):
    def test_full_synthetic_coverage_with_zero_approvals(self):
        queue=dummy_queue()
        local,public=model.classify_queue(queue)
        self.assertEqual(len(local),7)
        self.assertEqual(public["families_with_one_source_access_evidence_anchor"],7)
        self.assertEqual(public["families_with_independently_verified_original_records"],0)
        self.assertEqual(public["families_scientifically_approved"],0)
        self.assertEqual(set(public["evidence_access_tier_counts"]),model.TIERS)
        self.assertEqual(sum(public["evidence_access_tier_counts"].values()),7)
        self.assertTrue(all(not row[field] for row in local for field in model.REVIEW_FIELDS))

    def test_public_report_does_not_expose_identity_or_role_assignments(self):
        local,public=model.classify_queue(dummy_queue())
        output=json.dumps(public)
        self.assertNotIn("fake-family",output)
        self.assertNotIn("source_family_sha256",output)
        self.assertNotIn("provisional_role_component_counts",output)
        self.assertNotIn("doi:",output)
        self.assertNotIn("patent:",output)
        self.assertFalse(any(row["source_family_sha256"] in output for row in local))

    def test_source_access_tier_not_equivalent_to_approved_record(self):
        local,_=model.classify_queue(dummy_queue())
        self.assertEqual({r["status"] for r in local},
                         {"UNREVIEWED_NO_SOURCE_ROW_APPROVAL"})
        self.assertEqual({r["identifier_coverage_note"] for r in local},
                         {"One example only; complete source linkage NOT reviewed"})

    def test_missing_source_anchor_rejected(self):
        queue=dummy_queue()
        queue["families"][0]["source_identifier_examples_only_not_exhaustive"]=[]
        with self.assertRaisesRegex(ValueError, "One independently recorded evidence anchor"):
            model.classify_queue(queue)

    def test_ambiguous_multiple_anchors_rejected(self):
        queue=dummy_queue()
        queue["families"][0]["source_identifier_examples_only_not_exhaustive"].append(
            list(model.PRIMARY_SOURCE_NOTES)[1])
        with self.assertRaisesRegex(ValueError, "One independently recorded evidence anchor"):
            model.classify_queue(queue)

    def test_duplicate_source_family_rejected(self):
        queue=dummy_queue()
        queue["families"][1]["source_family_sha256"]=queue["families"][0]["source_family_sha256"]
        with self.assertRaisesRegex(ValueError, "Invalid or duplicated"):
            model.classify_queue(queue)

    def test_unreviewed_status_is_required(self):
        queue=dummy_queue()
        queue["families"][0]["review_status"]="APPROVE"
        with self.assertRaises(PermissionError):
            model.classify_queue(queue)
        queue=dummy_queue()
        queue["status"]="SOURCE_APPROVED"
        with self.assertRaises(PermissionError):
            model.classify_queue(queue)

    def test_real_archived_public_result_has_no_role_leakage(self):
        if not (model.INPUT.exists() and model.LOCAL_OUT.exists() and model.PUBLIC_OUT.exists()):
            self.skipTest("Expected private worklist not mounted")
        queue=json.loads(model.INPUT.read_text(encoding="utf-8"))
        local,public=model.classify_queue(queue)
        self.assertEqual(public,json.loads(model.PUBLIC_OUT.read_text(encoding="utf-8")))
        actual=json.loads(model.LOCAL_OUT.read_text(encoding="utf-8"))
        self.assertEqual(actual["families"],local)
        self.assertEqual(actual["input_private_queue_sha256"],
                         hashlib.sha256(model.INPUT.read_bytes()).hexdigest())
        self.assertEqual(public["scientific_gate"],
                         "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
