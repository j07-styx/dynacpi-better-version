"""Synthetic source-family freeze/approval gate tests; no real biology approved."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import require_approved_cohort, sha256, current_status
from dynacpi.approved_source_family_gate import SOURCE_MANIFEST_NAME
ROLES=("base_train", "base_validation", "adapter_train", "adapter_validation", "test")


class SourceFamilyScientificGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="synthetic_source_review_")
        self.root=Path(self.tmp.name)
        self.cohort=self.root/"synthetic_cohort.json"
        self.split=self.root/"synthetic_split.json"
        self.source=self.root/SOURCE_MANIFEST_NAME
        tasks=[{"task_id":f"fictional_{i}",
                "protein_component_id":f"fake_protein_comp_{i}",
                "source_review_record_id":f"fictional_review_{i}",
                "n_unique_compounds":20} for i in range(30)]
        self.cohort_data={
            "status":"APPROVED_EXPERIMENTAL_COHORT_SOURCE_REVIEWED",
            "review_policy_id":"SYNTHETIC_ONLY",
            "reviewer_identity":"AUTOMATED_TEST_NOT_PERSON",
            "review_evidence_manifest_sha256":"f"*64,
            "tasks":tasks,
        }
        self.roles={f"fictional_{i}":ROLES[i//6] for i in range(30)}
        self.source_data={
            "status":"SOURCE_FAMILY_GRAPH_FROZEN_AND_SOURCE_REVIEWED",
            "source_linkage_policy_id":"SYNTHETIC_ONLY",
            "reviewer_identity":"AUTOMATED_TEST_NOT_PERSON",
            "review_evidence_manifest_sha256":"f"*64,
            "freeze_timestamp_utc":"2026-11-08T00:00:00Z",
            "source_family_by_protein_component":{
                f"fake_protein_comp_{i}":hashlib.sha256(f"fictional_study_{i}".encode()).hexdigest()
                for i in range(30)},
        }
        self.split_data={
            "status":"FROZEN_PROTEIN_GROUP_SPLIT",
            "split_policy_id":"SYNTHETIC_ONLY",
            "frozen_by":"AUTOMATED_TEST_NOT_PERSON",
            "freeze_timestamp_utc":"2026-11-08T00:00:00Z",
            "task_roles":self.roles,
        }
        self.write_all()

    def tearDown(self):
        self.tmp.cleanup()

    def write_all(self):
        self.cohort.write_text(json.dumps(self.cohort_data),encoding="utf-8")
        self.source_data["cohort_sha256"]=sha256(self.cohort)
        self.source.write_text(json.dumps(self.source_data),encoding="utf-8")
        self.split_data["cohort_sha256"]=sha256(self.cohort)
        self.split_data["source_family_manifest_sha256"]=sha256(self.source)
        self.split.write_text(json.dumps(self.split_data),encoding="utf-8")

    def check_fail(self,phrase):
        with self.assertRaises(PermissionError) as cm:
            require_approved_cohort(self.cohort,self.split)
        self.assertIn(phrase,str(cm.exception))

    def test_synthetic_consistent_metadata_passes_artifact_gate_only(self):
        r=require_approved_cohort(self.cohort,self.split)
        self.assertIn("NOT_BIOLOGICAL_VALIDATION",r["status"])
        self.assertEqual(r["source_family_artifact_proof"]["source_family_count"],30)
        self.assertEqual(set(r["source_family_artifact_proof"]["role_source_family_counts"]),set(ROLES))

    def test_known_cross_role_source_shared_family_is_blocked(self):
        graph=self.source_data["source_family_by_protein_component"]
        graph["fake_protein_comp_7"]=graph["fake_protein_comp_1"]
        self.write_all()
        self.check_fail("DOI/PMID/patent-connected family assigned across roles")

    def test_missing_family_mapping_is_blocked(self):
        del self.source_data["source_family_by_protein_component"]["fake_protein_comp_4"]
        self.write_all()
        self.check_fail("must cover exactly all approved protein components")

    def test_extra_family_mapping_is_blocked(self):
        self.source_data["source_family_by_protein_component"]["fake_non_cohort"]="a"*64
        self.write_all()
        self.check_fail("must cover exactly all approved protein components")

    def test_duplicate_json_manifest_keys_fail_closed(self):
        raw=json.dumps(self.source_data)
        raw=raw.replace('"status":', '"status":"SOURCE_FAMILY_GRAPH_FROZEN_AND_SOURCE_REVIEWED","status":',1)
        self.source.write_text(raw)
        self.split_data["source_family_manifest_sha256"]=sha256(self.source)
        self.split.write_text(json.dumps(self.split_data))
        self.check_fail("source-family evidence JSON invalid")

    def test_invalid_source_family_hash_rejected(self):
        self.source_data["source_family_by_protein_component"]["fake_protein_comp_4"]="not_a_hash"
        self.write_all()
        self.check_fail("malformed source-family identity")

    def test_insufficient_independent_families_rejected(self):
        graph=self.source_data["source_family_by_protein_component"]
        # Combine all six distinct families in one role; still same role, but
        # this future evaluation lacks three independent families in that role.
        for i in range(6):
            graph[f"fake_protein_comp_{i}"]=graph["fake_protein_comp_0"]
        self.write_all()
        self.check_fail("fewer than three independent source families")

    def test_forged_review_status_rejected(self):
        self.source_data["status"]="UNREVIEWED_PRELIMINARY_METADATA"
        self.write_all()
        self.check_fail("missing reviewed frozen source-family statuses")

    def test_missing_source_manifest_rejected(self):
        self.source.unlink()
        self.check_fail("frozen source-family map required")

    def test_tamper_after_frozen_checksum_fails_closed(self):
        with self.source.open("ab") as f:
            f.write(b" ")
        self.check_fail("not bound to source-family graph checksum")

    def test_mismatched_approved_cohort_hash_in_source_rejected(self):
        self.write_all()
        self.source_data["cohort_sha256"]="0"*64
        self.source.write_text(json.dumps(self.source_data))
        self.split_data["source_family_manifest_sha256"]=sha256(self.source)
        self.split.write_text(json.dumps(self.split_data))
        self.check_fail("source-family graph does not match approved cohort checksum")

    def test_review_evidence_digest_required(self):
        self.source_data["review_evidence_manifest_sha256"]="FAKE"
        self.write_all()
        self.check_fail("source-family review evidence digest invalid")

    def test_timezone_aware_freeze_date_required(self):
        self.source_data["freeze_timestamp_utc"]="2026-11-08T00:00:00"
        self.write_all()
        self.check_fail("source-family freeze timestamp must have a timezone")

    def test_source_family_count_deduplicated_within_training_role(self):
        graph=self.source_data["source_family_by_protein_component"]
        graph["fake_protein_comp_1"]=graph["fake_protein_comp_0"]
        self.write_all()
        proof=require_approved_cohort(self.cohort,self.split)["source_family_artifact_proof"]
        self.assertEqual(proof["source_family_count"],29)
        self.assertEqual(proof["role_source_family_counts"]["base_train"],5)

    def test_real_unreviewed_gate_still_blocked(self):
        self.assertEqual(current_status(ROOT)["status"],"BLOCKED_UNAPPROVED_NOT_TRAINING_READY")


if __name__=="__main__":
    unittest.main(verbosity=2)
