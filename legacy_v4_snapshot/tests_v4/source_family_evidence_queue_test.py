"""Privacy and integrity checks for precomputed seven-family review queue."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "scripts/build_source_family_evidence_queue_private_dev.py"
spec = importlib.util.spec_from_file_location("source_review_queue", SOURCE_SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class SevenFamilyReviewEvidenceTests(unittest.TestCase):
    def test_minimal_metadata_projection(self):
        preflight = {
            "linkage": {
                "family_members": {"f1": ["compx", "compy"], "f2": ["compz"]},
                "shared_identifier_examples": {
                    "doi:10.fake/review": {"component_prefixes": ["compx"]}
                },
            },
            "source_independent_review_leads": [{
                "source_family_id": "f1", "task_id": "t",
                "doi_candidates": ["10.fake/review"],
                "pmid_candidates": [], "patent_candidates": [],
                "target_names": ["Example target"],
                "uniprot_accessions": [],
                "explicit_physical_binding_method_keywords": ["SPR"],
                "potential_indirect_or_competition_keywords": [],
                "median_pKd": 8.2, "smiles_isomeric": "test-value-that-must-not-leak",
            }],
        }
        rows = [
            {"source_family_sha256": "f1", "protein_component_sha256": "compx",
             "provisional_role": "train"},
            {"source_family_sha256": "f1", "protein_component_sha256": "compy",
             "provisional_role": "test"},
        ]
        result = review.project_review_evidence(preflight, rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_identifier_examples_only_not_exhaustive"],
                         ["doi:10.fake/review"])
        self.assertEqual(result[0]["preexisting_ranked_source_leads"][0]["review_status"],
                         "UNREVIEWED_PRIMARY_SOURCE_REQUIRED")
        self.assertEqual(result[0]["review_status"], "UNREVIEWED_METADATA_CONFLICT_NOT_APPROVED")
        serialized = json.dumps(result).lower()
        self.assertNotIn("median_pkd", serialized)
        self.assertNotIn("smiles", serialized)
        self.assertNotIn("test-value-that-must-not-leak", serialized)

    def test_cannot_add_unmatched_component_or_single_role_family(self):
        preflight = {
            "linkage": {"family_members": {"f1": ["compa", "compb"]},
                        "shared_identifier_examples": {}},
            "source_independent_review_leads": [],
        }
        with self.assertRaisesRegex(ValueError, "membership"):
            review.project_review_evidence(preflight, [
                {"source_family_sha256": "f1", "protein_component_sha256": "unknown",
                 "provisional_role": "train"},
                {"source_family_sha256": "f1", "protein_component_sha256": "compb",
                 "provisional_role": "test"},
            ])
        with self.assertRaisesRegex(ValueError, "non-conflicting"):
            review.project_review_evidence(preflight, [
                {"source_family_sha256": "f1", "protein_component_sha256": "compa",
                 "provisional_role": "train"},
            ])

    def test_saved_private_queue_retains_unapproved_state(self):
        if not review.OUT.exists():
            self.skipTest("Local confidential preflight not available")
        data = json.loads(review.OUT.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "PRIVATE_HUMAN_REVIEW_QUEUE_UNAPPROVED")
        self.assertEqual(len(data["families"]), 7)
        self.assertEqual(sum(len(f["preexisting_ranked_source_leads"]) for f in data["families"]), 5)
        self.assertEqual(sum(bool(f["source_identifier_examples_only_not_exhaustive"])
                             for f in data["families"]), 7)
        self.assertTrue(all(f["review_status"] == "UNREVIEWED_METADATA_CONFLICT_NOT_APPROVED"
                            for f in data["families"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
