"""Synthetic-only role leakage tests plus read-only real-source summary consistency."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_provisional_bindingdb_source_split_dev.py"
spec = importlib.util.spec_from_file_location("publication_role_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class MetadataOnlyRoleAuditTests(unittest.TestCase):
    def test_source_family_crosses_train_and_holdout(self):
        roles = {"train": ["p1", "p2"], "inner_dev": ["p3"], "test": ["p4"]}
        families = {"p1": "fA", "p2": "fB", "p3": "fA", "p4": "fB"}
        x = audit.assess_roles(roles, families)
        self.assertEqual(x["families_crossing_two_or_more_roles"], 2)
        self.assertEqual(x["components_in_cross_role_families"], 4)
        self.assertEqual(x["pairwise_overlap"]["train_vs_inner_dev"]["families_in_both"], 1)
        self.assertEqual(x["pairwise_overlap"]["train_vs_test"]["families_in_both"], 1)
        self.assertFalse(x["publication_family_disjoint"])

    def test_transitive_three_role_family_once_not_tripled(self):
        roles = {"train": ["p1"], "inner_dev": ["p2"], "test": ["p3"]}
        x = audit.assess_roles(roles, {"p1": "f", "p2": "f", "p3": "f"})
        self.assertEqual(x["families_crossing_two_or_more_roles"], 1)
        self.assertEqual(x["components_in_cross_role_families"], 3)
        self.assertEqual(x["cross_role_pattern_counts"]["train+inner_dev+test"], 1)
        self.assertEqual(sum(y["families_in_both"] for y in x["pairwise_overlap"].values()), 3)

    def test_disjoint_families(self):
        roles = {"train": ["p1"], "inner_dev": ["p2"], "test": ["p3"]}
        x = audit.assess_roles(roles, {"p1": "a", "p2": "b", "p3": "c"})
        self.assertTrue(x["publication_family_disjoint"])
        self.assertEqual(x["families_crossing_two_or_more_roles"], 0)
        self.assertEqual(x["readiness"], "NO_METADATA_CROSS_ROLE_SOURCE_LINKS_DETECTED")

    def test_fail_closed_missing_components_and_role_reuse(self):
        roles = {"train": ["p1"], "inner_dev": ["p2"], "test": ["p3"]}
        with self.assertRaises(ValueError):
            audit.assess_roles(roles, {"p1": "x", "p2": "x"})
        with self.assertRaises(ValueError):
            audit.assess_roles({"train": ["p1"], "inner_dev": ["p1"], "test": []}, {"p1": "x"})
        with self.assertRaises(ValueError):
            audit.assess_roles({"train": ["p1", "p1"], "inner_dev": [], "test": []}, {"p1": "x"})

    def test_metadata_integrity_gate_rejects_fake_review(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            a = parent/"families.json"
            b = parent/"split.json"
            family = {"status": "SCIENTIFICALLY_APPROVED",
                      "scientific_gate": "APPROVED",
                      "linkage": {"source_family_by_protein_component": {"p1": "f"}}}
            split = {"train_components": ["p1"], "inner_dev_components": [],
                     "test_components": []}
            a.write_text(json.dumps(family))
            b.write_text(json.dumps(split))
            with self.assertRaises(ValueError):
                audit.generate(a,b)

    def test_real_metadata_result_reproducible_without_label_inputs(self):
        if not (audit.FAMILY_PATH.exists() and audit.SPLIT_PATH.exists() and audit.OUT.exists()):
            self.skipTest("No local audit inputs or saved result; synthetic checks still run")
        saved = json.loads(audit.OUT.read_text(encoding="utf-8"))
        recomputed = audit.generate()
        self.assertEqual(saved, recomputed)
        self.assertEqual(saved["families_crossing_two_or_more_roles"], 7)
        self.assertEqual(saved["components_in_cross_role_families"], 25)
        self.assertEqual(saved["scientific_gate"], "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")

    def test_private_reviewer_queue_only_contains_cross_role_family_members(self):
        script = ROOT / "scripts/build_provisional_source_overlap_private_review_dev.py"
        source_spec = importlib.util.spec_from_file_location("review_queue", script)
        module = importlib.util.module_from_spec(source_spec)
        source_spec.loader.exec_module(module)
        roles = {"train": ["a", "b"], "inner_dev": ["c"], "test": ["d"]}
        mapping = {"a": "f1", "b": "f2", "c": "f1", "d": "f3"}
        records = module.review_rows(mapping, roles)
        self.assertEqual(len(records), 2)
        self.assertEqual({r["provisional_role"] for r in records}, {"train", "inner_dev"})
        self.assertEqual({r["review_status"] for r in records}, {"UNREVIEWED_METADATA_CONFLICT"})
        self.assertTrue(all(set(r) == set(module.FIELDS) for r in records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
