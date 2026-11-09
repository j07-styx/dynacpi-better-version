"""Regression tests: old TransferCPI training MUST block before GPU/labels."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dynacpi.provisional_transfer_training_lock import (
    ProvisionalRetrainingBlocked,
    classify_retraining_request,
    refuse_provisional_transfer_retraining,
)
from scripts.audit_provisional_bindingdb_source_split_dev import generate


class TransferTrainingLockTests(unittest.TestCase):
    def test_real_metadata_prevents_new_run_without_reading_labels(self):
        with self.assertRaisesRegex(ProvisionalRetrainingBlocked, "PUBLICATION_SOURCE_LINKS_CROSS_PROVISIONAL_ROLES"):
            refuse_provisional_transfer_retraining(ROOT)

    def test_fake_clean_disjoint_and_approved_metadata_still_retired(self):
        clean = {"families_crossing_two_or_more_roles": 0, "publication_family_disjoint": True}
        status = {"status": "ARTIFACT_GATE_PASSED_NOT_BIOLOGICAL_VALIDATION"}
        self.assertEqual(
            classify_retraining_request(clean, status),
            "RETIRED_PROVISIONAL_PROTOCOL_REQUIRES_NEW_APPROVED_COHORT_AND_SPLIT",
        )

    def test_fake_clean_disjoint_without_cohort_approval_blocked(self):
        clean = {"families_crossing_two_or_more_roles": 0, "publication_family_disjoint": True}
        self.assertEqual(
            classify_retraining_request(clean, {"status": "BLOCKED_UNAPPROVED_NOT_TRAINING_READY"}),
            "UNAPPROVED_EXPERIMENTAL_COHORT",
        )

    def test_missing_or_changed_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ProvisionalRetrainingBlocked):
                refuse_provisional_transfer_retraining(root)
            for p in ("research/bindingdb_publication_connected_protein_review_preflight_20261108.json",
                      "research/bindingdb_frozen_component_split_manifest_dev.json",
                      "research/bindingdb_provisional_role_source_overlap_audit_20261108.json"):
                dest = root / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((ROOT / p).read_bytes())
            with (root/"research/bindingdb_frozen_component_split_manifest_dev.json").open("ab") as f:
                f.write(b" ")
            with self.assertRaisesRegex(ProvisionalRetrainingBlocked, "fail closed"):
                refuse_provisional_transfer_retraining(root)

    def test_metadata_gate_is_before_loader_and_gpu_lease(self):
        entry = (ROOT/"scripts/train_transfer_cpi_dev.py").read_text(encoding="utf-8")
        after_main = entry.split("def main(args):", 1)[1]
        self.assertLess(after_main.index("refuse_provisional_transfer_retraining(ROOT)"),
                        after_main.index("load_development()"))
        at_entry = entry.split('if __name__=="__main__":', 1)[1]
        self.assertLess(at_entry.index("refuse_provisional_transfer_retraining(ROOT)"),
                        at_entry.index("with gpu_lease(ROOT)"))

    def test_real_script_rejects_before_any_new_output(self):
        # Do not allocate GPU or consume labels; source audit rejects at entrypoint.
        p = subprocess.run(
            [sys.executable, "-B", str(ROOT/"scripts/train_transfer_cpi_dev.py"),
             "--model", "transfer", "--seed", "997319", "--epochs", "1"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("PUBLICATION_SOURCE_LINKS_CROSS_PROVISIONAL_ROLES", p.stderr)
        self.assertFalse((ROOT/"research/transfer_cpi_dev_20261108/transfer_seed997319").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
