"""Synthetic source-record review evidence tests; no actual approvals."""
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/build_record_source_evidence_worklist_v4.py"
spec=importlib.util.spec_from_file_location("source_evidence_worklist_v4",SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class EvidenceWorklistSynthetic(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.folder=self.root/"data/audit"
        self.folder.mkdir(parents=True)
        for filename in (module.QUEUE_NAME,module.GROUP_NAME,module.SOURCE_NAME):
            (self.folder/filename).write_text("synthetic-input-only")
        self.expected={}
        for n in range(2):
            rid=str(1000+n)
            self.expected[rid]={key:(
                "task_fake_"+str(n) if key=="task_id" else
                "assay_fake" if key=="assay_key" else
                rid if key=="reactant_set_id" else
                "component"+str(n) if key=="protein_component_id" else
                "a"*64 if key=="protein_sha256" else
                "11111111" if key=="original_pmid" else
                "10.1000/synthetic" if key=="original_doi" else
                "synthetic source text" if key in ("original_curator_source","original_source_organism") else
                "b"*64 if key=="source_row_metadata_sha256" else
                "") for key in module.ANCHORS}
        digest=module.sha(self.folder/module.QUEUE_NAME)
        self.patch=patch.object(module,"expected_rows",return_value=(self.expected,digest,
            module.sha(self.folder/module.GROUP_NAME),module.sha(self.folder/module.SOURCE_NAME)))
        self.patch.start()
        module.validate(self.root)
        self.file=self.folder/module.WORKLIST_NAME

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def rows(self):
        return module.read_csv(self.file)

    def write_rows(self,rows):
        with self.file.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=module.ALL_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def complete_one(self,rows):
        row=rows[0]
        row.update({
            "review_decision":"APPROVE",
            "source_method_locator":"Synthetic paper methods section 3.1",
            "source_table_or_figure_locator":"Synthetic Table 2, row X",
            "physically_assayed_target":"Synthetic protein chain",
            "physical_construct_description":"Recombinant synthetic residues 1-100",
            "physical_construct_sequence_sha256":"c"*64,
            "source_assay_method":"synthetic direct equilibrium titration",
            "source_record_endpoint":"Kd",
            "source_value_qualifier":"EXACT",
            "original_record_mapping_verified":"YES",
            "direct_equilibrium_kd_verified":"YES",
            "physical_construct_verified":"YES",
            "assay_conditions_compatible":"YES",
            "reviewer_identity":"FAKE_SYNTHETIC_REVIEWER",
            "review_date":"2026-11-05",
            "review_notes":"Synthetic-only complete review declaration for validation testing, not a scientific assertion."
        })
        return rows

    def test_initial_private_scaffold_is_unreviewed(self):
        result=module.validate(self.root)
        self.assertEqual(result["source_record_count"],2)
        self.assertEqual(result["self_reported_review_state_counts"]["UNREVIEWED"],2)
        self.assertEqual(result["source_scientifically_approved_task_count"],0)
        self.assertFalse(result["main_scientific_training_authorized"])
        self.assertFalse(result["local_private_worklist_created_this_run"])
        self.assertFalse(result["contains_individual_affinity_values"])
        content=self.file.read_text()
        self.assertNotIn("median_pKd",content)
        self.assertNotIn("smiles",content.lower())

    def test_unreviewed_positive_declaration_is_rejected(self):
        rows=self.rows()
        rows[0]["physical_construct_verified"]="YES"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError,"Unreviewed row asserts"):
            module.validate(self.root)

    def test_approval_without_source_locators_is_rejected(self):
        rows=self.complete_one(self.rows())
        rows[0]["source_table_or_figure_locator"]=""
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError,"missing external evidence"):
            module.validate(self.root)

    def test_wrong_endpoint_is_rejected(self):
        rows=self.complete_one(self.rows())
        rows[0]["source_record_endpoint"]="IC50"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError,"experimental Kd"):
            module.validate(self.root)

    def test_qualified_kd_is_rejected_as_exact(self):
        rows=self.complete_one(self.rows())
        rows[0]["source_value_qualifier"]="GREATER_THAN"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError,"Censored/qualified"):
            module.validate(self.root)

    def test_synthetic_complete_declaration_never_certifies_science(self):
        rows=self.complete_one(self.rows())
        self.write_rows(rows)
        result=module.validate(self.root)
        self.assertEqual(result["self_reported_review_state_counts"]["APPROVE"],1)
        self.assertEqual(result["fully_declared_task_count"],1)
        self.assertEqual(result["source_scientifically_approved_task_count"],0)
        self.assertFalse(result["main_scientific_training_authorized"])
        self.assertFalse(result["human_primary_source_independently_audited"])
        self.assertEqual(self.rows()[0]["review_notes"],rows[0]["review_notes"])

    def test_source_metadata_tampering_is_rejected(self):
        rows=self.rows()
        rows[1]["original_doi"]="10.1000/altered"
        self.write_rows(rows)
        with self.assertRaisesRegex(PermissionError,"Immutable source metadata"):
            module.validate(self.root)

    def test_missing_and_duplicate_reactants_are_rejected(self):
        rows=self.rows()
        self.write_rows(rows[:1])
        with self.assertRaisesRegex(ValueError,"count mismatch"):
            module.validate(self.root)
        rows[1]["reactant_set_id"]=rows[0]["reactant_set_id"]
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError,"Duplicate/unknown"):
            module.validate(self.root)

    def test_reviewer_signed_unreviewed_row_is_rejected(self):
        rows=self.rows()
        rows[0]["reviewer_identity"]="fake"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError,"must not contain a reviewer signoff"):
            module.validate(self.root)

    def test_rejection_requires_rationale(self):
        rows=self.rows()
        rows[0]["review_decision"]="REJECT"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError,"missing reviewer"):
            module.validate(self.root)

if __name__=="__main__":
    unittest.main(verbosity=2)
