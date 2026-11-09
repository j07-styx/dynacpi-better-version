"""Synthetic-only exact Crossref DOI exception checks and real privacy invariants.

No affinity measurements or reserved test labels are accessed. Publication
identifiers reside ONLY in private data/audit outputs; public findings are
aggregates and cannot approve any protein construct.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.reconcile_seven_family_bibliographic_aliases_v4 import (
    PUBLIC_OUTPUT, PRIVATE_OUTPUT,PRIVATE_INPUT,
)
from scripts.verify_seven_family_crossref_doi_exception_v4 import (
    PRIVATE_NEW,PUBLIC_NEW,exact_crossref_metadata,unresolved_dois
)


def fictional_record():
    return {
        "status":"PRIVATE_RECIPROCAL_METADATA_SOURCE_REVIEW_NO_ASSAY_APPROVAL",
        "families":[{"source_family_sha256":"fictional-family",
                     "DOI_records_metadata_only":[
                         {"original_doi":"10.9999/fakejournal",
                          "status":"NOT_FOUND_OR_AMBIGUOUS_IN_MED"}]}],
    }


def fictional_public():
    return {
        "status":"SEVEN_FAMILY_BIBLIOGRAPHIC_RECONCILIATION_ONLY_NO_ASSAY_APPROVAL",
        "unresolved_DOI_identifiers_in_MED":1,
    }


class CrossrefBibliographyTests(unittest.TestCase):
    def test_exact_doi_and_article_title_needed(self):
        response={"message":{"DOI":"10.9999/FAKEJOURNAL",
                   "type":"journal-article","title":["Fabricated Paper"],
                   "publisher":"Fictional Publisher"}}
        result=exact_crossref_metadata("10.9999/fakejournal",response)
        self.assertTrue(result["exact_crossref_doi_match"])
        self.assertEqual(result["source_kind"],"journal-article")
        self.assertIn("NO_ASSAY",result["review_stage"])

    def test_wrong_doi_cannot_be_promoted(self):
        with self.assertRaisesRegex(ValueError,"exactly match"):
            exact_crossref_metadata("10.9999/a",{
                "message":{"DOI":"10.9999/b","title":["Fictional"]}})

    def test_missing_article_title_cannot_be_promoted(self):
        with self.assertRaisesRegex(ValueError,"no article title"):
            exact_crossref_metadata("10.9999/a",{
                "message":{"DOI":"10.9999/a","title":[]}})

    def test_missing_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_crossref_metadata("10.9999/a",{"message":None})

    def test_one_unresolved_private_doi_exact_identification(self):
        private=fction=fictional_record()
        public=fictional_public()
        identified=unresolved_dois(private,public)
        self.assertEqual(identified,[("fictional-family","10.9999/fakejournal")])

    def test_false_approval_status_rejected(self):
        private=fictional_record()
        private["status"]="APPROVED_FOR_TRAINING"
        with self.assertRaises(PermissionError):
            unresolved_dois(private,fictional_public())

    def test_wrong_unresolved_count_rejected(self):
        public=fictional_public()
        public["unresolved_DOI_identifiers_in_MED"]=2
        with self.assertRaises(ValueError):
            unresolved_dois(fictional_record(),public)

    def test_private_source_metadata_unchanged_from_replay(self):
        original=json.loads(PRIVATE_INPUT.read_text(encoding="utf-8"))
        self.assertEqual(original["status"],
            "CONFIDENTIAL_COMPLETE_ELIGIBLE_SOURCE_IDS_NO_APPROVAL")
        self.assertEqual(len(original["families"]),7)
        self.assertTrue(all(not f["exact_assayed_protein_construct_verified"]
                            for f in original["families"]))

    def test_published_public_aggregates_contain_no_source_id_linkage(self):
        public=json.loads(PUBLIC_OUTPUT.read_text(encoding="utf-8"))
        cross=json.loads(PUBLIC_NEW.read_text(encoding="utf-8"))
        private=json.loads(PRIVATE_OUTPUT.read_text(encoding="utf-8"))
        detail=json.loads(PRIVATE_NEW.read_text(encoding="utf-8"))
        self.assertEqual(public["reciprocal_DOI_PMID_aliases_confirmed_as_same_publication"],72)
        self.assertEqual(public["pmid_identifiers_with_exact_MED_metadata"],90)
        self.assertEqual(public["doi_identifiers_with_exact_MED_metadata"],73)
        self.assertEqual(cross["exact_crossref_DOI_identifiers_verified"],1)
        self.assertEqual(cross["bindingdb_assay_rows_verified"],0)
        self.assertFalse(detail["scientific_approval"])
        self.assertFalse(detail["physical_construct_verified"])
        combined=json.dumps((public,cross))
        self.assertNotIn(detail["original_doi"],combined)
        self.assertNotIn(detail["source_family_sha256"],combined)
        for family in private["families"]:
            self.assertNotIn(family["source_family_sha256"],combined)
        self.assertNotIn("eligible_task_ids",combined)
        self.assertNotIn("provisional_role",combined)

    def test_all_220_original_source_ids_absent_from_public_artifacts(self):
        # Compare only private SOURCE INDEX strings with PUBLIC deliverables.
        # Never print an offending protected source ID into test logs.
        secret=json.loads(PRIVATE_INPUT.read_text(encoding="utf-8"))
        all_ids={s["original_source_identifier"]
                 for family in secret["families"]
                 for s in family["all_original_source_identifiers"]}
        family_ids={f["source_family_sha256"] for f in secret["families"]}
        component_ids={cid for family in secret["families"]
                       for cid in family["all_original_protein_component_ids"]}
        private_task_ids={tid for family in secret["families"]
                          for tid in family[
                            "technically_eligible_tasks_missing_all_DOI_PMID_patent"]}
        self.assertEqual(len(all_ids),220)
        paths=[
            ROOT/"scripts/reconcile_seven_family_bibliographic_aliases_v4.py",
            ROOT/"scripts/verify_seven_family_crossref_doi_exception_v4.py",
            ROOT/"tests_v4/seven_family_bibliographic_alias_reconciliation_test.py",
            ROOT/"tests_v4/seven_family_crossref_exception_test.py",
            ROOT/"research/bindingdb_seven_family_bibliographic_alias_20261109.json",
            ROOT/"research/bindingdb_seven_family_crossref_exception_20261109.json",
            ROOT/"research/BINDINGDB_SEVEN_FAMILY_BIBLIOGRAPHIC_IDENTITY_20261109.md",
        ]
        public_content="".join(p.read_text(encoding="utf-8") for p in paths)
        leaks=sum(v in public_content
                  for v in (*all_ids,*family_ids,*component_ids,*private_task_ids))
        self.assertEqual(leaks,0,"Public bibliography contains confidential source linkage")

    def test_alias_and_crossref_do_not_release_scientific_gate(self):
        public=json.loads(PUBLIC_OUTPUT.read_text(encoding="utf-8"))
        cross=json.loads(PUBLIC_NEW.read_text(encoding="utf-8"))
        self.assertEqual(public["scientific_gate"],
                         "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")
        self.assertEqual(cross["scientific_gate"],
                         "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")
        self.assertEqual(public["original_assay_record_crosswalks_verified"],0)
        self.assertEqual(cross["physical_constructs_verified"],0)


if __name__=="__main__":
    unittest.main(verbosity=2)
