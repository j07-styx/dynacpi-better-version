"""Fictional six-assay construct/endpoint method triage regression.

All assay names, target names and source IDs in synthetic cases are invented.
Actual private annotated task-to-source links are never embedded in Git.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import csv
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.triage_six_construct_endpoint_evidence_v4 import (
    assess,load_private_annotations,PRIVATE_OUT,PUBLIC_OUT,
    ORIGINAL_REVIEW_QUEUE,PRIOR_SIX,PRIVATE_ANNOTATIONS,
)


def fake_evidence():
    records=(48,20,19,24,18,18)
    relationship=("D","UNKNOWN","H","H","UNKNOWN","UNKNOWN")
    construct=("RECOMBINANT_PROTEIN","KINASE_DOMAIN_PANEL","KINASE_DOMAIN",
               "KINASE_DOMAIN","UNKNOWN_PHYSICAL_CONSTRUCT","UNKNOWN_PHYSICAL_CONSTRUCT")
    risk=("SOURCE_ROW_MAPPING_UNVERIFIED","DOMAIN_CONSTRUCT_UNVERIFIED",
          "HOMOLOGOUS_DOMAIN_CONSTRUCT_UNVERIFIED",
          "HOMOLOGOUS_DOMAIN_CONSTRUCT_UNVERIFIED",
          "ASSAY_PH_UNVERIFIED","ORIGINAL_SOURCE_NOT_IDENTIFIED")
    original=[]
    prior_records=[]
    annotations=[]
    for i,(n,relation,con,r) in enumerate(zip(records,relationship,construct,risk)):
        target=f"fictional_protein_{i}"
        assay=f"fictional_assay_{i}"
        task=f"fictional_task_{i}"
        original.append({
            "task_id":task,"target_names":target,
            "assay_key":assay,"record_count":str(n),
            "curation_decision":"UNREVIEWED",
        })
        prior_records.append({
            "task_id":task,"target":target,"assay_key":assay,
            "source_record_count":n,
        })
        annotations.append({
            "assay_key":assay,"target":target,"annotation":"FAKE_METHOD_METADATA",
            "target_relationship":relation,"construct_description":con,
            "endpoint_indicator":(
                "KD_TABLE_WITH_CENSORING_PRESENT" if i in (1,2,3)
                else "METHOD_NOT_CONFIRMED"
            ),
            "primary_source_status":"FICTIONAL_SOURCE_NOT_VERIFIED",
            "source_methods_risk":r,
            "registry_assay_reference":"NO_REAL_REGISTRY_RECORD",
        })
    for i in range(24):
        original.append({
            "task_id":f"unrelated_fake_task_{i}",
            "target_names":f"unrelated_fake_protein_{i}",
            "assay_key":f"unrelated_fake_assay_{i}",
            "record_count":"15","curation_decision":"UNREVIEWED",
        })
    prior={
        "status":"SIX_TASKS_EXTERNAL_PRIMARY_SOURCE_EVIDENCE_NO_APPROVAL",
        "source_approved_task_count":0,
        "candidate_record_count":147,
        "records":prior_records,
    }
    return original,prior,annotations


class SourceMethodTriageTests(unittest.TestCase):
    def test_six_fake_tasks_only_metadata_reject_all_approval(self):
        original,prior,annotations=fake_evidence()
        detailed,public=assess(annotations,original,prior)
        self.assertEqual(len(detailed),6)
        self.assertEqual(public["fixed_original_review_tasks"],6)
        self.assertEqual(public["original_candidate_source_record_count"],147)
        self.assertEqual(public["target_relationship_annotation_counts"],{
            "direct_registry":1,"homologous_registry":2,"unknown_or_unverified":3})
        self.assertEqual(public["scientifically_approved_tasks"],0)
        self.assertFalse(public["training_or_prospective_test_scoring_authorized"])

    def test_homologous_flag_cannot_be_changed_to_direct(self):
        original,prior,annotations=fake_evidence()
        bad=deepcopy(annotations)
        idx=next(i for i,v in enumerate(bad) if v["target_relationship"]=="H")
        bad[idx]["target_relationship"]="D"
        with self.assertRaisesRegex(ValueError,"categories changed"):
            assess(bad,original,prior)

    def test_homologous_target_cannot_lose_risk_flag(self):
        original,prior,annotations=fake_evidence()
        bad=deepcopy(annotations)
        idx=next(i for i,v in enumerate(bad) if v["target_relationship"]=="H")
        bad[idx]["source_methods_risk"]="FALSE_FULL_LENGTH_VALIDATION"
        with self.assertRaisesRegex(ValueError,"flagged as risk"):
            assess(bad,original,prior)

    def test_kinase_domain_cannot_lose_construct_risk(self):
        original,prior,annotations=fake_evidence()
        bad=deepcopy(annotations)
        bad[1]["source_methods_risk"]="ALL_VERIFIED"
        with self.assertRaisesRegex(ValueError,"cannot silently become full length"):
            assess(bad,original,prior)

    def test_changed_assay_key_rejected(self):
        original,prior,annotations=fake_evidence()
        bad=deepcopy(annotations)
        bad[0]["assay_key"]="tampered_fake_assay"
        with self.assertRaisesRegex(ValueError,"frozen original task"):
            assess(bad,original,prior)

    def test_changed_original_record_count_rejected(self):
        original,prior,annotations=fake_evidence()
        original[0]["record_count"]="10000"
        with self.assertRaisesRegex(ValueError,"witness mismatch"):
            assess(annotations,original,prior)

    def test_human_review_approval_cannot_be_rewritten(self):
        original,prior,annotations=fake_evidence()
        original[0]["curation_decision"]="APPROVE"
        with self.assertRaises(PermissionError):
            assess(annotations,original,prior)
        original,prior,annotations=fake_evidence()
        prior["source_approved_task_count"]=1
        with self.assertRaises(PermissionError):
            assess(annotations,original,prior)

    def test_cannot_sneak_an_approval_flag_into_evidence(self):
        original,prior,annotations=fake_evidence()
        bad=deepcopy(annotations)
        bad[0]["approved_for_training"]=True
        with self.assertRaisesRegex(ValueError,"Unexpected source evidence schema"):
            assess(bad,original,prior)

    def test_private_annotation_manifest_must_remain_unreviewed(self):
        temp={
            "status":"APPROVED_FOR_SCIENTIFIC_MODEL",
            "scientifically_approved_tasks":1,
            "annotations":[{} for _ in range(6)],
        }
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            f=Path(td)/"fake_private_manifest.json"
            f.write_text(json.dumps(temp))
            with self.assertRaises(PermissionError):
                load_private_annotations(f)

    def test_saved_private_method_tags_never_authorize_science(self):
        saved=json.loads(PRIVATE_OUT.read_text(encoding="utf-8"))
        public=json.loads(PUBLIC_OUT.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"],"PRIVATE_PUBLIC_SOURCE_METHOD_TRIAGE_NOT_APPROVED")
        self.assertEqual(len(saved["source_evidence_objects"]),6)
        for row in saved["source_evidence_objects"]:
            self.assertFalse(row["exact_source_record_Kd_crosswalk_verified"])
            self.assertFalse(row["individual_assayed_protein_construct_verified"])
            self.assertFalse(row["full_length_sequence_as_construct_authorized"])
            self.assertFalse(row["training_or_reserved_test_scoring_authorized"])
        self.assertEqual(public["source_metadata_indicating_kinase_domain_assay_tasks"],3)
        self.assertEqual(public["scientifically_approved_tasks"],0)

    def test_live_pure_replay_matches_saved_public_aggregate(self):
        annotations=load_private_annotations()
        with ORIGINAL_REVIEW_QUEUE.open(encoding="utf-8-sig",newline="") as stream:
            original=list(csv.DictReader(stream))
        prior=json.loads(PRIOR_SIX.read_text(encoding="utf-8"))
        _,public=assess(annotations,original,prior)
        saved=json.loads(PUBLIC_OUT.read_text(encoding="utf-8"))
        self.assertEqual(public,saved)

    def test_public_outputs_and_code_do_not_expose_protected_mapping(self):
        config=load_private_annotations()
        public_paths=[
            ROOT/"scripts/triage_six_construct_endpoint_evidence_v4.py",
            ROOT/"tests_v4/six_construct_endpoint_method_triage_test.py",
            PUBLIC_OUT,
            ROOT/"research/DYNACPI_SIX_SOURCE_METHOD_CONSTRUCT_RISKS_20261109.md",
        ]
        all_public="".join(p.read_text(encoding="utf-8") for p in public_paths)
        private_src=json.loads(PRIVATE_OUT.read_text(encoding="utf-8"))
        for v in private_src["source_evidence_objects"]:
            self.assertNotIn(v["assay_key"],all_public)
            self.assertNotIn(v["task_id"],all_public)
            self.assertNotIn(v["registry_assay_reference"],all_public)
            self.assertNotIn(v["target"],all_public)
        public_json=PUBLIC_OUT.read_text(encoding="utf-8")
        self.assertNotIn("original_source_identifier",public_json)
        self.assertNotIn("provisional_role",public_json)

    def test_existing_metadata_reviews_are_byte_identical(self):
        prior=json.loads(PRIOR_SIX.read_text(encoding="utf-8"))
        h=sha256(ORIGINAL_REVIEW_QUEUE.read_bytes()).hexdigest()
        self.assertEqual(h,prior["original_review_csv_sha256"])
        private=json.loads(PRIVATE_OUT.read_text(encoding="utf-8"))
        self.assertEqual(private["original_review_csv_sha256"],h)
        self.assertEqual(private["prior_six_leads_sha256"],
                         sha256(PRIOR_SIX.read_bytes()).hexdigest())
        self.assertTrue(PRIVATE_ANNOTATIONS.exists())
        annotation=load_private_annotations()
        stored=json.loads(PRIVATE_OUT.read_text(encoding="utf-8"))
        for source,observed in zip(annotation,stored["source_evidence_objects"]):
            self.assertEqual(source,{key:observed[key] for key in source})

    def test_real_scientific_gate_not_released_by_metadata(self):
        result=json.loads(PUBLIC_OUT.read_text(encoding="utf-8"))
        self.assertEqual(result["scientific_gate"],
                         "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")
        self.assertEqual(result["tasks_with_exact_original_source_row_crosswalk_verified"],0)
        self.assertEqual(result["physically_assayed_protein_constructs_sequence_verified"],0)


if __name__=="__main__":
    unittest.main(verbosity=2)
