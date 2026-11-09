"""Synthetic-only + immutable-metadata integrity checks for assay context triage.

All source/target/assay identities in synthetic fixtures are invented.
No numerical binding affinity values, chemistry or reserved test outcomes.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import triage_unattributed_assay_contexts_v4 as m


def pretend():
    names=(
        [m.KINASE_NAME]*54+
        ["D3R2021","D3R2022","D3R2023","D3R2025"]+
        ["OctetRed","OctetRed_Method1","OctetRed_Method2"]+
        ["Thermofluor"]
    )
    tasks=[]
    contexts=[]
    missing=[]
    for i,name in enumerate(names):
        c="fictional_protein_component_a" if i<61 else "fictional_protein_component_b"
        protein=sha256(f"fake_seq_{i}".encode()).hexdigest()
        assay=f"fake_assay_{i}"
        task=f"fake_task_{i}"
        row={key:None for key in m.TASK_FIELDS}
        row.update({"task_id":task,"assay_key":assay,
                    "protein_component_id":c,"protein_seq_sha256":protein,
                    "n_unique_compounds":15})
        tasks.append(row)
        context={
            "sequence_sha256":protein,
            "assay_key":assay,
            "name":name,
            "description":"Synthetic methods description",
            "article_doi_count":0,
            "preliminary_size_band":"synthetic",
            "raw_record_count":25,
            "signals":{k:False for k in m.SIGNAL_KEYS},
        }
        contexts.append(context)
        missing.append({
            "task_id":task,"protein_component_id":c,"assay_key":assay,
            "source_record_verified":False,
            "assay_approved":False,"training_eligible":False,
        })
    for i in range(1084):
        row={key:None for key in m.TASK_FIELDS}
        row.update({"task_id":f"other_fake_task_{i}",
                    "protein_component_id":"other_component",
                    "protein_seq_sha256":sha256(f"other_seq_{i}".encode()).hexdigest(),
                    "assay_key":f"other_assay_{i}","n_unique_compounds":15})
        tasks.append(row)
    private={"status":"PRIVATE_REVIEW_ONLY_NO_SOURCE_ASSIGNMENTS",
             "records":missing}
    return tasks,contexts,private


class AssayContextReviewTests(unittest.TestCase):
    def test_synthetic_groups_and_zero_approval(self):
        t,c,old=pretend()
        detailed,out=m.audit_contexts(t,c,old)
        self.assertEqual(len(detailed),62)
        self.assertEqual(out["assay_context_categories"],{
            m.CATEGORIES[0]:54,m.CATEGORIES[1]:4,m.CATEGORIES[2]:3,
            m.CATEGORIES[3]:1,m.CATEGORIES[4]:0
        })
        self.assertEqual(out["tasks_with_exact_source_record_verified"],0)
        self.assertEqual(out["tasks_with_independent_source_approval"],0)
        self.assertEqual(out["tasks_added_to_training_or_test"],0)
        self.assertNotIn("fake_task_0",json.dumps(out))

    def test_four_metadata_categories_and_unrecognized(self):
        self.assertEqual(m.classify(m.KINASE_NAME),m.CATEGORIES[0])
        self.assertEqual(m.classify("D3R2025"),m.CATEGORIES[1])
        self.assertEqual(m.classify("OctetRed_Method2"),m.CATEGORIES[2])
        self.assertEqual(m.classify("Thermofluor"),m.CATEGORIES[3])
        self.assertEqual(m.classify("UnknownAssay"),m.CATEGORIES[4])

    def test_missing_exact_sequence_assay_context_rejected(self):
        t,c,old=pretend()
        c.pop()
        with self.assertRaisesRegex(ValueError,"Missing exact"):
            m.audit_contexts(t,c,old)

    def test_changed_component_rejected(self):
        t,c,old=pretend()
        old["records"][0]["protein_component_id"]="new_component"
        with self.assertRaisesRegex(ValueError,"identity"):
            m.audit_contexts(t,c,old)

    def test_approval_flags_fail_closed(self):
        t,c,old=pretend()
        old["records"][0]["assay_approved"]=True
        with self.assertRaises(PermissionError):
            m.audit_contexts(t,c,old)

    def test_metadata_DOI_count_must_remain_absent(self):
        t,c,old=pretend()
        c[0]["article_doi_count"]=1
        with self.assertRaisesRegex(ValueError,"DOI metadata changed"):
            m.audit_contexts(t,c,old)

    def test_keyword_flags_only_metadata_not_measurements(self):
        t,c,old=pretend()
        c[0]["signals"]["kd"]=True
        with self.assertRaisesRegex(ValueError,"keyword signals changed"):
            m.audit_contexts(t,c,old)

    def test_malformed_assay_signals_fail_closed(self):
        t,c,old=pretend()
        c[0]["signals"]["kd"]="NOT_A_BOOLEAN"
        with self.assertRaisesRegex(ValueError,"boolean metadata"):
            m.audit_contexts(t,c,old)

    def test_duplicate_assay_context_fails(self):
        t,c,old=pretend()
        c.append(deepcopy(c[0]))
        with self.assertRaisesRegex(ValueError,"Duplicate original assay"):
            m.audit_contexts(t,c,old)

    def test_changed_original_task_inventory_fails(self):
        t,c,old=pretend()
        t.pop()
        with self.assertRaisesRegex(ValueError,"1146"):
            m.audit_contexts(t,c,old)

    def test_only_exact_allowlisted_source_metadata_can_be_read(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"fictional_tasks.jsonl"
            row={k:None for k in m.TASK_FIELDS}
            row.update({"task_id":"t","n_unique_compounds":15})
            path.write_text(json.dumps(row)+"\n",encoding="utf-8")
            digest=sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len(m.read_jsonl_metadata(path,digest,m.TASK_FIELDS)),1)
            row["median_pKd"]="UNSAFE_POTENCY_LABEL"
            path.write_text(json.dumps(row)+"\n",encoding="utf-8")
            with self.assertRaises(PermissionError):
                m.read_jsonl_metadata(
                    path,sha256(path.read_bytes()).hexdigest(),m.TASK_FIELDS)
            with self.assertRaises(PermissionError):
                m.read_jsonl_metadata(path,digest,m.TASK_FIELDS)

    def test_original_saved_private_and_public_aggregate_replay(self):
        preflight=json.loads(m.PREFLIGHT.read_text(encoding="utf-8"))
        checksums=preflight["data_source_integrity_sha256"]
        tasks=m.read_jsonl_metadata(m.TASK,checksums[m.TASK.name],m.TASK_FIELDS)
        contexts=m.read_jsonl_metadata(
            m.CONTEXT,checksums[m.CONTEXT.name],m.CONTEXT_FIELDS)
        neighbors=json.loads(m.NEIGHBORS.read_text(encoding="utf-8"))
        detail,aggregate=m.audit_contexts(tasks,contexts,neighbors)
        old=json.loads(m.PRIVATE_OUT.read_text(encoding="utf-8"))
        saved=json.loads(m.PUBLIC_OUT.read_text(encoding="utf-8"))
        self.assertEqual(aggregate,saved)
        self.assertEqual(detail,old["records"])
        self.assertEqual(saved["assay_context_exact_task_sequence_matches"],62)
        self.assertEqual(saved["scientific_gate"],
                         "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")

    def test_public_outputs_do_not_reveal_protected_metadata(self):
        private=json.loads(m.PRIVATE_OUT.read_text(encoding="utf-8"))
        paths=(
            m.PUBLIC_OUT,
            ROOT/"scripts/triage_unattributed_assay_contexts_v4.py",
            ROOT/"tests_v4/unattributed_assay_context_triage_test.py",
            ROOT/"research/DYNACPI_UNATTRIBUTED_ASSAY_CONTEXT_TRIAGE_20261109.md",
        )
        public="".join(p.read_text(encoding="utf-8") for p in paths)
        for row in private["records"]:
            self.assertNotIn(row["task_id"],public)
            self.assertNotIn(row["protein_component_id"],public)
            self.assertNotIn(row["protein_sequence_sha256"],public)
            self.assertNotIn(row["assay_key"],public)
            self.assertNotIn(row["assay_description_sha256"],public)
        self.assertEqual(private["status"],
                         "PRIVATE_ASSAY_CONTEXT_REVIEW_NO_SOURCE_ASSIGNMENT")

if __name__=="__main__":
    unittest.main(verbosity=2)
