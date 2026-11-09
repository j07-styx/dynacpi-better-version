"""Synthetic and real-artifact integrity tests for complete source-only lineage.

Never tests or inspects affinity labels, predictions or reserved-test outcomes.
Seven fake, transitive DOI-connected families emulate the source grouping.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SPEC=importlib.util.spec_from_file_location(
    "all_eligible_study_source_index",
    ROOT/"scripts/reconstruct_conflicted_source_family_provenance_dev.py")
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fiction():
    tasks={}
    sources={}
    family_map={}
    members={}
    shared_examples={}
    queue=[]
    for i in range(7):
        family=sha256(f"fake-family-{i}".encode()).hexdigest()
        a=sha256(f"fake-component-A-{i}".encode()).hexdigest()
        b=sha256(f"fake-component-B-{i}".encode()).hexdigest()
        family_map[a]=family
        family_map[b]=family
        members[family]=[a,b]
        doi=f"doi:10.synthetic/volume{i}"
        shared_examples[doi]={"components":2,"component_prefixes":sorted([a[:12],b[:12]])}
        keyed=[
            ((f"fake-protein-{i}-A","a"),a,{"doi":[doi[4:]],"patent":[f"US_FAKE_{i}"],"pmid":[]}),
            ((f"fake-protein-{i}-B","b"),b,{"doi":[doi[4:]],"patent":[],"pmid":[f"FAKE_{i}"]}),
            ((f"fake-protein-{i}-A","c"),a,{"doi":[],"patent":[],"pmid":[]}),
        ]
        for key, comp, meta in keyed:
            tasks[key]={"task_id":"fictional_task_"+str(len(tasks)),
                        "protein_component_id":comp}
            sources[key]=meta
        queue.append({
            "source_family_sha256":family,
            "affected_provisional_role_component_counts":{"train":1,"test":1},
            "review_status":"UNREVIEWED_METADATA_CONFLICT_NOT_APPROVED",
        })
    preflight={
        "scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "linkage":{
            "source_family_by_protein_component":family_map,
            "family_members":members,
            "source_identifier_total":21,
            "shared_source_identifier_count":7,
            "shared_identifier_examples":shared_examples,
        },
    }
    return tasks,sources,preflight,queue


class EligibleSourceLineageTests(unittest.TestCase):
    def test_reconstruct_and_no_labels_exported(self):
        t,s,p,q=fiction()
        for x in s.values():
            x["Kd (nM)"]="SECRET_AFFINITY_SHOULD_NOT_LEAK"
            x["median_pKd"]="SECRET_AFFINITY_SHOULD_NOT_LEAK"
        private,public=module.reconstruct(t,s,p,q)
        self.assertEqual(len(private),7)
        self.assertEqual(public["all_eligible_tasks_total"],21)
        self.assertEqual(public["all_eligible_source_identifiers_total"],21)
        self.assertEqual(public["all_shared_cross_component_source_identifiers_total"],7)
        self.assertEqual(public["conflict_families_eligible_tasks_total"],21)
        self.assertEqual(public["conflict_families_distinct_original_source_identifiers"],21)
        self.assertEqual(public["conflict_families_source_identifier_counts_by_type"],
                         {"doi":7,"patent":7,"pmid":7})
        self.assertEqual(public["conflict_families_tasks_without_DOI_PMID_patent"],7)
        output=json.dumps((private,public))
        self.assertNotIn("SECRET_AFFINITY_SHOULD_NOT_LEAK",output)
        self.assertEqual(public["families_scientifically_approved"],0)

    def test_public_output_excludes_private_family_and_role_provenance(self):
        _,_,p,q=fiction()
        t,s,_,_=fiction()
        private,public=module.reconstruct(t,s,p,q)
        string=json.dumps(public)
        self.assertNotIn("source_family_sha256",string)
        self.assertNotIn('"original_source_identifier":',string)
        self.assertNotIn('"affected_roles_and_components_PRIVATE":',string)
        self.assertNotIn("fake-component",string)
        self.assertFalse(any(r["source_family_sha256"] in string for r in private))

    def test_source_id_crossing_independent_families_rejected(self):
        t,s,p,q=fiction()
        keys=list(t)
        s[keys[0]]["doi"].append(s[keys[3]]["doi"][0])
        with self.assertRaisesRegex(ValueError, "unexpectedly crosses"):
            module.reconstruct(t,s,p,q)

    def test_source_id_inventory_total_mismatch_rejected(self):
        t,s,p,q=fiction()
        p["linkage"]["source_identifier_total"]=999
        with self.assertRaisesRegex(ValueError,"total does not match"):
            module.reconstruct(t,s,p,q)

    def test_incomplete_shared_identifier_index_rejected(self):
        t,s,p,q=fiction()
        p["linkage"]["shared_identifier_examples"]={}
        with self.assertRaisesRegex(ValueError,"incomplete or inconsistent"):
            module.reconstruct(t,s,p,q)

    def test_shared_component_witness_mismatch_rejected(self):
        t,s,p,q=fiction()
        sample=next(iter(p["linkage"]["shared_identifier_examples"].values()))
        sample["component_prefixes"]=["fake_fake"]
        with self.assertRaisesRegex(ValueError,"witnesses disagree"):
            module.reconstruct(t,s,p,q)

    def test_missing_bibliographic_task_record_rejected(self):
        t,s,p,q=fiction()
        s.pop(next(iter(s)))
        with self.assertRaisesRegex(ValueError,"Missing or additional"):
            module.reconstruct(t,s,p,q)

    def test_family_missing_from_metadata_rejected(self):
        t,s,p,q=fiction()
        del p["linkage"]["source_family_by_protein_component"][
            next(iter(p["linkage"]["source_family_by_protein_component"]))]
        with self.assertRaisesRegex(ValueError,"Task protein component absent"):
            module.reconstruct(t,s,p,q)

    def test_changed_human_review_status_fails_closed(self):
        t,s,p,q=fiction()
        q[0]["review_status"]="APPROVE"
        with self.assertRaises(PermissionError):
            module.reconstruct(t,s,p,q)

    def test_duplicate_family_fails_closed(self):
        t,s,p,q=fiction()
        q[1]["source_family_sha256"]=q[0]["source_family_sha256"]
        with self.assertRaisesRegex(ValueError, "Unknown/duplicated"):
            module.reconstruct(t,s,p,q)

    def test_actual_existing_public_and_private_reports_consistent(self):
        if not (module.PRIVATE_OUT.exists() and module.PUBLIC_OUT.exists()):
            self.skipTest("Private source report unavailable")
        public=json.loads(module.PUBLIC_OUT.read_text(encoding="utf-8"))
        internal=json.loads(module.PRIVATE_OUT.read_text(encoding="utf-8"))
        self.assertEqual(len(internal["families"]),7)
        self.assertEqual(sum(x["technically_eligible_task_count"] for x in internal["families"]),
                         public["conflict_families_eligible_tasks_total"])
        self.assertEqual(sum(len(x["all_original_source_identifiers"]) for x in internal["families"]),
                         public["conflict_families_distinct_original_source_identifiers"])
        self.assertEqual(sum(len(x["technically_eligible_tasks_missing_all_DOI_PMID_patent"])
                             for x in internal["families"]),
                         public["conflict_families_tasks_without_DOI_PMID_patent"])
        self.assertEqual(internal["original_frozen_preflight_sha256"],
                         sha256(module.PREFLIGHT.read_bytes()).hexdigest())
        self.assertEqual(internal["original_private_conflict_queue_sha256"],
                         sha256(module.QUEUE.read_bytes()).hexdigest())
        self.assertTrue(all(x["approval_status"]=="UNREVIEWED_NOT_SCIENTIFICALLY_APPROVED"
                            for x in internal["families"]))

if __name__=="__main__":
    unittest.main(verbosity=2)
