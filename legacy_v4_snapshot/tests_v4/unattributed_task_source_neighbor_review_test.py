"""Fictional original-source neighborhood classification and real privacy checks.

Never reads affinity columns, query predictions, molecular identities,
reserved test outcomes or protein sequences. Fake DOI strings are invented.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_unattributed_task_source_neighbor_review_v4 as m


def fiction():
    f1=sha256(b"fiction-family-1").hexdigest()
    f2=sha256(b"fiction-family-2").hexdigest()
    a=sha256(b"fiction-comp-a").hexdigest()
    b=sha256(b"fiction-comp-b").hexdigest()
    c=sha256(b"fiction-comp-c").hexdigest()
    tasks={
        "s1":{"protein_component_id":a,"assay_key":"assay-one"},
        "u1":{"protein_component_id":a,"assay_key":"assay-one"},
        "u2":{"protein_component_id":a,"assay_key":"assay-two"},
        "u3":{"protein_component_id":b,"assay_key":"assay-one"},
        "u4":{"protein_component_id":b,"assay_key":"assay-three"},
        "s2":{"protein_component_id":c,"assay_key":"assay-four"},
        "u5":{"protein_component_id":c,"assay_key":"assay-five"},
    }
    graph={"scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
           "linkage":{"source_family_by_protein_component":{a:f1,b:f1,c:f2},
                      "family_members":{f1:[a,b],f2:[c]}}}
    source={"status":"CONFIDENTIAL_COMPLETE_ELIGIBLE_SOURCE_IDS_NO_APPROVAL",
            "families":[
                {"source_family_sha256":f1,"all_original_protein_component_ids":[a,b],
                 "technically_eligible_task_count":5,
                 "approval_status":"UNREVIEWED_NOT_SCIENTIFICALLY_APPROVED",
                 "technically_eligible_tasks_missing_all_DOI_PMID_patent":[
                     "u1","u2","u3","u4"
                 ],
                 "all_original_source_identifiers":[
                     {"original_source_identifier":"doi:10.fake/research-one",
                      "eligible_task_ids":["s1"],"protein_component_ids":[a]}
                 ]},
                {"source_family_sha256":f2,"all_original_protein_component_ids":[c],
                 "technically_eligible_task_count":2,
                 "approval_status":"UNREVIEWED_NOT_SCIENTIFICALLY_APPROVED",
                 "technically_eligible_tasks_missing_all_DOI_PMID_patent":["u5"],
                 "all_original_source_identifiers":[
                     {"original_source_identifier":"pmid:12345678",
                      "eligible_task_ids":["s2"],"protein_component_ids":[c]}
                 ]}
            ]}
    return tasks,graph,source


class UnattributedSourceNeighborTests(unittest.TestCase):
    def test_all_four_review_tiers_keep_tasks_unattributed(self):
        tasks,g,s=fiction()
        rows,out=m.review_neighbors(tasks,g,s,strict=False)
        self.assertEqual(len(rows),5)
        self.assertEqual(out["source_neighbor_tier_counts"],{
            m.TIERS[0]:1,m.TIERS[1]:2,m.TIERS[2]:1,m.TIERS[3]:1,
        })
        by={r["task_id"]:r for r in rows}
        self.assertEqual(by["u1"]["review_tier"],m.TIERS[0])
        self.assertEqual(by["u2"]["review_tier"],m.TIERS[1])
        self.assertEqual(by["u3"]["review_tier"],m.TIERS[2])
        self.assertEqual(by["u4"]["review_tier"],m.TIERS[3])
        self.assertTrue(all(not r["source_record_verified"] for r in rows))
        self.assertFalse(out["source_imputation_performed"])

    def test_neighbor_source_ids_are_never_attributed_to_missing_task(self):
        tasks,g,s=fiction()
        rows,out=m.review_neighbors(tasks,g,s,strict=False)
        self.assertEqual(
            next(r for r in rows if r["task_id"]=="u1")[
                "NEIGHBOR_source_identifiers_NOT_ATTRIBUTED"
            ],["doi:10.fake/research-one"])
        self.assertEqual(out["original_source_records_verified"],0)
        self.assertNotIn("doi:10.fake/research-one",json.dumps(out))

    def test_forged_family_source_partition_rejected(self):
        tasks,g,s=fiction()
        s["families"][0]["technically_eligible_tasks_missing_all_DOI_PMID_patent"].remove("u4")
        with self.assertRaisesRegex(ValueError,"partition"):
            m.review_neighbors(tasks,g,s,strict=False)

    def test_cross_component_or_family_source_link_rejected(self):
        tasks,g,s=fiction()
        s["families"][0]["all_original_source_identifiers"][0][
            "protein_component_ids"]=[next(iter(g["linkage"]["family_members"].values()))[1]]
        with self.assertRaisesRegex(ValueError,"component witness mismatch"):
            m.review_neighbors(tasks,g,s,strict=False)

    def test_missing_source_does_not_mean_source_approval(self):
        tasks,g,s=fiction()
        s["families"][0]["approval_status"]="APPROVED"
        with self.assertRaises(PermissionError):
            m.review_neighbors(tasks,g,s,strict=False)

    def test_unknown_component_or_metadata_graph_change_rejected(self):
        tasks,g,s=fiction()
        tasks["u1"]["protein_component_id"]="UNKNOWN_COMPONENT"
        with self.assertRaisesRegex(ValueError,"inventory diverged"):
            m.review_neighbors(tasks,g,s,strict=False)

    def test_frozen_metadata_file_contains_no_affinity_fields(self):
        row={k:("" if "sha256" in k else 0) for k in m.ALLOWED}
        row.update({"task_id":"t1","protein_component_id":"c1",
                    "assay_key":"a1","n_unique_compounds":15})
        with TemporaryDirectory() as td:
            f=Path(td)/"fake_meta.jsonl"
            f.write_text(json.dumps(row)+"\n",encoding="utf8")
            hashed=sha256(f.read_bytes()).hexdigest()
            self.assertEqual(m.read_tasks(f,hashed)["t1"]["assay_key"],"a1")
            row["median_pKd"]="NEVER_READ_THIS_SECRET"
            f.write_text(json.dumps(row)+"\n",encoding="utf8")
            with self.assertRaises(PermissionError):
                m.read_tasks(f,sha256(f.read_bytes()).hexdigest())

    def test_frozen_file_checksum_fails_closed(self):
        with TemporaryDirectory() as td:
            f=Path(td)/"fake_meta.jsonl"
            f.write_text("[]\n",encoding="utf8")
            with self.assertRaises(PermissionError):
                m.read_tasks(f,"0"*64)

    def test_original_private_review_results_and_public_aggregates(self):
        rows=json.loads(m.PRIVATE_OUT.read_text(encoding="utf-8"))
        public=json.loads(m.PUBLIC_OUT.read_text(encoding="utf-8"))
        self.assertEqual(rows["status"],
                         "PRIVATE_REVIEW_ONLY_NO_SOURCE_ASSIGNMENTS")
        self.assertEqual(len(rows["records"]),62)
        self.assertEqual(public["technical_tasks_with_no_original_DOI_PMID_or_patent"],62)
        self.assertEqual(sum(public["source_neighbor_tier_counts"].values()),62)
        self.assertTrue(all(not r["training_eligible"] and not r["assay_approved"]
                            for r in rows["records"]))
        self.assertFalse(public["source_imputation_performed"])
        pub=m.PUBLIC_OUT.read_text(encoding="utf-8")
        pub+=(ROOT/"research/DYNACPI_UNATTRIBUTED_SOURCE_NEIGHBOR_REVIEW_20261109.md").read_text(encoding="utf-8")
        for r in rows["records"]:
            self.assertNotIn(r["task_id"],pub)
            self.assertNotIn(r["protein_component_id"],pub)
            self.assertNotIn(r["source_family_sha256"],pub)
            for sid in r["NEIGHBOR_source_identifiers_NOT_ATTRIBUTED"]:
                self.assertNotIn(sid,pub)

    def test_current_metadata_is_hash_bound_and_has_1146_eligible_tasks(self):
        preflight=json.loads(m.PREFLIGHT.read_text(encoding="utf-8"))
        expected=preflight["data_source_integrity_sha256"][m.TASK_FILE.name]
        tasks=m.read_tasks(m.TASK_FILE,expected)
        self.assertEqual(len(tasks),1146)
        original=json.loads(m.SOURCE.read_text(encoding="utf-8"))
        details,public=m.review_neighbors(tasks,preflight,original)
        self.assertEqual(public,json.loads(m.PUBLIC_OUT.read_text(encoding="utf-8")))
        self.assertEqual(len(details),62)

if __name__=="__main__":
    unittest.main(verbosity=2)
