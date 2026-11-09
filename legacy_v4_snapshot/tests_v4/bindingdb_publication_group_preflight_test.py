"""Source publication/protein-cold audit invariants and fail-closed guards."""
import csv
import hashlib
import json
import unittest
from pathlib import Path

from dynacpi.balm_public_data_dev import ROOT
from dynacpi.scientific_cohort_gate import current_status
from dynacpi.source_publication_split_guard import (
    SourceFamilySplitError,require_source_disjoint_roles,
)
from scripts.prioritize_bindingdb_component_source_review_dev import (
    choose_review_leads,source_description_flags,
)
from scripts.audit_bindingdb_source_linked_protein_groups_dev import (
    DisjointSet,connect_components,source_ids,choose_source_independent_leads,
)

RESEARCH=ROOT/"research"
PRI=RESEARCH/"bindingdb_component_balanced_source_review_priority_20261108.json"
FINAL=RESEARCH/"bindingdb_publication_connected_protein_review_preflight_20261108.json"
CSV=RESEARCH/"bindingdb_publication_disjoint_source_review_queue_20261108.csv"


class SourceIndependentReviewPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior=json.loads(PRI.read_text(encoding="utf8"))
        cls.new=json.loads(FINAL.read_text(encoding="utf8"))
        cls.leads=cls.new["source_independent_review_leads"]
        cls.grouping=cls.new["linkage"]["source_family_by_protein_component"]

    def test_fitc_not_itc_and_indirect_probe_detected(self):
        method,risks=source_description_flags(
            "Binding to human RPA70N using FITC-ATRIP fluorescent polarization")
        self.assertNotIn("ITC",method)
        self.assertIn("indirect_activity_or_probe",risks)
        m,r=source_description_flags("Binding directly by isothermal titration calorimetry")
        self.assertEqual(m,("ITC",))
        self.assertEqual(r,())
        m,r=source_description_flags("dissociation rate constant measured by SPR")
        self.assertIn("SPR",m)
        self.assertIn("kinetics_not_equilibrium_kd",r)
        m,r=source_description_flags("Biolayer interferometry via Octet, direct Kd")
        self.assertIn("BLI",m)

    def test_disjoint_set_and_identifier_normalization(self):
        ds=DisjointSet(["A","B","C","D"])
        self.assertTrue(ds.union("A","B"))
        self.assertTrue(ds.union("B","C"))
        self.assertFalse(ds.union("A","C"))
        self.assertEqual(len(ds.groups()),2)
        ids=source_ids({
            "doi":{"HTTPS://doi.org/10.1016/ABC"},
            "pmid":{"123456"},"patent":{"US777777"}
        })
        self.assertEqual(ids,{"doi:10.1016/abc","pmid:123456","patent:us777777"})

    def test_actual_eligible_source_family_merges(self):
        x=self.new
        self.assertEqual(x["overview"]["technical_tasks_ge15"],1146)
        self.assertEqual(x["overview"]["preliminary_MMseqs_components"],168)
        self.assertEqual(x["overview"]["source_connected_components"],142)
        self.assertEqual(x["overview"]["protein_components_merged_across_source_edges"],26)
        self.assertEqual(x["overview"]["largest_source_family_task_count"],530)
        self.assertEqual(x["overview"]["old_40_component_only_shortlist_source_families"],38)
        self.assertEqual(x["overview"]["new_independent_source_family_count"],40)
        self.assertEqual(len(self.leads),40)
        self.assertEqual(len(set(self.grouping.values())),142)
        self.assertEqual(len({r["source_family_id"] for r in self.leads}),40)
        self.assertEqual(len({r["protein_component_id"] for r in self.leads}),40)
        self.assertTrue(all(r["review_decision"]=="UNREVIEWED" for r in self.leads))

    def test_source_ids_disjoint_even_when_doi_or_pmid_multiple(self):
        seen=set()
        for row in self.leads:
            row_sources={
                kind+":"+str(value).strip().lower()
                for kind,field in (
                    ("doi","doi_candidates"),("pmid","pmid_candidates"),
                    ("patent","patent_candidates")
                ) for value in row[field] if value
            }
            self.assertFalse(row_sources&seen)
            seen|=row_sources

    def test_metadata_only_review_queue_has_no_approved_task_or_label(self):
        with CSV.open(encoding="utf8",newline="") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),40)
        self.assertEqual({r["review_decision"] for r in rows},{"UNREVIEWED"})
        self.assertNotIn("Kd (nM)",rows[0])
        self.assertNotIn("pKd",rows[0])
        self.assertNotIn("model_score",rows[0])
        self.assertEqual(self.new["scientific_gate"],
                         "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")
        self.assertEqual(self.new["status"],
          "QUARANTINED_LABEL_BLIND_PUBLICATION_LINKED_PROTEIN_COMPONENTS_NO_SPLIT_OR_APPROVAL")
        self.assertEqual(current_status(ROOT)["status"],
                         "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")

    def test_fail_closed_guard_blocks_shared_source_across_roles(self):
        fam_a="a"*64
        fam_b="b"*64
        meta=[
            {"task_id":"A","protein_component_id":"c1"},
            {"task_id":"B","protein_component_id":"c2"},
            {"task_id":"C","protein_component_id":"c3"},
        ]
        families={"c1":fam_a,"c2":fam_a,"c3":fam_b}
        split={"A":"base_train","B":"test","C":"adapter_validation"}
        with self.assertRaises(SourceFamilySplitError):
            require_source_disjoint_roles(split,meta,families)
        split["B"]="base_train"
        ok=require_source_disjoint_roles(split,meta,families)
        self.assertEqual(ok["source_connected_families_checked"],2)
        self.assertIn("NO_APPROVAL",ok["status"])
        with self.assertRaises(SourceFamilySplitError):
            require_source_disjoint_roles(split,meta,{})
        with self.assertRaises(SourceFamilySplitError):
            require_source_disjoint_roles({"A":"base_train"},meta,families)
        with self.assertRaises(SourceFamilySplitError):
            require_source_disjoint_roles({"A":"base_train","B":"unknown","C":"test"},meta,families)

    def test_actual_selected_40_pass_only_metadata_disjoint_guard(self):
        roles=["base_train","base_validation","adapter_train","adapter_validation","test"]
        sample={
            row["task_id"]:roles[i%len(roles)]
            for i,row in enumerate(self.leads)
        }
        proof=require_source_disjoint_roles(sample,self.leads,self.grouping)
        self.assertEqual(proof["tasks_checked"],40)
        self.assertEqual(proof["source_connected_families_checked"],40)
        self.assertIn("NO_APPROVAL",proof["status"])
        # The rule does not freeze that artificial in-memory assignment.

if __name__=="__main__":
    unittest.main()
