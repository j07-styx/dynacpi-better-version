"""Synthetic regression suite for technical 30-task BindingDB source lineage audit.

Uses only temporary synthetic numbers/sequences. Never touches scientific data.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"audit_30_record_level_provenance_v4.py"
spec=importlib.util.spec_from_file_location("dynacpi_record_lineage",SCRIPT)
aud=importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)

def make_fixture(root, *, bad_pkd=False, missing_source=False,
                 qualified_kd=False, changed_decision=False):
    folder=root/"data"/"audit"
    folder.mkdir(parents=True)
    sequence="ACDEFGHIKL"
    seqhash=hashlib.sha256(sequence.encode()).hexdigest()
    reviews=[];groups=[];sources=[];candidates=[]
    for n in range(30):
        reactants=[str(20000+n*3)]
        if n==0:reactants.append("20001")
        values=[50.0,200.0] if n==0 else [100.0]
        pkds=[9-math.log10(v) for v in values]
        reviews.append({
            "task_id":f"task_{n:02d}","assay_key":f"assay_{n}",
            "target_names":f"Synthetic protein {n}",
            "protein_sha256":seqhash,"record_count":str(len(reactants)),
            "unique_compounds":"1","article_dois":"10.1000/test",
            "pmids":"10000001","patent_numbers":"",
            "curation_decision":"APPROVE" if changed_decision and n==0 else "UNREVIEWED"
        })
        groups.append({
            "task_id":f"task_{n:02d}","reactant_ids":reactants,
            "median_pKd":statistics.median(pkds),
            "range_pKd":max(pkds)-min(pkds),"replicates":len(reactants)
        })
        for j,(key,kd,pkd) in enumerate(zip(reactants,values,pkds)):
            s={
                "BindingDB Reactant_set_id":key,
                "Kd (nM)":"<50" if qualified_kd and n==0 and j==0 else str(kd),
                "Ki (nM)":"","IC50 (nM)":"","EC50 (nM)":"",
                "protein_construct_sequence_sha256":seqhash,
                "Number of Protein Chains in Target (>1 implies a multichain complex)":"1",
                "Article DOI":"10.1000/test","PMID":"10000001","Patent Number":""
            }
            if not (missing_source and n==0 and j==0):sources.append(s)
            candidates.append({
                "reactant_set_id":key,"assay_key":f"assay_{n}","sequence":sequence,
                "pKd":pkd+(.3 if bad_pkd and n==1 else 0)
            })
    with (folder/aud.FILES["review"]).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(reviews[0]));w.writeheader();w.writerows(reviews)
    for key,items in (("median",groups),("source",sources),("candidate",candidates)):
        with (folder/aud.FILES[key]).open("w",encoding="utf-8") as f:
            for item in items:f.write(json.dumps(item)+"\n")

class SourceLineageSyntheticTest(unittest.TestCase):
    def test_valid_thirty_tasks_two_replicates(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_fixture(root)
            report=aud.audit(root)
            self.assertEqual(report["tasks"],30)
            self.assertEqual(report["compound_groups"],30)
            self.assertEqual(report["source_reactant_references"],31)
            self.assertEqual(report["original_reactant_ids_found"],31)
            self.assertEqual(report["unaggregated_reactant_ids_found"],31)
            self.assertEqual(report["summary_counts"]["COMPOUND_GROUPS_WITH_REPLICATES"],1)
            self.assertEqual(report["critical_lineage_flags"],{})
            self.assertFalse(report["main_training_permitted"])
            self.assertEqual(report["approved_tasks"],0)
            text=(root/"research/source_access_v4/record_level_provenance_30.json").read_text()
            self.assertNotIn('"Kd (nM)"',text)
            self.assertNotIn('"median_pKd"',text)

    def test_wrong_kd_conversion_is_flagged(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_fixture(root,bad_pkd=True)
            report=aud.audit(root)
            self.assertEqual(report["critical_lineage_flags"]["RAW_KD_TO_PK_D_CONVERSION_MISMATCH"],1)

    def test_missing_source_record_is_flagged(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_fixture(root,missing_source=True)
            report=aud.audit(root)
            self.assertEqual(report["original_reactant_ids_found"],30)
            self.assertEqual(report["critical_lineage_flags"]["SOURCE_RECORD_MISSING"],1)

    def test_qualified_kd_is_not_treated_as_numeric(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_fixture(root,qualified_kd=True)
            report=aud.audit(root)
            self.assertEqual(report["summary_counts"]["RAW_KD_NOT_STRICT_POSITIVE_NUMERIC"],1)
            self.assertEqual(report["approved_tasks"],0)
        self.assertIsNone(aud.positive_kd("<50"))
        self.assertIsNone(aud.positive_kd(">100"))
        self.assertIsNone(aud.positive_kd("-1"))
        self.assertEqual(aud.positive_kd("50.0"),50.0)

    def test_changed_review_decision_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_fixture(root,changed_decision=True)
            with self.assertRaises(PermissionError):aud.audit(root)

if __name__=="__main__":
    unittest.main(verbosity=2)
