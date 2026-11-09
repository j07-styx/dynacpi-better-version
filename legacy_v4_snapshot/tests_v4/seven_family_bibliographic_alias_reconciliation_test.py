"""Synthetic-only bibliography alias review tests; no assay labels or private IDs."""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import reconcile_seven_family_bibliographic_aliases_v4 as m


def fixture():
    families=[]
    for i in range(7):
        families.append({
            "source_family_sha256":sha256(f"fictional_family_{i}".encode()).hexdigest(),
            "all_original_source_identifiers":[],
            "technically_eligible_tasks_missing_all_DOI_PMID_patent":list(
                range(9 if i<6 else 8)),
        })
    do=["10.9999/fictitious"+str(i).zfill(4) for i in range(74)]
    pm=[str(50000000+i) for i in range(90)]
    patents=["US_FAKE_"+str(i) for i in range(56)]
    for k,items in (("doi",do),("pmid",pm),("patent",patents)):
        for idx,ident in enumerate(items):
            fam=families[idx%7]
            fam["all_original_source_identifiers"].append({
                "original_source_identifier":f"{k}:{ident}"
            })
    pmdata={p:[] for p in pm}
    doidata={v:[] for v in do}
    for i in range(74):
        pub={"pmid":pm[i],"doi":do[i],"open_access":False,"pmcid":None}
        pmdata[pm[i]]=[pub]
        doidata[do[i]]=[pub]
    return families,pmdata,doidata


class AliasReconciliationTests(unittest.TestCase):
    def test_both_directions_confirm_same_bibliographic_article(self):
        f,p,d=fixture()
        private,public=m.reconcile(f,p,d)
        self.assertEqual(len(private),7)
        self.assertEqual(public["reciprocal_DOI_PMID_aliases_confirmed_as_same_publication"],74)
        self.assertEqual(public["pmid_identifiers_with_exact_MED_metadata"],74)
        self.assertEqual(public["doi_identifiers_with_exact_MED_metadata"],74)
        self.assertEqual(public["families_scientifically_approved"],0)
        self.assertEqual(public["original_assay_record_crosswalks_verified"],0)
        self.assertNotIn(f[0]["source_family_sha256"],json.dumps(public))
        self.assertNotIn("10.9999",json.dumps(public))

    def test_one_direction_only_does_not_promote(self):
        f,p,d=fixture()
        doi="10.9999/fictitious0002"
        d[doi]=[]
        _,public=m.reconcile(f,p,d)
        self.assertEqual(public["reciprocal_DOI_PMID_aliases_confirmed_as_same_publication"],73)
        self.assertEqual(public["one_way_only_DOI_PMID_alias_leads_not_promoted"],1)

    def test_cross_preliminary_family_link_is_not_merged(self):
        f,p,d=fixture()
        far_pmid=str(50000000+74)
        cross_doi="10.9999/fictitious0000"
        p[far_pmid]=[{"pmid":far_pmid,"doi":cross_doi,"pmcid":None,
                       "open_access":False}]
        private,public=m.reconcile(f,p,d)
        self.assertEqual(public["cross_preliminary_family_bibliography_link_leads_require_review"],1)
        self.assertEqual(public["families_scientifically_approved"],0)
        self.assertNotIn(cross_doi,json.dumps(public))

    def test_aliases_cannot_be_nonbijective(self):
        f,p,d=fixture()
        p[str(50000000+74)]=[{"pmid":str(50000000+74),
                              "doi":"10.9999/fictitious0000",
                              "open_access":False,"pmcid":None}]
        # Cross-family link alone is treated as a review lead, not a merged alias.
        _,public=m.reconcile(f,p,d)
        self.assertEqual(public["reciprocal_DOI_PMID_aliases_confirmed_as_same_publication"],74)

    def test_doi_url_normalization_and_bad_strings(self):
        self.assertEqual(m.normalize_doi("HTTPS://DOI.ORG/10.1000/Foo"),
                         "10.1000/foo")
        self.assertEqual(m.normalize_doi("doi:10.1000/Foo"),"10.1000/foo")
        self.assertEqual(m.normalize_doi("x"),"")
        self.assertEqual(m.normalize_doi("10.1"),"")

    def test_europe_pmc_exact_only_filters_nonmed_and_fuzzy(self):
        items=["50000000","50000001"]
        raw=[
            {"source":"MED","pmid":items[0],"doi":"10.1000/a"},
            {"source":"PMC","pmid":items[1],"doi":"10.1000/b"},
            {"source":"MED","pmid":"50000002","doi":"10.1000/c"},
        ]
        matches=m.exact_matches("pmid",items,raw)
        self.assertEqual(len(matches[items[0]]),1)
        self.assertEqual(matches[items[1]],[])

    def test_fetch_helper_query_is_exact_AND_MED(self):
        payload={"hitCount":1,"resultList":{"result":[
            {"source":"MED","pmid":"50000000","doi":"10.1000/a"}]}}
        with patch.object(m,"fetch",return_value=json.dumps(payload).encode()) as stub:
            kind,ids,found=m.query_med_batch("pmid",["50000000"])
            self.assertEqual((kind,ids),("pmid",["50000000"]))
            self.assertEqual(found["50000000"][0]["doi"],"10.1000/a")
            url=stub.call_args.args[0]
            self.assertIn("SRC%3AMED",url)
            self.assertIn("EXT_ID%3A50000000",url)

    def test_input_frozen_inventory_integrity(self):
        f,p,d=fixture()
        data={"status":"CONFIDENTIAL_COMPLETE_ELIGIBLE_SOURCE_IDS_NO_APPROVAL",
              "families":f}
        with tempfile.TemporaryDirectory() as td:
            fn=Path(td)/"fake_bibliography.json"
            fn.write_text(json.dumps(data))
            self.assertEqual(len(m.load_private_identifiers(fn)),7)
            f[0]["all_original_source_identifiers"].pop()
            fn.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"totals"):
                m.load_private_identifiers(fn)

    def test_reject_forged_approval(self):
        f,p,d=fixture()
        data={"status":"SCIENTIFICALLY_APPROVED","families":f}
        with tempfile.TemporaryDirectory() as td:
            fn=Path(td)/"fake_bibliography.json"
            fn.write_text(json.dumps(data))
            with self.assertRaises(PermissionError):
                m.load_private_identifiers(fn)

    def test_input_unsupported_bibliography_type_rejected(self):
        f,p,d=fixture()
        f[0]["all_original_source_identifiers"][0]["original_source_identifier"]="drug:fake"
        with self.assertRaises(ValueError):
            m.reconcile(f,p,d)


if __name__=="__main__":
    unittest.main(verbosity=2)
