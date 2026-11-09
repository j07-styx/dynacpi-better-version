"""Assert train-only alignment-guided dual-cold experiment partition integrity."""
import json,unittest
from pathlib import Path

from dynacpi.balm_public_data_dev import ROOT

RULES={"identity30_cover80":271,"identity40_cover80":396}
BASE=ROOT/"research"
SOURCE=BASE/"balm_train_only_smith_waterman_protein_group_audit_20261108.json"

class AlignmentGroupScaffoldCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=json.loads(SOURCE.read_text(encoding="utf-8"))
        cls.results={}
        for rule in RULES:
            path=BASE/("balm_esm8m_nested_sw_"+rule+"_murcko_scaffold_cold_traincv_dev.json")
            cls.results[rule]=json.loads(path.read_text(encoding="utf-8"))

    def test_alignment_protocol_and_exhaustiveness(self):
        self.assertEqual(self.audit["pairs_scored"],639*638//2)
        self.assertEqual(self.audit["proteins"],639)
        for rule,count in RULES.items():
            self.assertEqual(self.audit["rules"][rule]["connected_components"],count)
            result=self.results[rule]
            self.assertEqual(result["protein_group_rule"],rule)
            self.assertEqual(result["heldout_alignment_group_count"],count)

    def test_each_train_record_evaluated_once_under_scaffold_exclusion(self):
        for rule,data in self.results.items():
            outer=data["outer_folds"]
            self.assertEqual(len(outer),4)
            self.assertEqual(data["original_frozen_train_records"],15538)
            self.assertEqual(sum(f["heldout_pairs"] for f in outer),15538)
            self.assertEqual(sum(f["heldout_alignment_groups"] for f in outer),RULES[rule])
            for fold in outer:
                self.assertEqual(fold["murcko_scaffold_overlap"],0)
                self.assertGreater(fold["fit_unique_pairs"],3000)
                self.assertEqual(len(fold["inner_folds"]),3)
                self.assertIn(fold["inherited_beta_selected_inside_this_outer_fold"],[0.,.25,.5,.75,1.])
                self.assertIn("nested_kmer_esm",fold["outer_metrics"])

    def test_bootstrap_coverage_and_baselines(self):
        for rule,data in self.results.items():
            oof=data["out_of_fold"]
            self.assertEqual(oof["bootstrap"]["replicates"],5000)
            for key in ("train_mean","ridge_kmer","ridge_kmer_plus_esm",
                        "joint_kmer","joint_kmer_esm","nested_kmer","nested_kmer_esm"):
                self.assertIn(key,oof["pooled_rmse"])
                self.assertIn(key,oof["component_macro_rmse"])
            for comparison in ("nested_kmer_esm_minus_train_mean",
                               "nested_kmer_esm_minus_nested_kmer"):
                self.assertIn(comparison,oof["bootstrap"]["comparisons"])

    def test_alignment_policy_does_not_claim_homology_certification(self):
        for rule,data in self.results.items():
            self.assertTrue("TRAIN_ONLY" in data["status"])
            self.assertTrue(any("not verified biological homology" in s for s in data["caveats"]))
            self.assertTrue(any("No frozen inner-development" in s for s in data["caveats"]))

if __name__=="__main__":unittest.main()
