"""Reconcile ESM-conditioned gradient booster results with prior cold CV."""
import json
import unittest
from dynacpi.balm_public_data_dev import ROOT

class ESMBoosterReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data={}
        cls.prior={}
        for rule in ("identity30_cover80","identity40_cover80"):
            cls.data[rule]=json.loads((ROOT/"research"/
                f"balm_esm_conditioned_booster_sw_{rule}_scaffold_cold_dev.json").read_text())
            cls.prior[rule]=json.loads((ROOT/"research"/
                f"balm_esm8m_nested_sw_{rule}_murcko_scaffold_cold_traincv_dev.json").read_text())

    def test_train_only_scaffold_cold_outer_coverage(self):
        for result in self.data.values():
            folds=result["outer_folds"]
            self.assertEqual(len(folds),4)
            self.assertEqual(sum(f["heldout_original_pairs"] for f in folds),15538)
            self.assertEqual(sum(f["heldout_alignment_components"] for f in folds),
                             result["alignment_component_count"])
            self.assertTrue(all(f["scaffold_overlap"]==0 for f in folds))

    def test_reference_model_reproduces_prior_scores(self):
        for rule,result in self.data.items():
            ref=self.prior[rule]["out_of_fold"]
            got=result["out_of_fold"]
            for m in ("pooled_rmse","component_macro_rmse"):
                self.assertAlmostEqual(got[m]["nested_reference_joint"],
                                       ref[m]["nested_kmer_esm"],places=8)

    def test_pca_fit_unique_proteins_inside_outer_fit(self):
        for result in self.data.values():
            for fold in result["outer_folds"]:
                self.assertEqual(set(fold["pca_fit"]),{"16","32"})
                self.assertIn(fold["inherited_beta"],[0,.25,.5,.75,1])
                for n,log in fold["pca_fit"].items():
                    self.assertGreater(log["training_unique_proteins"],int(n)+2)
                    self.assertGreater(log["training_variance_explained"],.5)
                    self.assertLess(log["training_variance_explained"],1.01)

    def test_group_bootstrap_includes_fixed_control(self):
        for result in self.data.values():
            boot=result["out_of_fold"]["bootstrap"]
            self.assertEqual(boot["replicates"],5000)
            for mode in ("nested_esm_pca16_joint","nested_esm_pca32_joint",
                         "nested_esm_pca32_no_aa"):
                self.assertIn(mode+"_minus_nested_reference_joint",boot["comparisons"])

if __name__=="__main__":
    unittest.main()
