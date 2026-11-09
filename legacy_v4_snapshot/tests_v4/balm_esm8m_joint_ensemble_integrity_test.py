"""Report integrity: frozen ESM2 + ordinary joint ensemble strict scaffold-cold."""
import json,unittest
from pathlib import Path
from dynacpi.balm_public_data_dev import ROOT

class ESM8MJointEnsembleReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/"research"
        cls.control=json.loads((base/"balm_bindingdb_nested_scaffold_cold_mean_shrinkage_traincv_dev.json").read_text())
        cls.result=json.loads((base/"balm_esm8m_augmented_joint_ensemble_scaffold_cold_traincv_dev.json").read_text())
        cls.ridge=json.loads((base/"balm_esm2_8m_vs_kmer_strict_scaffold_cold_traincv_dev.json").read_text())
        cls.manifest=json.loads((base/"balm_esm2_t6_8m_train_only_feature_manifest_dev.json").read_text())

    def test_manifest_and_train_only_scaffold_coverage(self):
        d=self.result
        self.assertEqual(d["feature_cache_sha256"],self.manifest["cache_sha256"])
        self.assertEqual(sum(f["holdout_original_pairs"] for f in d["folds"]),15538)
        self.assertEqual(sum(f["heldout_components"] for f in d["folds"]),519)
        self.assertTrue(all(f["murcko_overlap_count"]==0 for f in d["folds"]))
        self.assertEqual(d["inherited_betas"],[1.,1.,1.,.75])

    def test_standard_reproduces_prior_architecture_exactly(self):
        old=self.control["outer_oof"]
        new=self.result["out_of_fold"]
        self.assertAlmostEqual(new["pooled_rmse"]["nested_standard"],
                               old["pooled_rmse"]["nested_selected_shrinkage"],places=7)
        self.assertAlmostEqual(new["component_macro_rmse"]["nested_standard"],
                               old["macro_component_rmse"]["nested_selected_shrinkage"],places=7)

    def test_esm_kmer_feature_ridge_reproduces_standalone_ablation(self):
        reference=self.ridge["out_of_fold"]
        got=self.result["out_of_fold"]
        self.assertAlmostEqual(got["pooled_rmse"]["ridge_kmer_esm8m"],
                               reference["pooled_rmse"]["chemical_kmer_esm8m"],places=7)
        self.assertAlmostEqual(got["component_macro_rmse"]["ridge_kmer_esm8m"],
                               reference["component_macro_rmse"]["chemical_kmer_esm8m"],places=7)

    def test_macro_benefit_but_pooled_uncertain(self):
        result=self.result["out_of_fold"]
        self.assertLess(result["component_macro_rmse"]["nested_esm8m_with_inherited_beta"],
                        result["component_macro_rmse"]["nested_standard"])
        comparison=result["bootstrap"]["comparisons"]["nested_esm8m_with_inherited_beta_minus_nested_standard"]
        self.assertLess(comparison["macro_CI95"][1],0)
        self.assertLess(comparison["pooled_CI95"][0],0)
        self.assertGreater(comparison["pooled_CI95"][1],0)

if __name__=="__main__":unittest.main()
