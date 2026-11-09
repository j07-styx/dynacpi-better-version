"""Derangement control integrity and negative-control protocol tests."""
import unittest
import numpy as np
from scripts.ablate_balm_esm8m_protein_derangement_scaffold_dev import shuffled_feature_map,SEEDS
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import cache_feature_map

class FrozenESMNegativeControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real,_=cache_feature_map()
    def test_global_derangements_reassign_unique_values(self):
        outputs=[]
        sample_keys=sorted(self.real)
        source_values={v.tobytes() for v in self.real.values()}
        self.assertEqual(len(source_values),len(sample_keys))
        for seed in SEEDS:
            swapped,h=shuffled_feature_map(self.real,seed)
            self.assertEqual(set(swapped),set(self.real))
            self.assertEqual(len(h),64)
            self.assertTrue(all(not np.array_equal(self.real[s],swapped[s]) for s in sample_keys))
            self.assertEqual(source_values,{v.tobytes() for v in swapped.values()})
            outputs.append(h)
        self.assertEqual(len(set(outputs)),len(SEEDS))
    def test_same_seed_is_deterministic(self):
        left,h1=shuffled_feature_map(self.real,2103)
        right,h2=shuffled_feature_map(self.real,2103)
        self.assertEqual(h1,h2)
        for key in list(self.real)[:12]:
            np.testing.assert_array_equal(left[key],right[key])



class FrozenESMDerangementReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]/"research"
        cls.report=json.loads(
            (root/"balm_esm8m_protein_derangement_negative_controls_scaffold_cv_dev.json").read_text())
        cls.reference=json.loads(
            (root/"balm_esm8m_augmented_joint_ensemble_scaffold_cold_traincv_dev.json").read_text())

    def test_negative_control_metadata_and_fold_coldness(self):
        report=self.report
        self.assertEqual(report["derangement_seeds"],[2103,2104,2105])
        self.assertEqual(len(set(report["derangement_mapping_sha256"].values())),3)
        self.assertEqual(len(report["folds"]),4)
        self.assertEqual(sum(f["holdout_rows"] for f in report["folds"]),15538)
        self.assertEqual(sum(f["heldout_components"] for f in report["folds"]),519)
        self.assertTrue(all(f["scaffold_overlap_count"]==0 for f in report["folds"]))

    def test_unchanged_true_esm_matches_independent_previous_report(self):
        old=self.reference["out_of_fold"]
        cur=self.report["out_of_fold"]
        self.assertAlmostEqual(
            cur["pooled_rmse"]["real_esm"],
            old["pooled_rmse"]["nested_esm8m_with_inherited_beta"],places=8)
        self.assertAlmostEqual(
            cur["component_macro_rmse"]["real_esm"],
            old["component_macro_rmse"]["nested_esm8m_with_inherited_beta"],places=8)

    def test_all_derangements_worse_macro_with_positive_group_bootstrap(self):
        cur=self.report["out_of_fold"]
        reference=cur["component_macro_rmse"]["real_esm"]
        for seed in SEEDS:
            name="deranged_esm_"+str(seed)
            self.assertGreater(cur["component_macro_rmse"][name],reference)
            comparison=cur["bootstrap"]["comparisons"][name+"_minus_real_esm"]
            self.assertGreater(comparison["macro_CI95"][0],0)
            self.assertLess(comparison["pooled_CI95"][0],0)
            self.assertGreater(comparison["pooled_CI95"][1],0)

if __name__=="__main__":unittest.main()

