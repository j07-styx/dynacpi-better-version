"""Reproducibility integrity checks for matched BALM boost controls."""
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/"research"

class MatchedBoosterReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads((RES/"balm_bindingdb_explicit_chemical_protein_interaction_ensemble_dev.json").read_text())
        cls.control=json.loads((RES/"balm_bindingdb_ridge_boosting_interaction_controls_dev.json").read_text())
        cls.bootstrap=json.loads((RES/"balm_bindingdb_boost_ensemble_control_group_bootstrap_dev.json").read_text())

    def test_cohort_and_fixed_coefficient(self):
        self.assertEqual(self.control["inner_dev_rows"],3891)
        self.assertEqual(self.bootstrap["dev_rows"],3891)
        self.assertEqual(self.bootstrap["heldout_protein_components"],173)
        self.assertEqual(self.control["fixed_blend_weight"],.25)

    def test_reproduces_original_baseline(self):
        original=self.original["inner_dev_candidate_metrics"]
        current=self.control["development_metrics"]
        self.assertAlmostEqual(original["ridge"]["pooled_rmse"],current["ridge"]["pooled_rmse"],places=9)
        self.assertAlmostEqual(original["ridge_interaction_blend_0.25"]["pooled_rmse"],
                               current["ridge_plus_explicit_interaction_boost"]["pooled_rmse"],places=9)

    def test_bootstrap_reconciles(self):
        for name,metrics in self.control["development_metrics"].items():
            self.assertAlmostEqual(metrics["pooled_rmse"],self.bootstrap["rmse_pooled"][name],places=9)
            self.assertAlmostEqual(metrics["component_macro_rmse"],self.bootstrap["rmse_macro"][name],places=9)

    def test_three_folds(self):
        self.assertEqual(len(self.control["train_only_groupcv_folds"]),3)
        for f in self.control["train_only_groupcv_folds"]:
            self.assertEqual(set(f["results"]),{
                "ridge","ridge_plus_chemical_boost","ridge_plus_joint_boost",
                "ridge_plus_explicit_interaction_boost"})

if __name__=="__main__": unittest.main()
