"""Invariants of train-only dual protein/canonical-chemical cold evaluation."""
import unittest
import numpy as np
from scripts.crossvalidate_balm_dual_cold_ensemble_train_only_dev import (
    canonical_smiles,oof_bootstrap,MODES)

class DualColdCrossValidationTests(unittest.TestCase):
    def test_equivalent_smiles_canonicalize(self):
        self.assertEqual(canonical_smiles("CCO"),canonical_smiles("OCC"))
        self.assertEqual(canonical_smiles("C(C)O"),canonical_smiles("CCO"))
    def test_stereochemistry_retained(self):
        self.assertNotEqual(canonical_smiles("N[C@@H](C)C(=O)O"),
                            canonical_smiles("N[C@H](C)C(=O)O"))
    def test_invalid_smiles_rejected(self):
        with self.assertRaises(ValueError):
            canonical_smiles("not_a_molecule")
    def test_group_bootstrap_zero_delta(self):
        sums={name:[4.,9.,16.] for name in MODES}
        out=oof_bootstrap(sums,[1,2,4],replicates=64,seed=45)
        for name in MODES:
            if name=="ridge":continue
            self.assertAlmostEqual(out["paired_bootstrap"]["controls"][name+"_minus_ridge"]["pooled_delta"],0)
            self.assertAlmostEqual(out["paired_bootstrap"]["controls"][name+"_minus_ridge"]["macro_delta"],0)
    def test_bootstrap_direction(self):
        sums={name:[1.,1.] for name in MODES}
        sums[MODES[2]]=[.5,.5]
        out=oof_bootstrap(sums,[1,1],replicates=50)
        self.assertLess(out["paired_bootstrap"]["controls"][MODES[2]+"_minus_ridge"]["pooled_delta"],0)




class DualColdReportIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        import json
        cls.data=json.loads(
            (Path(__file__).resolve().parents[1]/
             "research/balm_bindingdb_train_only_dual_cold_group_cv_with_mean_bootstrap.json").read_text())
    def test_no_canonical_compound_overlap_and_full_coverage(self):
        folds=self.data["cv_folds"]
        self.assertEqual(len(folds),4)
        self.assertEqual(sum(x["holdout_rows"] for x in folds),15538)
        self.assertEqual(sum(x["holdout_protein_components"] for x in folds),519)
        for fold in folds:
            self.assertEqual(fold["canonical_compound_overlap_count"],0)
            self.assertGreater(fold["dual_cold_train_unique_rows"],1000)
    def test_mean_baseline_and_bootstrap_present(self):
        scores=self.data["pooled_out_of_fold_across_nonoverlapping_4_folds"]
        self.assertEqual(scores["paired_bootstrap"]["replicates"],5000)
        for k in ("train_mean","ridge","ridge_plus_joint_boost"):
            self.assertIn(k,scores["pooled_rmse"])
            self.assertIn(k,scores["macro_component_rmse"])
        self.assertIn("ridge_plus_joint_boost_minus_train_mean",scores["paired_bootstrap"]["controls"])
    def test_ensemble_beats_ridge_but_mean_uncertainty_overlaps(self):
        scores=self.data["pooled_out_of_fold_across_nonoverlapping_4_folds"]
        self.assertLess(scores["pooled_rmse"]["ridge_plus_joint_boost"],scores["pooled_rmse"]["ridge"])
        comp=scores["paired_bootstrap"]["controls"]["ridge_plus_joint_boost_minus_train_mean"]
        lower,upper=comp["pooled_95pct_group_bootstrap_CI"]
        self.assertLess(lower,0)
        self.assertGreater(upper,0)

if __name__=="__main__":unittest.main()

