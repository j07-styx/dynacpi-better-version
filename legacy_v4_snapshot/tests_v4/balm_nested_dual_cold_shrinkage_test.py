"""Tests for nested dual-cold CPI mean shrinkage without touching held-out data."""
import unittest
import numpy as np
from scripts.crossvalidate_balm_dual_cold_with_mean_bootstrap_dev import canonical_smiles
from scripts.train_balm_nested_dual_cold_mean_shrinkage_dev import (
    canonical_filter,BETAS,oof_bootstrap
)

class NestedDualColdPriorTests(unittest.TestCase):
    def test_canonical_filter_removes_all_identical_compounds(self):
        raw_train=[
            {"smiles":"OCC","protein":"A","group":"a"},
            {"smiles":"CCO","protein":"B","group":"b"},
            {"smiles":"CCN","protein":"C","group":"c"},
        ]
        validation=[{"smiles":"CCO","protein":"D","group":"d"}]
        c={s:canonical_smiles(s) for s in ("OCC","CCO","CCN")}
        result=canonical_filter(raw_train,validation,c)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["smiles"],"CCN")

    def test_stereoisomer_allowed_without_identity_leak(self):
        cis="N[C@@H](C)C(=O)O"
        trans="N[C@H](C)C(=O)O"
        c={s:canonical_smiles(s) for s in (cis,trans)}
        result=canonical_filter([{"smiles":cis}],[{"smiles":trans}],c)
        self.assertEqual(len(result),1)

    def test_fixed_prior_interpolation_anchors(self):
        self.assertEqual(BETAS[0],0.)
        self.assertEqual(BETAS[-1],1.)
        mean=np.array([1.,1.])
        prediction=np.array([5.,10.])
        np.testing.assert_allclose(mean+BETAS[0]*(prediction-mean),mean)
        np.testing.assert_allclose(mean+BETAS[-1]*(prediction-mean),prediction)

    def test_bootstrap_with_nested_mode(self):
        expected=("train_mean","ridge","ridge_plus_joint_boost","nested_selected_shrinkage")
        scores={name:[1.,2.,3.] for name in expected}
        result=oof_bootstrap(scores,[1,2,3],replicates=100,seed=10)
        key="nested_selected_shrinkage_minus_train_mean"
        self.assertIn(key,result["paired_bootstrap"]["controls"])
        self.assertAlmostEqual(result["paired_bootstrap"]["controls"][key]["pooled_delta"],0)

if __name__=="__main__":
    unittest.main()


class NestedDualColdReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path
        cls.data=json.loads(
            (Path(__file__).resolve().parents[1]/
             "research/balm_bindingdb_nested_dual_cold_mean_shrinkage_paired_bootstrap_dev.json").read_text())

    def test_all_outer_rows_components_and_coldness(self):
        d=self.data
        self.assertEqual(d["original_train_rows"],15538)
        self.assertEqual(d["outer_fold_count"],4)
        self.assertEqual(sum(f["heldout_rows"] for f in d["outer_folds"]),15538)
        self.assertEqual(sum(f["heldout_components"] for f in d["outer_folds"]),519)
        self.assertTrue(all(f["heldout_canonical_compound_overlap_count"]==0
                            for f in d["outer_folds"]))

    def test_three_nested_folds_per_outer_and_selection(self):
        d=self.data
        allowed=set(d["predeclared_betas"])
        self.assertEqual([f["inner_selected_beta"] for f in d["outer_folds"]],
                         [.75,1.,.75,.75])
        for f in d["outer_folds"]:
            self.assertIn(f["inner_selected_beta"],allowed)
            self.assertEqual(len(f["inner_dual_cold_folds"]),3)
            for inner in f["inner_dual_cold_folds"]:
                self.assertEqual(set(inner["scores"]),set(str(x) for x in allowed))
                self.assertGreater(inner["fit_rows"],300)

    def test_nested_scores_reconcile_with_unchanged_controls(self):
        d=self.data["outer_oof"]
        self.assertAlmostEqual(d["pooled_rmse"]["train_mean"],1.4450910501878775,places=8)
        self.assertAlmostEqual(d["pooled_rmse"]["ridge_plus_joint_boost"],1.4244631296206303,places=8)
        self.assertAlmostEqual(d["pooled_rmse"]["nested_selected_shrinkage"],1.4021539029340628,places=8)
        self.assertLess(d["macro_component_rmse"]["nested_selected_shrinkage"],
                        d["macro_component_rmse"]["train_mean"])

    def test_paired_intervals_reproduce_interpretation(self):
        boot=self.data["outer_oof"]["paired_bootstrap"]
        self.assertEqual(boot["replicates"],5000)
        c=boot["controls"]["nested_selected_shrinkage_minus_ridge_plus_joint_boost"]
        self.assertLess(c["pooled_95pct_group_bootstrap_CI"][1],0)
        self.assertLess(c["macro_95pct_group_bootstrap_CI"][1],0)
        b=boot["controls"]["nested_selected_shrinkage_minus_train_mean"]
        self.assertLess(b["pooled_95pct_group_bootstrap_CI"][0],0)
        self.assertGreater(b["pooled_95pct_group_bootstrap_CI"][1],0)
        self.assertLess(b["macro_95pct_group_bootstrap_CI"][1],0)

if __name__=="__main__":
    unittest.main()

