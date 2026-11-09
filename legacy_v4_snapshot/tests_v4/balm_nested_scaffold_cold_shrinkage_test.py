"""Strict Murcko-scaffold cold split and nested metrics unit tests."""
import unittest
import numpy as np
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import (
    scaffold_key,scaffold_filter,BETAS
)
from scripts.crossvalidate_balm_dual_cold_with_mean_bootstrap_dev import oof_bootstrap

class ScaffoldColdTests(unittest.TestCase):
    def test_two_benzene_derivatives_share_scaffold(self):
        self.assertEqual(scaffold_key("Cc1ccccc1"),scaffold_key("CCc1ccccc1"))
        self.assertNotEqual(scaffold_key("c1ccccc1"),scaffold_key("C1CCCCC1"))

    def test_acyclic_group_shared_by_all_acyclic_compounds(self):
        self.assertEqual(scaffold_key("CCO"),"ACYCLIC_SINGLE_GROUP")
        self.assertEqual(scaffold_key("CC(=O)O"),"ACYCLIC_SINGLE_GROUP")

    def test_filter_excludes_identical_scaffold_derivatives(self):
        records=[{"smiles":"Cc1ccccc1"},{"smiles":"CCc1ccccc1"},
                 {"smiles":"C1CCCCC1"},{"smiles":"CCO"}]
        val=[{"smiles":"c1ccccc1"},{"smiles":"CC(=O)O"}]
        keys={s:scaffold_key(s) for s in {r["smiles"] for r in records+val}}
        held=scaffold_filter(records,val,keys)
        self.assertEqual(len(held),1)
        self.assertEqual(held[0]["smiles"],"C1CCCCC1")

    def test_invalid_smiles_rejected(self):
        with self.assertRaises(ValueError):scaffold_key("not_a_real_smiles")

    def test_output_still_shrinks_to_fold_mean(self):
        mu=1.5
        predicted=2.5
        self.assertEqual(mu+BETAS[0]*(predicted-mu),mu)
        self.assertEqual(mu+BETAS[-1]*(predicted-mu),predicted)

    def test_group_bootstrap_contains_mean_and_unshrunk_controls(self):
        sums={name:[2.,3.] for name in ("train_mean","ridge",
             "ridge_plus_joint_boost","nested_selected_shrinkage")}
        stats=oof_bootstrap(sums,[3,4],replicates=64,seed=13)
        self.assertIn("nested_selected_shrinkage_minus_train_mean",
                      stats["paired_bootstrap"]["controls"])
        self.assertIn("nested_selected_shrinkage_minus_ridge_plus_joint_boost",
                      stats["paired_bootstrap"]["controls"])

if __name__=="__main__":
    unittest.main()

class ScaffoldColdReportIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path
        cls.data=json.loads(
            (Path(__file__).resolve().parents[1]/
             "research/balm_bindingdb_nested_scaffold_cold_mean_shrinkage_traincv_dev.json").read_text())

    def test_strict_cold_coverage(self):
        d=self.data
        folds=d["outer_folds"]
        self.assertEqual(d["original_train_rows"],15538)
        self.assertEqual(d["outer_fold_count"],4)
        self.assertEqual(sum(x["heldout_rows"] for x in folds),15538)
        self.assertEqual(sum(x["heldout_components"] for x in folds),519)
        self.assertGreater(d["murcko_scaffold_group_count_in_train"],2000)
        self.assertTrue(all(x["heldout_murcko_scaffold_overlap_count"]==0 for x in folds))

    def test_beta_selection_and_inner_folds(self):
        d=self.data
        folds=d["outer_folds"]
        self.assertEqual([x["inner_selected_beta"] for x in folds],[1.,1.,1.,.75])
        for f in folds:
            self.assertEqual(len(f["inner_dual_cold_folds"]),3)
            self.assertGreater(f["outer_fit_unique_pairs"],3000)
            for inner in f["inner_dual_cold_folds"]:
                self.assertGreater(inner["fit_rows"],300)

    def test_macro_mean_improvement_and_pooled_uncertainty(self):
        result=self.data["outer_oof"]
        self.assertLess(result["macro_component_rmse"]["nested_selected_shrinkage"],
                        result["macro_component_rmse"]["train_mean"])
        self.assertLess(result["pooled_rmse"]["nested_selected_shrinkage"],
                        result["pooled_rmse"]["ridge_plus_joint_boost"])
        comps=result["paired_bootstrap"]["controls"]
        a=comps["nested_selected_shrinkage_minus_ridge_plus_joint_boost"]
        b=comps["nested_selected_shrinkage_minus_train_mean"]
        self.assertLess(a["pooled_95pct_group_bootstrap_CI"][1],0)
        self.assertLess(a["macro_95pct_group_bootstrap_CI"][1],0)
        self.assertLess(b["pooled_95pct_group_bootstrap_CI"][0],0)
        self.assertGreater(b["pooled_95pct_group_bootstrap_CI"][1],0)

    def test_model_metrics_reconcile(self):
        d=self.data["outer_oof"]
        self.assertAlmostEqual(d["pooled_rmse"]["nested_selected_shrinkage"],1.412784189201056)
        self.assertAlmostEqual(d["macro_component_rmse"]["nested_selected_shrinkage"],1.3194295640534692)

if __name__=="__main__":
    unittest.main()

