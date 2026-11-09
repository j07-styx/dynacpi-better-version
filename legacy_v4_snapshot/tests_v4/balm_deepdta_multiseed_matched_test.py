"""Matched neural family comparison bootstrap and cohort integrity tests."""
import json,unittest
import numpy as np

from dynacpi.balm_public_data_dev import ROOT,get_train_dev
from scripts.compare_balm_deepdta_multiseed_matched_dev import (
    paired_seed_families,RULES,
)
from scripts.compare_balm_perceiver_vs_ensemble_dev import row_hash

RES=ROOT/"research"
DEEP=RES/"balm_deepdta_multiseed_dev_20261108"

class DeepDTAMultiSeedMatchedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dev=get_train_dev()["inner_dev"]
        cls.sha=row_hash(dev)
        cls.dev=dev

    def test_seed_family_bootstrap_identical_models_yield_zero(self):
        x=np.asarray([[2.,3.,4.],[4.,1.,2.]])
        n=np.asarray([1.,2.,4.])
        metrics=paired_seed_families(x,x,n,draws=256)
        self.assertAlmostEqual(metrics["delta_a_minus_b_pooled"],0.)
        self.assertAlmostEqual(metrics["delta_a_minus_b_macro"],0.)
        np.testing.assert_allclose(metrics["ci95_pooled"],[0.,0.],atol=1e-12)
        np.testing.assert_allclose(metrics["ci95_macro"],[0.,0.],atol=1e-12)

    def test_seed_family_bootstrap_detects_clear_better(self):
        low=np.full((2,4),2.)
        high=np.full((3,4),8.)
        n=np.ones(4)
        met=paired_seed_families(low,high,n,draws=256)
        self.assertLess(met["ci95_pooled"][1],0)
        self.assertLess(met["ci95_macro"][1],0)

    def test_bootstrap_reproducibility_and_seed_order_independence(self):
        a=np.asarray([[2.,3.,1.],[6.,1.,2.]])
        b=np.asarray([[3.,2.,1.],[4.,3.,2.]])
        n=np.asarray([2.,4.,6.])
        first=paired_seed_families(a,b,n,draws=128)
        second=paired_seed_families(a[::-1],b[::-1],n,draws=128)
        self.assertAlmostEqual(first["delta_a_minus_b_pooled"],
                               second["delta_a_minus_b_pooled"])
        np.testing.assert_allclose(first["ci95_pooled"],second["ci95_pooled"],atol=1e-14)

    def test_training_and_dev_row_hash_is_stable(self):
        self.assertEqual(len(self.dev),3891)
        self.assertEqual(len(set(r["group"] for r in self.dev)),173)
        for ep in (10,30):
            for seed in RULES[ep]:
                path=DEEP/f"deepdta_seed{seed}_ep{ep}.json"
                report=json.loads(path.read_text())
                self.assertEqual(report["dev_row_sha256"],self.sha)
                self.assertEqual(report["train_rows"],12315)
                self.assertEqual(report["dev_rows"],3891)
                self.assertEqual(report["seed"],seed)
                self.assertEqual(report["epochs"],ep)
                self.assertEqual(len(report["history"]),ep)
                self.assertEqual(len(report["dev_predictions_by_epoch"]),ep)
                self.assertEqual(len(report["dev_predictions_by_epoch"][-1]),3891)
                self.assertEqual(report["dev_protein_groups"],173)

    def test_first_epoch_replays_but_later_epochs_diverge(self):
        """Observed baseline limitation, not a demand for identical trajectories."""
        audit=json.loads((RES/"balm_deepdta_budget_replay_divergence_audit_20261108.json").read_text())
        for seed in RULES[30]:
            p10=DEEP/f"deepdta_seed{seed}_ep10.json"
            p30=DEEP/f"deepdta_seed{seed}_ep30.json"
            d10=json.loads(p10.read_text())
            d30=json.loads(p30.read_text())
            np.testing.assert_array_equal(
                d10["dev_predictions_by_epoch"][0],
                d30["dev_predictions_by_epoch"][0])
            detail=audit["seed_diagnostics"][str(seed)]
            self.assertTrue(detail["same_initial_epoch_predictions"])
            self.assertEqual(detail["first_divergent_epoch"],2)
            self.assertGreater(detail["max_abs_prediction_difference_epoch2_to10"],0)

    def test_report_stats_include_seed_mean_and_single_locked_baseline(self):
        for ep in (10,30):
            p=RES/f"balm_deepdta_vs_perceiver_and_cpi_paired_dev_20261108_ep{ep}.json"
            result=json.loads(p.read_text())
            self.assertEqual(result["seeds"],RULES[ep])
            self.assertEqual(result["dev_row_sha256"],self.sha)
            self.assertIn("deepdta",result["seed_family_metrics"])
            self.assertIn("perceivercpi",result["seed_family_metrics"])
            comparisons=result["comparisons"]
            for name in (
                "locked_ridge_interactions_minus_deepdta_seedmean_FINAL",
                "locked_ridge_interactions_minus_deepdta_BEST_DEV_EPOCH_OPTIMISTIC",
                "deepdta_seedmean_minus_perceiver_seedmean_FINAL",
            ):
                self.assertIn(name,comparisons)

if __name__=="__main__":
    unittest.main()

