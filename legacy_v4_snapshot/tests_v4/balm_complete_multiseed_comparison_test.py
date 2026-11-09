"""Regression/rigor checks for completed multi-seed matched BALM development audit."""
import hashlib,json,unittest
from pathlib import Path

import numpy as np

from dynacpi.balm_public_data_dev import ROOT,get_train_dev
from scripts.compare_balm_perceiver_vs_ensemble_dev import row_hash
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import metric

RES=ROOT/"research"
SEED_DIR=RES/"balm_perceiver_multiseed_dev_20261108"
EXPECTED={10:[2103,2601,2602,2603,2604,2605],30:[2601,2602]}


class BalmMatchedMultiSeedCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev=get_train_dev()["inner_dev"]
        cls.devhash=row_hash(cls.dev)
        cls.audit=json.loads((RES/"balm_matched_multiseed_complete_audit_20261108.json").read_text())
        cls.memory=json.loads((RES/"balm_memory_vs_perceiver_matched_innerdev_20261108.json").read_text())
        cls.seeds={
            ep:{s:json.loads((SEED_DIR/f"perceiver_seed{s}_ep{ep}.json").read_text())
                for s in ids} for ep,ids in EXPECTED.items()
        }

    def test_exact_eight_runs_and_same_protein_cold_development_rows(self):
        self.assertEqual(len(list(SEED_DIR.glob("perceiver_seed*_ep*.json"))),8)
        self.assertEqual(len(self.dev),3891)
        self.assertEqual(len(set(x["group"] for x in self.dev)),173)
        self.assertEqual(self.audit["row_sha256"],self.devhash)
        for ep,runset in self.seeds.items():
            self.assertEqual(sorted(runset),EXPECTED[ep])
            for seed,run in runset.items():
                self.assertEqual(run["dev_row_sha256"],self.devhash)
                self.assertEqual(len(run["history"]),ep)
                self.assertEqual(len(run["dev_predictions_by_epoch"]),ep)
                self.assertEqual(run["epochs"],ep)
                self.assertEqual(run["seed"],seed)
                self.assertEqual(len(run["dev_predictions_by_epoch"][-1]),3891)
                self.assertTrue(np.isfinite(run["dev_predictions_by_epoch"][-1]).all())

    def test_fair_10_vs_30_epoch_identical_prefix_per_seed(self):
        self.assertEqual(set(self.seeds[30]),{2601,2602})
        for seed in (2601,2602):
            a=self.seeds[10][seed]
            b=self.seeds[30][seed]
            np.testing.assert_array_equal(
                np.asarray(a["dev_predictions_by_epoch"]),
                np.asarray(b["dev_predictions_by_epoch"][:10]))
        for seed,best in ((2601,12),(2602,14)):
            result=self.audit["seed_budgets"]["30"]["per_seed"]
            row=next(r for r in result if r["seed"]==seed)
            self.assertEqual(row["best_dev_pooled_epoch"],best)
            self.assertGreater(row["final_minus_best_pooled"],0)

    def test_prespecified_primary_winning_rule_and_paired_5000_bootstraps(self):
        for ep in EXPECTED:
            report=json.loads((RES/
                ("balm_perceiver_vs_ensemble_paired_dev_20261108.json" if ep==10
                 else "balm_perceiver_vs_ensemble_paired_dev_20261108_ep30.json")).read_text())
            self.assertEqual(report["seeds"],EXPECTED[ep])
            self.assertEqual(report["bootstrap_draws"],5000)
            for type_ in ("FINAL_EPOCH_PRIMARY","BEST_DEV_EPOCH_OPTIMISTIC",
                          "SEED_AVERAGED_PREDICTIONS_FINAL"):
                comp=report["comparisons"][f"ridge_interaction_blend_0.25_vs_perceiver_{type_}"]
                self.assertLess(comp["ci95_pooled"][1],0)
                self.assertLess(comp["ci95_macro"][1],0)

    def test_memory_source_checkpoint_and_dev_input_cache(self):
        meta=json.loads((RES/"balm_esm2_t6_8m_innerdev_input_only_manifest_dev_20261108.json").read_text())
        path=ROOT/meta["local_cache"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),meta["cache_sha256"])
        self.assertEqual(meta["unique_dev_proteins"],209)
        self.assertFalse(meta["dev_target_labels_used_for_embedding"])
        self.assertFalse(meta["reserved_prospective_test_sequences_or_labels_accessed"])
        self.assertTrue(self.memory["no_reserved_test_data_access"])
        self.assertEqual(self.memory["dev_row_sha256"],self.devhash)
        self.assertEqual(set(self.memory["perceiver_seeds_by_budget"]["10"]),set(EXPECTED[10]))

    def test_dev_predictions_reproduce_scored_metrics(self):
        y=np.asarray([r["label"] for r in self.dev],dtype=np.float64)
        groups=np.asarray([r["group"] for r in self.dev])
        self.assertEqual(len(self.memory["predictions_by_model"]),6)
        for name,pred in self.memory["predictions_by_model"].items():
            values=np.asarray(pred,dtype=np.float64)
            self.assertEqual(len(values),3891)
            obtained=metric(y,values,groups)
            for k,expected in self.memory["model_metrics"][name].items():
                self.assertLess(abs(obtained[k]-expected),1e-5)
        paired=self.memory["paired_group_bootstrap"]
        novelty=paired["memory_joint_esm8_k64_minus_locked_explicit_interaction_ensemble"]
        self.assertLess(novelty["ci95_pooled"][0],0)
        self.assertGreater(novelty["ci95_pooled"][1],0)
        self.assertLess(novelty["ci95_macro"][0],0)
        self.assertGreater(novelty["ci95_macro"][1],0)

    def test_deepdta_dev_optimized_counterexample_is_preserved(self):
        deep=self.audit["deepdta_inspired_control"]
        locked=self.audit["locked_classic_metrics"]["pooled_rmse"]
        self.assertLess(deep["optimistic_best_dev_pooled"],locked)
        self.assertGreater(deep["final_epoch_pooled"],locked)
        self.assertIn("NO_PROSPECTIVE_TEST",self.audit["status"])

if __name__=="__main__":unittest.main()
