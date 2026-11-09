"""End-to-end report integrity for BALM TRAIN-only affinity disagreement controls."""
import json,unittest
from dynacpi.balm_public_data_dev import ROOT

RES=ROOT/"research"
RULES=("identity30_cover80","identity40_cover80")

class BALMTrainAffinitySensitivityReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=json.loads((RES/"balm_train_only_affinity_replicate_disagreement_audit_20261108.json").read_text())
        cls.original={
            rule:json.loads((RES/
                ("balm_esm8m_nested_sw_"+rule+"_murcko_scaffold_cold_traincv_dev.json")).read_text())
            for rule in RULES
        }
        cls.results={
            rule:json.loads((RES/
                ("balm_sw_"+rule+"_affinity_disagreement_training_ablation_dev.json")).read_text())
            for rule in RULES
        }
    def test_original_affinity_disagreement_only_train(self):
        self.assertEqual(self.audit["raw_training_rows"],15538)
        self.assertEqual(self.audit["conflict_thresholds_log_affinity_units"]["1.0"]["original_rows_in_groups"],2542)

    def test_all_heldout_rows_scaffold_and_alignment_groups(self):
        for rule in RULES:
            d=self.results[rule]
            old=self.original[rule]
            self.assertEqual(d["alignment_group_rule"],rule)
            folds=d["folds"]
            self.assertEqual(len(folds),4)
            self.assertEqual(sum(f["heldout_original_pairs"] for f in folds),15538)
            self.assertEqual(
                sum(f["heldout_alignment_groups"] for f in folds),
                old["heldout_alignment_group_count"])
            self.assertTrue(all(f["scaffold_overlap"]==0 for f in folds))
            self.assertTrue(all(f["training_unique_after_conflict_exclusion"]<=
                                f["training_unique_original"] for f in folds))
            self.assertTrue(all(f["removed_groups"]["discarded_original_training_rows"]>0
                                for f in folds))

    def test_unmodified_base_model_reconstructs_original_cv(self):
        for rule in RULES:
            new=self.results[rule]["out_of_fold"]
            old=self.original[rule]["out_of_fold"]
            self.assertAlmostEqual(
                new["pooled_rmse"]["original_ensemble"],
                old["pooled_rmse"]["nested_kmer_esm"],places=8)
            self.assertAlmostEqual(
                new["component_macro_rmse"]["original_ensemble"],
                old["component_macro_rmse"]["nested_kmer_esm"],places=8)

    def test_paired_bootstrap_and_explicit_treatments(self):
        for rule in RULES:
            result=self.results[rule]["out_of_fold"]
            self.assertEqual(result["bootstrap"]["replicates"],5000)
            for name in ("original_ensemble","median_ensemble","exclude_ge1_ensemble"):
                self.assertIn(name,result["pooled_rmse"])
                self.assertIn(name,result["component_macro_rmse"])
            for key in ("median_ensemble_minus_original_ensemble",
                        "exclude_ge1_ensemble_minus_original_ensemble"):
                self.assertIn(key,result["bootstrap"]["comparisons"])

if __name__=="__main__":unittest.main()
