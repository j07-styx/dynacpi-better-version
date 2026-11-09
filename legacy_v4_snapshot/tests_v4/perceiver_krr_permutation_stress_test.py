"""Reproducibility and safety of synthetic permutation-guard stress.

No assay data, models, or reserved-test results are ever consulted.
"""
from __future__ import annotations
from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=spec_from_file_location(
    "fabricated_permutation_stress",
    ROOT/"scripts/stress_perceiver_krr_permutation_null_synthetic.py"
)
m=module_from_spec(spec)
spec.loader.exec_module(m)


class FabricatedPermutationGuardStressTests(unittest.TestCase):
    def test_same_seed_exact_aggregate_replay(self):
        a=m.synthetic_permutation_stress(n_trials=16,seed=2601)
        b=m.synthetic_permutation_stress(n_trials=16,seed=2601)
        self.assertEqual(a,b)

    def test_stress_cases_cover_signal_null_and_k5_k6(self):
        result=m.synthetic_permutation_stress(n_trials=16,seed=2601)
        self.assertEqual(set(result["results"]),{
            "geometry_signal_K5","geometry_signal_K6",
            "geometry_null_K5","geometry_null_K6"})
        for key,row in result["results"].items():
            self.assertEqual(row["fabricated_trials"],16)
            self.assertEqual(row["fictional_query_count_per_trial"],
                             8-row["support_count"])
            self.assertGreaterEqual(row["old_LOO_gate_active_trials"],
                                    row["permutation_guard_applied_trials"])
            self.assertEqual(row["old_LOO_gate_active_trials"],
                             row["permutation_nulls_computed_trials"])

    def test_public_report_exact_seed_counts_and_no_real_outcomes(self):
        if not m.OUTPUT.exists():
            self.skipTest("Public fabricated report not generated")
        saved=json.loads(m.OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"],
            "FABRICATED_SUPPORT_PERMUTATION_RELIABILITY_ONLY_NOT_CPI_EVIDENCE")
        self.assertEqual(saved["trial_count_per_case"],128)
        self.assertEqual(saved["n_support_null_permutations"],49)
        self.assertFalse(saved["test_split_used"])
        self.assertFalse(saved["real_affinity_labels_used"])
        self.assertFalse(saved["neural_training_performed"])
        self.assertNotIn("source_family_sha256",json.dumps(saved))
        self.assertNotIn("median_pKd",json.dumps(saved))

    def test_fabricated_noise_false_activations_decreased(self):
        saved=json.loads(m.OUTPUT.read_text(encoding="utf-8"))
        for k in (5,6):
            r=saved["results"]["geometry_null_K"+str(k)]
            self.assertLess(r["permutation_guard_applied_trials"],
                            r["old_LOO_gate_active_trials"])
            self.assertLessEqual(r["permutation_guard_applied_trials"],8)

    def test_known_signal_can_still_reduce_fake_rmse(self):
        saved=json.loads(m.OUTPUT.read_text(encoding="utf-8"))
        for k in (5,6):
            r=saved["results"]["geometry_signal_K"+str(k)]
            self.assertLess(r["fabricated_RMSE"]["support_permutation_geometry"],
                            r["fabricated_RMSE"]["perceiver_krr"])
            self.assertLessEqual(r["permutation_guard_applied_trials"],
                                 r["old_LOO_gate_active_trials"])

    def test_invalid_trial_budget_or_seed(self):
        with self.assertRaises(ValueError):
            m.synthetic_permutation_stress(n_trials=15)
        with self.assertRaises(ValueError):
            m.synthetic_permutation_stress(n_trials=16,seed=-3)

if __name__=="__main__":
    unittest.main(verbosity=2)
