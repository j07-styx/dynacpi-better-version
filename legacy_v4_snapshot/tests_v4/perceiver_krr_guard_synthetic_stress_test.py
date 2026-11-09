"""Safety, reproducibility and false-gain accounting for fabricated guard stress."""
from __future__ import annotations
from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=spec_from_file_location("krr_guard_fabricated_stress",
                             ROOT/"scripts/stress_perceiver_krr_guard_synthetic.py")
m=module_from_spec(spec)
spec.loader.exec_module(m)


class KRRGuardSyntheticStressTests(unittest.TestCase):
    def test_fixed_seed_reproducible(self):
        x=m.synthetic_stress(n_trials=16,seed=2601)
        self.assertEqual(x,m.synthetic_stress(n_trials=16,seed=2601))
        self.assertEqual(set(x["results"]),{
            "geometry_signal_K5","geometry_signal_K6",
            "geometry_null_K5","geometry_null_K6",
        })

    def test_all_scenarios_explicitly_fabricated(self):
        x=m.synthetic_stress(n_trials=16)
        self.assertEqual(x["status"],
            "FABRICATED_KRR_SUPPORT_GATE_STRESS_NOT_SCIENTIFIC_EVIDENCE")
        self.assertFalse(x["approval_or_training_performed"])
        for key,row in x["results"].items():
            self.assertEqual(row["fabricated_trials"],16)
            self.assertEqual(row["fabricated_queries_per_trial"],8-row["support_size"])
            self.assertTrue(0<=row["geometry_gate_active_trials"]<=16)
            self.assertEqual(set(row["fabricated_RMSE"]),{
                "perceiver_krr","krr_plus_geometry","krr_plus_shuffled_geometry",
            })

    def test_invalid_trials_or_seed_rejected(self):
        with self.assertRaises(ValueError):
            m.synthetic_stress(n_trials=8)
        with self.assertRaises(ValueError):
            m.synthetic_stress(n_trials=16,seed=-1)

    def test_archived_public_aggregate_never_approves_real_model(self):
        if not m.OUTPUT.exists():
            self.skipTest("Aggregate synthetic stress not present")
        public=json.loads(m.OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(public["status"],
             "FABRICATED_KRR_SUPPORT_GATE_STRESS_NOT_SCIENTIFIC_EVIDENCE")
        self.assertEqual(public["seed"],2601)
        self.assertEqual(public["trial_count_per_case"],128)
        self.assertFalse(public["approval_or_training_performed"])
        self.assertTrue(all(v["fabricated_trials"]==128
                            for v in public["results"].values()))
        self.assertNotIn("source_family_sha256",json.dumps(public))
        self.assertNotIn("BindingDB Reactant_set_id",json.dumps(public))

if __name__=="__main__":
    unittest.main(verbosity=2)
