"""Fictional source-family-clustered fairness and statistics tests.

All labels and predictions are fabricated. This is an evaluator-only smoke
suite and never reads BindingDB, the retired split, or prospective test data.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import math
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.perceiver_family_trial_evaluator import (
    evaluate_source_family_fewshot, ARMS,
)


def fake_rows():
    result=[]
    for family_idx in range(8):
        family=sha256(f"fake-source-family-{family_idx}".encode()).hexdigest()
        components=2 if family_idx==0 else 1
        for component_idx in range(components):
            component=sha256(f"fake-component-{family_idx}-{component_idx}".encode()).hexdigest()
            construct=f"fake-construct-{family_idx}-{component_idx}"
            task=f"fake-task-{family_idx}-{component_idx}"
            for seed in (2601,2602):
                for draw in (0,1):
                    panel=sha256(f"{task}|{seed}|{draw}|fictional_support".encode()).hexdigest()
                    for q in range(3):
                        y=6.0+0.15*family_idx+0.08*component_idx+0.1*q
                        baseline=y+0.3
                        global_=y+0.2
                        local=y+0.12
                        kr=y+0.11
                        available=(family_idx % 2 == 0)
                        true_geometry=y+0.05 if available else global_
                        shuffle=y+0.25 if available else global_
                        result.append({
                            "task_id":task,
                            "construct_id":construct,
                            "protein_component_id":component,
                            "source_family_sha256":family,
                            "assay_key":f"fictional-assay-{task}",
                            "seed":seed,"draw":draw,
                            "query_id":f"fake-query-{q}",
                            "support_panel_sha256":panel,
                            "support_count":5,"geometry_available":available,
                            "y":y,
                            "p0":baseline,
                            "p5_global":global_,
                            "p5_local":local,
                            "p5_kernel_ridge":kr,
                            "d5_no_geometry":global_,
                            "d5_shuffled":shuffle,
                            "d5_geometry":true_geometry,
                        })
    return result


def evaluate(rows):
    return evaluate_source_family_fewshot(rows,
        reference_arm="p5_kernel_ridge",expected_queries=3,
        expected_support=5,bootstrap_draws=250,bootstrap_seed=1109)


class FamilyTrialEvaluationTests(unittest.TestCase):
    def test_multiarm_source_family_macro_and_pooled(self):
        result=evaluate(fake_rows())
        self.assertEqual(result["n_source_families"],8)
        self.assertEqual(result["n_protein_components"],9)
        self.assertEqual(result["n_tasks"],9)
        self.assertEqual(result["n_episodes"],36)
        self.assertEqual(result["n_prediction_rows"],108)
        self.assertEqual(set(result["pooled_query_RMSE"]),set(ARMS))
        self.assertEqual(result["reference_arm_preselected"],"p5_kernel_ridge")
        self.assertAlmostEqual(result["pooled_query_RMSE"]["p0"],.3,places=10)
        # Do not pretend the new model wins against a stronger, fair control.
        self.assertGreater(result["source_family_macro_episode_RMSE"]["d5_geometry"],
                           result["source_family_macro_episode_RMSE"]["p5_kernel_ridge"])
        self.assertLess(result["source_family_macro_episode_RMSE"]["d5_geometry"],
                        result["source_family_macro_episode_RMSE"]["p5_global"])

    def test_pair_bootstrap_resamples_independent_families(self):
        result=evaluate(fake_rows())
        contrasts=result["paired_comparisons"]
        self.assertEqual(set(contrasts),{
            "d5_vs_p5_kernel_ridge","d5_vs_d5_no_geometry","d5_vs_d5_shuffled"})
        self.assertLess(contrasts["d5_vs_p5_kernel_ridge"]["family_macro_RMSE_reduction"],0)
        self.assertGreater(contrasts["d5_vs_d5_shuffled"]["paired_family_bootstrap_CI95"][0],0)
        self.assertGreater(contrasts["d5_vs_d5_no_geometry"]["family_macro_RMSE_reduction"],0)
        self.assertEqual(result,evaluate(fake_rows())) # seed determinism

    def test_nonuniform_component_count_does_not_double_count_family(self):
        result=evaluate(fake_rows())
        self.assertEqual(result["n_source_families"],8)
        self.assertEqual(result["n_protein_components"],9)
        # Exactly four of eight SOURCE FAMILIES have geometry available.
        # A second component within one of them may not grant 2 bootstrap votes.
        self.assertAlmostEqual(result["source_family_macro_episode_RMSE"]["d5_geometry"],
                               .125,places=10)

    def test_unpreselected_or_p0_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,"PRESELECTED"):
            evaluate_source_family_fewshot(fake_rows(),reference_arm="best_on_test")
        with self.assertRaisesRegex(ValueError,"PRESELECTED"):
            evaluate_source_family_fewshot(fake_rows(),reference_arm="p0")

    def test_duplicate_rows_rejected(self):
        rows=fake_rows()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            evaluate(rows+[rows[0]])

    def test_query_label_shift_between_draws_rejected(self):
        rows=fake_rows()
        rows[3]["y"]+=.001
        with self.assertRaisesRegex(ValueError,"label changed"):
            evaluate(rows)

    def test_changed_query_set_rejected(self):
        rows=fake_rows()
        rows[3]["query_id"]="other-query-not-in-task-set"
        with self.assertRaisesRegex(ValueError,"mismatched query"):
            evaluate(rows)

    def test_missing_query_rejected(self):
        rows=fake_rows()
        rows.pop(0)
        with self.assertRaisesRegex(ValueError,"fixed distinct queries"):
            evaluate(rows)

    def test_missing_seed_draw_grid_rejected(self):
        rows=[r for r in fake_rows() if not (
            r["task_id"]=="fake-task-0-0" and r["seed"]==2601 and r["draw"]==0
        )]
        with self.assertRaisesRegex(ValueError,"same fixed seed"):
            evaluate(rows)

    def test_mixed_support_panels_within_same_episode_rejected(self):
        rows=fake_rows()
        rows[1]["support_panel_sha256"]=sha256(b"other-panel").hexdigest()
        with self.assertRaisesRegex(ValueError,"inconsistent support"):
            evaluate(rows)

    def test_changing_component_family_mapping_rejected(self):
        rows=fake_rows()
        rows[0]["source_family_sha256"]=sha256(b"forged-family").hexdigest()
        with self.assertRaisesRegex(ValueError,"Task changed"):
            evaluate(rows)

    def test_unavailable_geometry_cannot_change_predictions(self):
        rows=fake_rows()
        target=next(r for r in rows if not r["geometry_available"])
        target["d5_geometry"]+=.3
        with self.assertRaisesRegex(ValueError,"Unavailable geometry"):
            evaluate(rows)

    def test_no_geometry_baseline_exactness_rejected_when_broken(self):
        rows=fake_rows()
        rows[0]["d5_no_geometry"]+=.1
        with self.assertRaisesRegex(ValueError,"exactly match"):
            evaluate(rows)

    def test_invalid_family_digest_rejected(self):
        rows=fake_rows()
        rows[0]["source_family_sha256"]="not_a_hash"
        with self.assertRaisesRegex(ValueError,"64-hex"):
            evaluate(rows)

    def test_nonfinite_or_missing_predictions_fail_closed(self):
        rows=fake_rows()
        rows[0]["d5_geometry"]=float("nan")
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            evaluate(rows)
        rows=fake_rows()
        del rows[0]["p5_global"]
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            evaluate(rows)

    def test_fewer_than_five_families_does_not_fake_certainty(self):
        rows=fake_rows()
        kept=set(r["source_family_sha256"] for r in rows[:12*4])
        selected=[r for r in rows if r["source_family_sha256"] in kept]
        with self.assertRaisesRegex(ValueError,"Not enough independent"):
            evaluate(selected)

    def test_bad_support_budget_and_bootstrap_rejected(self):
        rows=fake_rows()
        rows[0]["support_count"]=10
        with self.assertRaisesRegex(ValueError,"Support count"):
            evaluate(rows)
        with self.assertRaises(ValueError):
            evaluate_source_family_fewshot(fake_rows(),reference_arm="p5_global",
                expected_queries=3,bootstrap_draws=20)

if __name__=="__main__":
    unittest.main(verbosity=2)
