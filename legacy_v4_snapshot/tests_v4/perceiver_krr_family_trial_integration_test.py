"""Fictional full 10-arm K-shot trial integration and adversarial safety tests.

Test-only fabricated support/query affinities; every fake source family,
protein construct, and frozen prediction is synthetic. No real BindingDB
outcomes, model checkpoints or reserved test records may enter.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import runpy
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from dynacpi.episode_data import EvaluationLabels
from dynacpi.perceiver_family_trial_evaluator import (
    evaluate_source_family_fewshot,ARMS,KRR_CHALLENGERS
)
from dynacpi.perceiver_geometry_negative_control import (
    compare_geometry_negative_control,
)
from dynacpi.perceiver_krr_family_prediction_bundle import (
    predict_krr_family_challengers,
)
from dynacpi.perceiver_krr_family_eval_adapter import (
    evaluation_rows_with_krr_challengers,
)

fixture=runpy.run_path(
    str(ROOT/"tests_v4/perceiver_krr_geometry_guard_test.py"),
    run_name="synthetic_fixture_only",
)["fiction"]


def one_fake_trial(fi=0,seed=2601,draw=0,available=True,k=6):
    ep,p,g=fixture(k=k)
    ep=replace(ep,
        task_id=f"fake_krr_study_family_{fi}",
        protein_component_id=sha256(f"fake_protein_{fi}".encode()).hexdigest(),
        sequence_sha256=sha256(f"fake_construct_seq_{fi}".encode()).hexdigest())
    if not available:
        g=None
    # PREDICTION STAGE: no query labels exist in any input.
    standard=compare_geometry_negative_control(
        ep,p,g,available,shuffle_seed=2601)
    bundle=predict_krr_family_challengers(
        ep,p,g,available,shuffle_seed=2601)
    # EVALUATION STAGE: a separate object is constructed AFTER inference.
    labels=EvaluationLabels(
        task_id=ep.task_id,
        query_ids=tuple(q.compound_id for q in ep.queries),
        pKd_values=tuple(
            p[q.compound_id]+.4+(
                .9*(g[q.compound_id]-p[q.compound_id]) if available else 0.0)
            for q in ep.queries
        ),
    )
    source=sha256(f"fabricated_source_family_{fi}".encode()).hexdigest()
    rows=evaluation_rows_with_krr_challengers(
        ep,labels,standard,bundle,
        geometry_available=available,source_family_sha256=source,
        construct_id=f"fake_construct_{fi}",
        seed=seed,draw=draw)
    return ep,p,g,standard,bundle,labels,rows


def fake_matrix():
    all_rows=[]
    for fi in range(6):
        for seed in (2601,2602):
            for draw in (0,1):
                all_rows.extend(one_fake_trial(
                    fi=fi,seed=seed,draw=draw,
                    available=fi%3!=1
                )[-1])
    return all_rows


class KRRFamilyEndToEndTests(unittest.TestCase):
    def test_full_family_evaluator_with_all_ten_arms(self):
        matrix=fake_matrix()
        output=evaluate_source_family_fewshot(
            matrix,reference_arm="p5_kernel_ridge",
            include_krr_challengers=True,expected_queries=2,
            expected_support=6,bootstrap_draws=200)
        self.assertEqual(output["n_source_families"],6)
        self.assertEqual(output["n_protein_components"],6)
        self.assertEqual(output["n_tasks"],6)
        self.assertEqual(output["n_episodes"],24)
        self.assertEqual(output["n_prediction_rows"],48)
        self.assertTrue(output["include_krr_challengers"])
        self.assertEqual(set(output["pooled_query_RMSE"]),set(ARMS+KRR_CHALLENGERS))
        self.assertEqual(set(output["paired_comparisons"]) & {
            "krr_perm_vs_p5_kernel_ridge",
            "krr_perm_vs_krr_loo_geometry",
            "krr_perm_vs_krr_shuffled_permutation_geometry",
        },{
            "krr_perm_vs_p5_kernel_ridge",
            "krr_perm_vs_krr_loo_geometry",
            "krr_perm_vs_krr_shuffled_permutation_geometry",
        })

    def test_label_identity_never_enters_inference(self):
        ep,p,g,standard,bundle,labels,rows=one_fake_trial()
        self.assertTrue(all(not hasattr(q,"pKd") for q in ep.queries))
        self.assertEqual(tuple(r["y"] for r in rows),labels.pKd_values)
        self.assertTrue(all(r["support_count"]==len(ep.support) for r in rows))
        self.assertTrue(all(abs(r["krr_permutation_geometry"]-
                                r["p5_kernel_ridge"])<=.50000001 for r in rows))

    def test_legacy_evaluator_remains_unchanged(self):
        matrix=fake_matrix()
        original=evaluate_source_family_fewshot(
            matrix,reference_arm="p5_kernel_ridge",
            expected_queries=2,expected_support=6,
            bootstrap_draws=200)
        self.assertFalse(original["include_krr_challengers"])
        self.assertEqual(set(original["pooled_query_RMSE"]),set(ARMS))
        self.assertNotIn("krr_perm_vs_p5_kernel_ridge",original["paired_comparisons"])

    def test_missing_krr_arms_fail_closed_only_when_explicit(self):
        matrix=fake_matrix()
        del matrix[0]["krr_permutation_geometry"]
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            evaluate_source_family_fewshot(
                matrix,reference_arm="p5_kernel_ridge",
                include_krr_challengers=True,expected_queries=2,
                expected_support=6,bootstrap_draws=200)

    def test_no_geometry_krr_exact_fallback(self):
        ep,p,g,standard,bundle,labels,rows=one_fake_trial(available=False)
        for row in rows:
            self.assertEqual(row["krr_loo_geometry"],row["p5_kernel_ridge"])
            self.assertEqual(row["krr_permutation_geometry"],row["p5_kernel_ridge"])
            self.assertEqual(
                row["krr_shuffled_permutation_geometry"],row["p5_kernel_ridge"])

    def test_query_label_change_does_not_recompute_predictions(self):
        ep,p,g,standard,bundle,labels,rows=one_fake_trial()
        changed=replace(labels,pKd_values=tuple(x+8.0 for x in labels.pKd_values))
        altered=evaluation_rows_with_krr_challengers(
            ep,changed,standard,bundle,geometry_available=True,
            source_family_sha256=rows[0]["source_family_sha256"],
            construct_id="fake_construct_0",seed=2601,draw=0)
        for before,after in zip(rows,altered):
            self.assertNotEqual(before["y"],after["y"])
            for key in (*ARMS,*KRR_CHALLENGERS):
                self.assertEqual(before[key],after[key])

    def test_tampered_krr_baseline_rejected(self):
        ep,p,g,std,bundle,labels,rows=one_fake_trial()
        r=bundle.permutation[0]
        broken=replace(r,perceiver_krr_pKd=r.perceiver_krr_pKd+.01)
        bad=replace(bundle,permutation=(broken,*bundle.permutation[1:]))
        with self.assertRaisesRegex(ValueError,"frozen equal-support"):
            evaluation_rows_with_krr_challengers(
                ep,labels,std,bad,geometry_available=True,
                source_family_sha256=rows[0]["source_family_sha256"],
                construct_id="fake_construct_0",seed=2601,draw=0)

    def test_tampered_loo_prediction_rejected(self):
        ep,p,g,std,bundle,labels,rows=one_fake_trial()
        r=bundle.loo[0]
        bad=replace(bundle,loo=(
            replace(r,geometry_krr_pKd=r.geometry_krr_pKd+.1),*bundle.loo[1:]))
        with self.assertRaisesRegex(ValueError,"LOO challenger differs"):
            evaluation_rows_with_krr_challengers(
                ep,labels,std,bad,geometry_available=True,
                source_family_sha256=rows[0]["source_family_sha256"],
                construct_id="fake_construct_0",seed=2601,draw=0)

    def test_tampered_guard_decision_rejected(self):
        ep,p,g,std,bundle,labels,rows=one_fake_trial()
        r=bundle.permutation[0]
        bad=replace(bundle,permutation=(
            replace(r,permutation_guarded_pKd=r.perceiver_krr_pKd+1.5),
            *bundle.permutation[1:]))
        with self.assertRaisesRegex(ValueError,"Permutation decision"):
            evaluation_rows_with_krr_challengers(
                ep,labels,std,bad,geometry_available=True,
                source_family_sha256=rows[0]["source_family_sha256"],
                construct_id="fake_construct_0",seed=2601,draw=0)

    def test_wrong_query_alignment_rejected(self):
        ep,p,g,std,bundle,labels,rows=one_fake_trial()
        swapped=replace(bundle,permutation=bundle.permutation[::-1])
        with self.assertRaisesRegex(ValueError,"identical ordered query"):
            evaluation_rows_with_krr_challengers(
                ep,labels,std,swapped,geometry_available=True,
                source_family_sha256=rows[0]["source_family_sha256"],
                construct_id="fake_construct_0",seed=2601,draw=0)

    def test_wrong_geometry_availability_rejected(self):
        ep,p,g,std,bundle,labels,rows=one_fake_trial()
        with self.assertRaisesRegex(ValueError,"availability differs"):
            evaluation_rows_with_krr_challengers(
                ep,labels,std,bundle,geometry_available=False,
                source_family_sha256=rows[0]["source_family_sha256"],
                construct_id="fake_construct_0",seed=2601,draw=0)

    def test_wrong_family_or_study_identity_rejected(self):
        ep,p,g,std,bundle,labels,rows=one_fake_trial()
        changed=replace(ep,task_id="different_fake_study")
        with self.assertRaisesRegex(ValueError,"match the task"):
            evaluation_rows_with_krr_challengers(
                changed,labels,std,bundle,geometry_available=True,
                source_family_sha256=rows[0]["source_family_sha256"],
                construct_id="fake_construct_0",seed=2601,draw=0)

    def test_family_evaluator_rejects_oversize_krr_delta(self):
        matrix=fake_matrix()
        matrix[0]["krr_permutation_geometry"]=matrix[0]["p5_kernel_ridge"]+1.2
        with self.assertRaisesRegex(ValueError,"delta cap"):
            evaluate_source_family_fewshot(
                matrix,reference_arm="p5_kernel_ridge",
                include_krr_challengers=True,expected_queries=2,
                expected_support=6,bootstrap_draws=200)

    def test_unavailable_geometry_cannot_change_krr(self):
        matrix=fake_matrix()
        target=next(r for r in matrix if not r["geometry_available"])
        target["krr_permutation_geometry"]+=.25
        with self.assertRaisesRegex(ValueError,"P5-KRR"):
            evaluate_source_family_fewshot(
                matrix,reference_arm="p5_kernel_ridge",
                include_krr_challengers=True,expected_queries=2,
                expected_support=6,bootstrap_draws=200)

    def test_preselected_reference_still_required(self):
        with self.assertRaisesRegex(ValueError,"PRESELECTED"):
            evaluate_source_family_fewshot(
                fake_matrix(),reference_arm="krr_permutation_geometry",
                include_krr_challengers=True,expected_queries=2,
                expected_support=6,bootstrap_draws=200)
        with self.assertRaisesRegex(ValueError,"PRESELECTED p5_kernel_ridge"):
            evaluate_source_family_fewshot(
                fake_matrix(),reference_arm="p5_global",
                include_krr_challengers=True,expected_queries=2,
                expected_support=6,bootstrap_draws=200)

    def test_replay_identical_family_statistics(self):
        x=evaluate_source_family_fewshot(
            fake_matrix(),reference_arm="p5_kernel_ridge",
            include_krr_challengers=True,expected_queries=2,
            expected_support=6,bootstrap_draws=200)
        y=evaluate_source_family_fewshot(
            list(reversed(fake_matrix())),reference_arm="p5_kernel_ridge",
            include_krr_challengers=True,expected_queries=2,
            expected_support=6,bootstrap_draws=200)
        for key in KRR_CHALLENGERS:
            self.assertAlmostEqual(
                x["source_family_macro_episode_RMSE"][key],
                y["source_family_macro_episode_RMSE"][key],places=12)

if __name__=="__main__":
    unittest.main(verbosity=2)
