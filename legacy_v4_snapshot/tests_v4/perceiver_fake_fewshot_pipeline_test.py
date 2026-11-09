"""Full synthetic path: K-shot inference -> negative control -> evaluator rows.

Constructed from fake assay labels and fake independent study families.
No actual BindingDB values, checkpoints, held-out predictions, or model fits.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import math
import runpy
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.episode_data import EvaluationLabels
from dynacpi.perceiver_geometry_negative_control import compare_geometry_negative_control
from dynacpi.perceiver_trial_record_adapter import evaluation_rows_from_predictions
from dynacpi.perceiver_family_trial_evaluator import evaluate_source_family_fewshot

fixture=runpy.run_path(str(ROOT/"tests_v4/perceiver_geometry_negative_control_test.py"),
                       run_name="load_fictional_fixture_only")["fiction"]


def fake_pipeline(available=True):
    ep,p,g=fixture()
    if not available:
        g=None
    pred=compare_geometry_negative_control(ep,p,g,available,shuffle_seed=2601)
    labels=EvaluationLabels(
        task_id=ep.task_id,
        query_ids=tuple(q.compound_id for q in ep.queries),
        pKd_values=tuple(p[q.compound_id]+.35+.75*(
            g[q.compound_id]-p[q.compound_id] if available else 0.0
        ) for q in ep.queries),
    )
    return ep,p,g,pred,labels


def build_rows(fi,seed=2601,draw=0):
    ep,p,g,comparison,labels=fake_pipeline(available=(fi%2==0))
    ep=replace(ep,task_id=f"fake_independent_source_task_{fi}",
               protein_component_id=sha256(f"fake_protein_{fi}".encode()).hexdigest())
    labels=replace(labels,task_id=ep.task_id)
    # The comparison was computed first using only invented support labels;
    # its query predictions do not depend on query-label values.
    return evaluation_rows_from_predictions(
        ep,labels,comparison,
        geometry_available=(fi%2==0),
        source_family_sha256=sha256(f"independent_fake_source_{fi}".encode()).hexdigest(),
        construct_id=f"fake_construct_{fi}",seed=seed,draw=draw,
    )


class SyntheticEndToEndTrialTests(unittest.TestCase):
    def test_six_sources_complete_model_to_evaluator_pipeline(self):
        records=[]
        for fi in range(6):
            for seed in (2601,2602):
                for draw in (0,1):
                    records.extend(build_rows(fi,seed,draw))
        out=evaluate_source_family_fewshot(
            records,reference_arm="p5_kernel_ridge",
            expected_queries=2,expected_support=5,
            bootstrap_draws=200,bootstrap_seed=1109)
        self.assertEqual(out["n_source_families"],6)
        self.assertEqual(out["n_protein_components"],6)
        self.assertEqual(out["n_tasks"],6)
        self.assertEqual(out["n_episodes"],24)
        self.assertEqual(out["n_prediction_rows"],48)
        self.assertIn("d5_vs_d5_shuffled",out["paired_comparisons"])
        self.assertTrue(out["status"].startswith("UNVERIFIED"))

    def test_label_object_never_enters_prediction_function(self):
        ep,p,g,comparison,labels=fake_pipeline()
        self.assertFalse(hasattr(ep.queries[0],"pKd"))
        self.assertEqual(tuple(q.compound_id for q in ep.queries),
                         labels.query_ids)
        rows=evaluation_rows_from_predictions(
            ep,labels,comparison,
            geometry_available=True,
            source_family_sha256=sha256(b"fiction").hexdigest(),
            construct_id="fictional_construct",seed=0,draw=0)
        self.assertEqual([r["y"] for r in rows],list(labels.pKd_values))

    def test_support_digest_invariant_under_support_reordering(self):
        ep,p,g,comp,labels=fake_pipeline()
        fa=sha256(b"fake_family").hexdigest()
        first=evaluation_rows_from_predictions(
            ep,labels,comp,geometry_available=True,
            source_family_sha256=fa,construct_id="fake_construct",seed=0,draw=0)
        changed=replace(ep,support=ep.support[::-1])
        second=evaluation_rows_from_predictions(
            changed,labels,comp,geometry_available=True,
            source_family_sha256=fa,construct_id="fake_construct",seed=0,draw=0)
        self.assertEqual(first[0]["support_panel_sha256"],
                         second[0]["support_panel_sha256"])

    def test_reordered_query_labels_fail_before_join(self):
        ep,p,g,comp,labels=fake_pipeline()
        swapped=replace(labels,query_ids=labels.query_ids[::-1])
        with self.assertRaisesRegex(ValueError,"reordered"):
            evaluation_rows_from_predictions(
                ep,swapped,comp,geometry_available=True,
                source_family_sha256=sha256(b"fiction").hexdigest(),
                construct_id="fake_construct",seed=0,draw=0)

    def test_mismatched_task_label_rejected(self):
        ep,p,g,comp,labels=fake_pipeline()
        altered=replace(labels,task_id="different-task")
        with self.assertRaisesRegex(ValueError,"match the task"):
            evaluation_rows_from_predictions(
                ep,altered,comp,geometry_available=True,
                source_family_sha256=sha256(b"fiction").hexdigest(),
                construct_id="fake_construct",seed=0,draw=0)

    def test_nonfinite_query_labels_rejected(self):
        ep,p,g,comp,labels=fake_pipeline()
        altered=replace(labels,pKd_values=(float("nan"),labels.pKd_values[1]))
        with self.assertRaisesRegex(ValueError,"non-finite"):
            evaluation_rows_from_predictions(
                ep,altered,comp,geometry_available=True,
                source_family_sha256=sha256(b"fiction").hexdigest(),
                construct_id="fake_construct",seed=0,draw=0)

    def test_corrupted_baseline_arm_rejected(self):
        ep,p,g,comp,labels=fake_pipeline()
        broken=replace(comp.shuffled[0],p5_global=comp.shuffled[0].p5_global+0.1)
        tampered=replace(comp,shuffled=(broken,*comp.shuffled[1:]))
        with self.assertRaisesRegex(ValueError,"identical frozen"):
            evaluation_rows_from_predictions(
                ep,labels,tampered,geometry_available=True,
                source_family_sha256=sha256(b"fiction").hexdigest(),
                construct_id="fake_construct",seed=0,draw=0)

    def test_invalid_task_level_geometry_and_family_rejected(self):
        ep,p,g,comp,labels=fake_pipeline()
        with self.assertRaisesRegex(TypeError,"task-level bool"):
            evaluation_rows_from_predictions(
                ep,labels,comp,geometry_available=1,
                source_family_sha256=sha256(b"fiction").hexdigest(),
                construct_id="fake_construct",seed=0,draw=0)
        with self.assertRaisesRegex(ValueError,"source-family"):
            evaluation_rows_from_predictions(
                ep,labels,comp,geometry_available=True,
                source_family_sha256="made_up",construct_id="fake_construct",
                seed=0,draw=0)

if __name__=="__main__":
    unittest.main(verbosity=2)
