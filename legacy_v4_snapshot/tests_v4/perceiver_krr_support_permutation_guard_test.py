"""Fabricated-only support-permutation reliability guard tests.

No experimental assay data, actual model weights or held-out test labels.
Checks algebraic behavior, exact Perceiver-KRR fallback, deterministic null,
and that query-label objects cannot be supplied to the inference function.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.perceiver_krr_support_permutation_guard import (
    PermutationGuardSettings,
    compare_krr_with_support_permutation_guard,
    _support_null_permutations,
)
from dynacpi.perceiver_krr_geometry_guard import (
    compare_krr_geometry_with_support_validation,
)
from dynacpi.perceiver_geometry_negative_control import shuffle_compound_disagreement

fiction=runpy.run_path(
    str(ROOT/"tests_v4/perceiver_krr_geometry_guard_test.py"),
    run_name="fabricated_fixture_only",
)["fiction"]


def guard(e,p,g,avail=True,**kwargs):
    return compare_krr_with_support_permutation_guard(
        e,p,g if avail else None,avail,**kwargs)


class KRRSupportPermutationGuardTests(unittest.TestCase):
    def test_null_permutations_are_unique_nonidentity_deterministic(self):
        a=_support_null_permutations(5,2611,49)
        b=_support_null_permutations(5,2611,49)
        self.assertEqual(a,b)
        self.assertEqual(len(set(a)),49)
        self.assertTrue(all(p!=tuple(range(5)) for p in a))
        self.assertTrue(all(sorted(p)==list(range(5)) for p in a))

    def test_signal_can_pass_both_original_and_null_gates(self):
        e,p,g=fiction(mode="correlated")
        out=guard(e,p,g)
        self.assertEqual(len(out),len(e.queries))
        self.assertTrue(all(x.permutation_gate_passed for x in out))
        self.assertTrue(all(x.original_loo_gate>0 for x in out))
        self.assertEqual(out[0].null_pvalue,0.02)
        self.assertTrue(any(abs(x.applied_geometry_delta)>1e-8 for x in out))

    def test_calibration_only_and_constant_geometry_fallback(self):
        for mode in ("intercept","constant_geometry"):
            e,p,g=fiction(mode=mode)
            out=guard(e,p,g)
            for row in out:
                self.assertFalse(row.permutation_gate_passed)
                self.assertEqual(row.permutation_guarded_pKd,row.perceiver_krr_pKd)
                self.assertEqual(row.applied_geometry_delta,0.0)
                self.assertIsNone(row.null_pvalue)

    def test_k0_to_k4_never_change_strong_krr(self):
        for k in range(5):
            e,p,g=fiction(k=k)
            out=guard(e,p,g)
            self.assertTrue(all(x.permutation_guarded_pKd==x.perceiver_krr_pKd
                                and x.n_null_permutations_used==0 for x in out))

    def test_unavailable_geometry_uses_exact_perceiver_krr(self):
        e,p,g=fiction()
        out=guard(e,p,g,avail=False)
        self.assertTrue(all(x.permutation_guarded_pKd==x.perceiver_krr_pKd
                            for x in out))

    def test_perceiver_krr_is_identical_to_original_control(self):
        e,p,g=fiction()
        base=compare_krr_geometry_with_support_validation(e,p,g,True)
        out=guard(e,p,g)
        self.assertEqual([x.perceiver_krr_pKd for x in out],
                         [x.perceiver_krr_pKd for x in base])
        self.assertEqual([x.unguarded_geometry_pKd for x in out],
                         [x.geometry_krr_pKd for x in base])
        self.assertTrue(all(abs(x.applied_geometry_delta)<=.5 for x in out))

    def test_support_order_invariance(self):
        e,p,g=fiction()
        orig=guard(e,p,g)
        switched=guard(replace(e,support=e.support[::-1]),p,g)
        self.assertEqual([x.null_pvalue for x in orig],
                         [x.null_pvalue for x in switched])
        for a,b in zip(orig,switched):
            self.assertAlmostEqual(a.permutation_guarded_pKd,
                                   b.permutation_guarded_pKd,places=9)

    def test_query_reordering_preserves_compound_predictions(self):
        e,p,g=fiction()
        orig={x.compound_id:x for x in guard(e,p,g)}
        other=guard(replace(e,queries=e.queries[::-1]),p,g)
        self.assertEqual(set(orig),{x.compound_id for x in other})
        for x in other:
            self.assertAlmostEqual(x.permutation_guarded_pKd,
                                   orig[x.compound_id].permutation_guarded_pKd,places=9)
            self.assertEqual(x.null_pvalue,orig[x.compound_id].null_pvalue)

    def test_fixed_seed_reproducible(self):
        e,p,g=fiction()
        self.assertEqual(guard(e,p,g),guard(e,p,g))

    def test_shuffle_negative_control_unchanged_perceiver_baseline(self):
        e,p,g=fiction()
        sh=shuffle_compound_disagreement(e,p,g)
        original=guard(e,p,g)
        control=guard(e,p,sh)
        self.assertEqual([r.perceiver_krr_pKd for r in original],
                         [r.perceiver_krr_pKd for r in control])

    def test_parameter_validation_fails_closed(self):
        e,p,g=fiction()
        for settings in (
            PermutationGuardSettings(n_null_permutations=10),
            PermutationGuardSettings(n_null_permutations=200),
            PermutationGuardSettings(shuffle_seed=-1),
            PermutationGuardSettings(shuffle_seed=True),
            PermutationGuardSettings(max_pvalue=0),
            PermutationGuardSettings(max_pvalue=0.01),
        ):
            with self.assertRaises(ValueError):
                guard(e,p,g,permutation_settings=settings)

    def test_missing_and_nonfinite_geometry_scores_fail_closed(self):
        e,p,g=fiction()
        broken=dict(g)
        broken.pop(e.support[0].compound_id)
        with self.assertRaises(ValueError):
            guard(e,p,broken)
        broken=dict(g)
        broken[e.support[0].compound_id]=float("nan")
        with self.assertRaises(ValueError):
            guard(e,p,broken)
        with self.assertRaises(ValueError):
            compare_krr_with_support_permutation_guard(e,p,g,False)

    def test_query_labels_absent_from_prediction_boundary(self):
        e,p,g=fiction()
        self.assertTrue(all(not hasattr(q,"pKd") for q in e.queries))
        out=guard(e,p,g)
        self.assertEqual([x.compound_id for x in out],
                         [x.compound_id for x in e.queries])
        self.assertTrue(all(x.n_null_permutations_used==49 for x in out))

    def test_observed_pvalue_is_add_one_corrected(self):
        e,p,g=fiction()
        out=guard(e,p,g)
        for x in out:
            self.assertAlmostEqual(x.null_pvalue,
                (x.null_count_at_least_as_good+1)/(x.n_null_permutations_used+1))

if __name__=="__main__":
    unittest.main(verbosity=2)
