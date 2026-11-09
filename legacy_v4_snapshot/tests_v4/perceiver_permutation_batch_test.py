"""Scalar-reference parity, exact small-support nulls, and solve-count regression."""
import sys,unittest,runpy
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.perceiver_krr_support_permutation_guard import (
    _permuted_support_mse,_permuted_support_mses,_support_null_permutations,
    PermutationGuardSettings,compare_krr_with_support_permutation_guard)
from dynacpi.perceiver_krr_geometry_guard import KRRGeometrySettings
fiction=runpy.run_path(str(ROOT/"tests_v4/perceiver_krr_geometry_guard_test.py"),run_name="fixture")["fiction"]

class BatchPermutationTest(unittest.TestCase):
    def test_random_scalar_reference_and_degenerate_columns(self):
        rng=np.random.default_rng(2612)
        for n in (3,4,5,10):
            x=rng.normal(size=(n,6));k=x@x.T
            for scale in (0.,1e-5,1.,100.):
                r=rng.normal(size=n)*3;d=rng.normal(size=(n,49))*scale
                d[:,0]=4.
                settings=KRRGeometrySettings()
                kw=dict(chemical_ridge=.5,cap=.5,geometry_settings=settings)
                expected=np.array([_permuted_support_mse(k,r,d[:,j],**kw) for j in range(d.shape[1])])
                got=_permuted_support_mses(k,r,d,**kw)
                np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-12)
    def test_one_solve_per_support_fold(self):
        n=5;k=np.eye(n);r=np.arange(n,dtype=float);d=np.tile(r[:,None],(1,49))
        with patch("numpy.linalg.solve",wraps=np.linalg.solve) as solve:
            _permuted_support_mses(k,r,d,chemical_ridge=.5,cap=.5,geometry_settings=KRRGeometrySettings())
        self.assertEqual(solve.call_count,n)
    def test_exhausts_finite_null_without_duplicates(self):
        for n,total in ((3,5),(4,23),(5,119)):
            perms=_support_null_permutations(n,2612,199)
            self.assertEqual(len(perms),total)
            self.assertEqual(len(set(perms)),total)
            self.assertNotIn(tuple(range(n)),perms)
    def test_valid_high_budget_k5_runs(self):
        e,p,g=fiction(k=5,mode="correlated")
        out=compare_krr_with_support_permutation_guard(e,p,g,True,
            permutation_settings=PermutationGuardSettings(n_null_permutations=199))
        self.assertTrue(all(x.n_null_permutations_used==119 for x in out))
        for x in out:self.assertAlmostEqual(x.null_pvalue,(x.null_count_at_least_as_good+1)/120)
    def test_low_support_override_never_raises(self):
        for n in (3,4):
            e,p,g=fiction(k=n,mode="correlated")
            out=compare_krr_with_support_permutation_guard(e,p,g,True,
                geometry_settings=KRRGeometrySettings(min_support=3))
            for x in out:
                if x.original_loo_gate>0:
                    self.assertEqual(x.n_null_permutations_used,5 if n==3 else 23)
                    if n==3:self.assertFalse(x.permutation_gate_passed)
    def test_invalid_matrix_shape(self):
        with self.assertRaises(ValueError):
            _permuted_support_mses(np.eye(3),np.zeros(3),np.zeros((3,0)),
                chemical_ridge=.5,cap=.5,geometry_settings=KRRGeometrySettings())
if __name__=="__main__":unittest.main(verbosity=2)
