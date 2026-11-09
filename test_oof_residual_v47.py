"""Synthetic exclusion and selection checks for v4.7 OOF chemical prior."""
import unittest
import numpy as np
from run_oof_residual_v47_20261109 import crossfit_prior,choose_weight,CROSS_FIT_FOLDS
from run_profile_study import metric

class TestScaffoldOOF(unittest.TestCase):
    def setUp(self):
        self.pi=list(range(8))
        self.ci=list(range(12))
        self.features=np.arange(32,dtype=float).reshape(8,4)
        self.bits=np.zeros((12,16),dtype=float)
        for j in range(12):self.bits[j,j]=1.
        self.y=np.arange(96,dtype=float).reshape(8,12)
        self.groups=[[j*3+i for i in range(3)] for j in range(4)]

    def builder(self,pi,ci,feats,bits,y,groups):
        """Conservative fake prior uses only fit rectangle's y."""
        value=y[np.ix_(pi,ci)].mean()
        return np.full(len(bits),value),0.1

    def test_each_scaffold_group_excluded_no_label_peeking(self):
        pred,audit=crossfit_prior(self.pi,self.ci,self.features,self.bits,
                                 self.y,self.groups,"synthetic_1",
                                 prior_builder=self.builder)
        self.assertEqual(len(audit),CROSS_FIT_FOLDS)
        self.assertTrue(all(x["heldout_compounds"]==3 for x in audit))
        for group in self.groups:
            changed=self.y.copy()
            changed[:,group]+=1e5
            after,_=crossfit_prior(self.pi,self.ci,self.features,self.bits,
                                 changed,self.groups,"synthetic_1",
                                 prior_builder=self.builder)
            # Each group is evaluated only by a fit excluding its own labels.
            self.assertTrue(np.array_equal(pred[group],after[group]))
        self.assertTrue(np.isfinite(pred).all())

    def test_weight_selection_zero_exact_fallback(self):
        y=np.ones((4,9))
        prior=y.copy()
        bad=y+2.
        cases=[{"prior":prior,"ensemble":bad,"truth":y} for _ in range(4)]
        chosen,grid=choose_weight(cases)
        self.assertEqual(chosen["weight"],0.)
        self.assertEqual(len(grid),5)
        self.assertEqual(chosen["inner_macro_rmse"],0.)

    def test_nontrivial_cold_compound_residual(self):
        pred,_=crossfit_prior(self.pi,self.ci,self.features,self.bits,
                             self.y,self.groups,"synthetic_2",prior_builder=self.builder)
        insample_mean=self.y.mean()
        self.assertFalse(np.allclose(pred,insample_mean))

if __name__=="__main__":
    unittest.main(verbosity=2)
