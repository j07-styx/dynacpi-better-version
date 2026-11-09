"""Synthetic unit checks for nested model-combination math (no DAVIS labels)."""
import unittest
import numpy as np
from run_ensemble_stack_shrink_20261109 import predict_combo,gate,pick_inner,paired_bootstrap
from run_profile_study import metric


class EnsembleTests(unittest.TestCase):
    def test_alpha_zero_exactly_chemical(self):
        rng=np.random.default_rng(4)
        base=rng.normal(size=(8,18))
        pred=rng.normal(size=(8,18))
        for g in [np.ones_like(base),rng.random(base.shape)]:
            self.assertTrue(np.array_equal(base,predict_combo(base,pred,g,0.0)))
            self.assertTrue(np.allclose(pred,predict_combo(base,pred,np.ones_like(base),1.0),atol=1e-12,rtol=0))

    def test_no_negative_gates(self):
        pf=np.array([[1.,0.],[.9,.1],[0.,1.]])
        fp=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,1.,0.,0.]])
        for mode in ("global","chemical_tanimoto","chemical_times_protein_cosine"):
            g=gate([0,1],[0,1],[2],[2],pf,fp,mode)
            self.assertEqual(g.shape,(1,1))
            self.assertTrue(np.all(g>=0))
            self.assertTrue(np.all(g<=1))
        self.assertAlmostEqual(gate([0,1],[0,1],[2],[2],pf,fp,"global")[0,0],1.)

    def test_inner_selection_can_fall_back_to_chemical_only(self):
        prior=np.broadcast_to(np.array([2.,3.]),(2,2)).copy()
        y=prior.copy()
        ensemble=prior+5.0
        cases=[{"prior":prior,"ensemble":ensemble,"y":y,
                "gates":{mode:np.ones_like(prior)
                  for mode in ("global","chemical_tanimoto","chemical_times_protein_cosine")}}
                for _ in range(2)]
        best,grid=pick_inner(cases)
        self.assertEqual(best["weight"],0.0)
        self.assertEqual(best["inner_mean_target_macro_rmse"],0.)
        self.assertEqual(len(grid),15)

    def test_paired_bootstrap_zero_deltas(self):
        result=paired_bootstrap([0.]*24,[0]*8+[1]*8+[2]*8,n=100)
        self.assertEqual(result["paired_mean_delta"],0.)
        self.assertEqual(result["descriptive_protein_bootstrap_95pct"],[0.,0.])
        self.assertTrue(result["not_confirmatory"])


if __name__=="__main__":
    unittest.main(verbosity=2)
