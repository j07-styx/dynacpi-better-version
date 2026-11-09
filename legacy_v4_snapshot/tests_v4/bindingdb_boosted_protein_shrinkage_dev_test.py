import numpy as np
from scripts.train_bindingdb_boosted_protein_shrinkage_cv_selected_dev import metric, predict_ensemble

def test_ensemble_reduces_to_controls():
    chem=np.array([1., 3., 5.])
    joint=np.array([6., 4., 2.])
    mean=3.
    assert np.allclose(predict_ensemble(mean,chem,joint,0.,.5),[3.,3.,3.])
    assert np.allclose(predict_ensemble(mean,chem,joint,1.,0.),chem)
    assert np.allclose(predict_ensemble(mean,chem,joint,1.,1.),joint)
    assert np.allclose(predict_ensemble(mean,chem,joint,1.,.5),(chem+joint)/2)

def test_macro_metric_differs_from_pooled():
    truth=np.zeros(4)
    comps=np.array([0,1,1,1])
    one=metric(np.array([3.,0.,0.,0.]),truth,comps)
    assert abs(one["pooled_rmse"]-1.5)<1e-8
    assert abs(one["component_macro_rmse"]-1.5)<1e-8
    two=metric(np.array([0.,2.,2.,2.]),truth,comps)
    assert abs(two["pooled_rmse"]-np.sqrt(3))<1e-8
    assert abs(two["component_macro_rmse"]-1.0)<1e-8
