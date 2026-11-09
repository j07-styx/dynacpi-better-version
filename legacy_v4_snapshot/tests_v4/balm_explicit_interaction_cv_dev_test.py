"""Small independent shape and no-interaction-control checks for BALM pair model."""
import numpy as np
from scipy.sparse import csr_matrix
from scripts.train_balm_explicit_cross_interaction_svd_ridge_dev import (
    RANK,projected_pair,fit_predict
)

def test_pair_feature_shapes_and_finiteness():
    rng=np.random.default_rng(7)
    chem_train=csr_matrix(rng.random((32,40)).astype(np.float32))
    chem_valid=csr_matrix(rng.random((5,40)).astype(np.float32))
    prot_train=csr_matrix(rng.random((32,50)).astype(np.float32))
    prot_valid=csr_matrix(rng.random((5,50)).astype(np.float32))
    tr,va=projected_pair(chem_train,chem_valid,prot_train,prot_valid)
    assert tr.shape==(32,RANK**2)
    assert va.shape==(5,RANK**2)
    assert np.isfinite(tr.data).all() and np.isfinite(va.data).all()

def test_zero_cross_matches_additive_ridge():
    rng=np.random.default_rng(12)
    tr=csr_matrix(rng.normal(size=(70,24)).astype(np.float32))
    va=csr_matrix(rng.normal(size=(8,24)).astype(np.float32))
    ptr=csr_matrix(rng.normal(size=(70,16)).astype(np.float32))
    pva=csr_matrix(rng.normal(size=(8,16)).astype(np.float32))
    y=rng.normal(size=70)
    a=fit_predict(tr,va,ptr,pva,y,alpha=1.0,cross_scale=0.0)
    b=fit_predict(tr,va,csr_matrix((70,16)),csr_matrix((8,16)),y,alpha=1.0,cross_scale=0.0)
    assert np.allclose(a,b,atol=1e-9)
