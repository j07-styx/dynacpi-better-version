"""Numerical and serving-integrity tests for the affinity-profile model."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from profile_model import AffinityProfile,tanimoto,normalize


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(814)
        self.p=self.rng.normal(size=(5,7))
        self.c=(self.rng.random((6,20))>.65).astype(float)
        self.y=self.rng.normal(6,1,(5,6))
        self.qp=self.rng.normal(size=(3,7))
        self.qc=(self.rng.random((4,20))>.65).astype(float)

    def test_against_direct_kronecker_solve(self):
        m=AffinityProfile(.1,.7).fit(self.p,self.c,self.y)
        kp=m.protein_kernel(self.p);kc=tanimoto(self.c,self.c)
        residual=self.y-self.y.mean(0,keepdims=True)
        direct=np.linalg.solve(np.kron(kp,kc)+.7*np.eye(30),residual.ravel()).reshape(5,6)
        expected=m.chemical(self.qc)[None]+m.protein_kernel(self.qp)@direct@tanimoto(self.qc,self.c).T
        np.testing.assert_allclose(m.predict(self.qp,self.qc),expected,atol=1e-10,rtol=1e-10)

    def test_training_order_and_query_batch_invariance(self):
        m=AffinityProfile().fit(self.p,self.c,self.y)
        pr=[4,2,0,3,1];cr=[5,1,4,2,0,3]
        perm=AffinityProfile().fit(self.p[pr],self.c[cr],self.y[np.ix_(pr,cr)])
        np.testing.assert_allclose(m.predict(self.qp,self.qc),perm.predict(self.qp,self.qc),atol=1e-10)
        singles=np.concatenate([m.predict(self.qp,self.qc[i:i+1]) for i in range(4)],axis=1)
        np.testing.assert_allclose(m.predict(self.qp,self.qc),singles,atol=1e-10)

    def test_no_fabricated_interaction_when_labels_are_target_independent(self):
        y=np.broadcast_to(np.arange(6,dtype=float),(5,6))
        m=AffinityProfile().fit(self.p,self.c,y)
        expected=np.broadcast_to(m.chemical(self.qc),(3,4))
        np.testing.assert_allclose(m.predict(self.qp,self.qc),expected,atol=1e-10)

    def test_roundtrip_and_refuse_overwrite(self):
        m=AffinityProfile().fit(self.p,self.c,self.y)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'profile.npz';m.save(path)
            n=AffinityProfile.load(path)
            np.testing.assert_allclose(m.predict(self.qp,self.qc),n.predict(self.qp,self.qc),atol=1e-12)
            with self.assertRaises(RuntimeError):m.save(path)


if __name__=='__main__':unittest.main(verbosity=2)
