"""Unit tests for physically interpretable explicit chemical-protein features."""
import unittest
import numpy as np
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import bio_interaction

class ExplicitInteractionFeatureTests(unittest.TestCase):
    def test_feature_count_and_product_identity(self):
        c=np.zeros((2,21),dtype=np.float32)
        p=np.zeros((2,21),dtype=np.float32)
        c[0,0]=10.;c[1,0]=20.
        p[0,2]=.2 # D in AMINO order
        p[1,2]=.4
        z=bio_interaction(c,p)
        self.assertEqual(z.shape,(2,106))
        np.testing.assert_array_equal(z[:,:21],c)
        np.testing.assert_array_equal(z[:,21:42],p)
        self.assertAlmostEqual(float(z[0,42]),2.,places=5)
        self.assertAlmostEqual(float(z[1,42]),8.,places=5)
    def test_invariant_to_record_order(self):
        rng=np.random.RandomState(2103)
        c=rng.random_sample((12,21)).astype(np.float32)
        p=rng.random_sample((12,21)).astype(np.float32)
        index=np.array([7,0,9,1,11,2,3,5,4,10,6,8])
        np.testing.assert_array_equal(
            bio_interaction(c,p)[index],bio_interaction(c[index],p[index]))
    def test_protein_changes_explicit_cross_features(self):
        c=np.ones((2,21),dtype=np.float32)
        p=np.zeros((2,21),dtype=np.float32)
        p[1,2]=.3
        z=bio_interaction(c,p)
        self.assertNotEqual(float(z[0,42]),float(z[1,42]))
    def test_finite(self):
        c=np.ones((1,21),dtype=np.float32)
        p=np.ones((1,21),dtype=np.float32)/20
        self.assertTrue(np.isfinite(bio_interaction(c,p)).all())

if __name__=="__main__":unittest.main()
