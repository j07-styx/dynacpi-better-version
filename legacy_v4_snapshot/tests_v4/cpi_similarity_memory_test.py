"""Similarity-memory regression unit tests: exact weights, controls and fallback."""
import unittest
import numpy as np
from dynacpi.cpi_similarity_memory_dev import (
    MEMORY_SPECS,MemorySpec,neighbor_weighted_labels
)

class SimilarityMemoryTests(unittest.TestCase):
    def test_nearest_training_label(self):
        chem=np.array([[1.,.05,.10]],dtype=np.float32)
        protein=np.ones_like(chem)
        labels=np.array([7.,1.,3.])
        spec=MemorySpec("nearest",2,0,1)
        result=neighbor_weighted_labels(chem,protein,labels,2.,spec)
        self.assertAlmostEqual(result[0],7.)

    def test_protein_similarity_weights_affinity_neighbor(self):
        chem=np.array([[.8,.8]],dtype=np.float32)
        protein=np.array([[1.,.1]],dtype=np.float32)
        labels=np.array([9.,2.])
        spec=MemorySpec("joint",2,8,2)
        joint=neighbor_weighted_labels(chem,protein,labels,5.,spec)[0]
        spec_chem=MemorySpec("chem",2,0,2)
        chem_result=neighbor_weighted_labels(chem,protein,labels,5.,spec_chem)[0]
        self.assertGreater(joint,8.9)
        self.assertAlmostEqual(chem_result,5.5)

    def test_no_neighbors_fall_back_to_fold_train_mean(self):
        c=np.zeros((2,3),dtype=np.float32)
        p=np.ones_like(c)
        output=neighbor_weighted_labels(c,p,np.array([1.,2.,3.]),
                                        6.5,MEMORY_SPECS[0])
        np.testing.assert_array_equal(output,[6.5,6.5])

    def test_row_blocking_independent(self):
        rng=np.random.RandomState(2103)
        a=rng.uniform(size=(17,29)).astype(np.float32)
        b=rng.uniform(size=(17,29)).astype(np.float32)
        labels=rng.normal(size=29)
        spec=MemorySpec("joint",2,8,9)
        direct=neighbor_weighted_labels(a,b,labels,0.,spec,block_size=32)
        blocked=neighbor_weighted_labels(a,b,labels,0.,spec,block_size=3)
        np.testing.assert_allclose(direct,blocked,rtol=1e-10,atol=1e-10)

    def test_bad_similarities_rejected(self):
        with self.assertRaises(ValueError):
            neighbor_weighted_labels(np.zeros((2,2)),np.zeros((3,2)),
                                     [1.,2.],0.,MEMORY_SPECS[0])
        with self.assertRaises(ValueError):
            neighbor_weighted_labels(np.full((2,2),np.nan),
                                     np.zeros((2,2)),[1.,2.],0.,
                                     MEMORY_SPECS[0])

    def test_explicit_specs_predefined(self):
        self.assertEqual(len(MEMORY_SPECS),4)
        self.assertEqual([s.neighbor_count for s in MEMORY_SPECS],
                         [64,64,64,256])
        self.assertTrue(all(s.ligand_exponent>=2 for s in MEMORY_SPECS))

if __name__=="__main__":unittest.main()
