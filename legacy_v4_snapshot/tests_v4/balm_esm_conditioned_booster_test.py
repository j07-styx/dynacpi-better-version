"""Tests: train-only PCA and ESM-conditioned nonlinear CPI input invariants."""
import hashlib
import unittest

import numpy as np

from scripts.compare_balm_esm_pca_nonlinear_booster_alignment_dev import (
    embed_matrix, transform_train_only_pca, PCA_DIMS, NAMES,
)
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import cache_feature_map


class ESMConditionedBoosterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features,cls.manifest=cache_feature_map()

    def test_train_unique_pca_and_shapes(self):
        seqs=sorted(self.features)
        # For isolated feature-encoder tests, protein field is only a key; use
        # hashes as fake strings and look them up via a unit-test mapping.
        fake=[{"protein":"SEQ"+str(i)} for i in range(52)]
        hcache={"SEQ"+str(i):seqs[i] for i in range(52)}
        training=fake[:45]+fake[:8]
        validation=fake[45:52]
        encoded,held,report=transform_train_only_pca(
            training,validation,self.features,hcache,16)
        self.assertEqual(encoded.shape,(53,16))
        self.assertEqual(held.shape,(7,16))
        self.assertEqual(report["training_unique_proteins"],45)
        self.assertLessEqual(report["training_variance_explained"],1.000001)
        self.assertTrue(np.isfinite(held).all())

    def test_embedded_sequence_order_stable(self):
        seqs=sorted(self.features)
        hcache={"SEQ_A":seqs[10],"SEQ_B":seqs[11]}
        original=embed_matrix(
            [{"protein":"SEQ_A"},{"protein":"SEQ_B"}],self.features,hcache)
        reversed_=embed_matrix(
            [{"protein":"SEQ_B"},{"protein":"SEQ_A"}],self.features,hcache)
        np.testing.assert_array_equal(original[::-1],reversed_)
        np.testing.assert_allclose(np.linalg.norm(original,axis=1),1,atol=1e-5)

    def test_proposed_models_and_pca_dimensions(self):
        self.assertEqual(PCA_DIMS,(16,32))
        self.assertIn("nested_reference_joint",NAMES)
        self.assertIn("nested_esm_pca32_joint",NAMES)
        self.assertIn("nested_esm_pca32_no_aa",NAMES)

    def test_source_cache_immutable(self):
        self.assertEqual(self.manifest["protein_embeddings"],639)
        self.assertEqual(self.manifest["embedding_dimensions"],320)

if __name__=="__main__":
    unittest.main()
