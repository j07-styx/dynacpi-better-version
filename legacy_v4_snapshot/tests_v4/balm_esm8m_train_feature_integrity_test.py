"""Integrity tests for frozen TRAIN-only ESM2 and scaffold-cold feature ablations."""
import hashlib,json,unittest
from pathlib import Path
import numpy as np
from dynacpi.balm_public_data_dev import ROOT,get_train_only
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import (
    CACHE,MANIFEST,cache_feature_map,bootstrap
)

class ESM8MTrainOnlyFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_pretrained_frozen_encoder_manifest(self):
        info=self.manifest
        self.assertEqual(info["protein_embeddings"],639)
        self.assertEqual(info["embedding_dimensions"],320)
        self.assertEqual(info["pretrained_protein_encoder"] if "pretrained_protein_encoder" in info else info["upstream_pretrained"],
                         "esm2_t6_8M_UR50D")
        self.assertEqual(info["max_nterm_aa_tokens"],1022)
        self.assertEqual(info["proteins_truncated_to_window"],85)

    def test_cache_checksum_finite_and_normalized(self):
        self.assertEqual(hashlib.sha256(CACHE.read_bytes()).hexdigest(),
                         self.manifest["cache_sha256"])
        feature,_=cache_feature_map()
        self.assertEqual(len(feature),639)
        matrix=np.stack(list(feature.values()))
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_allclose(np.linalg.norm(matrix,axis=1),1.,atol=1e-5)

    def test_cache_exactly_matches_frozen_train_protein_hashes(self):
        features,_=cache_feature_map()
        expected={hashlib.sha256(r["protein"].encode()).hexdigest()
                  for r in get_train_only()}
        self.assertEqual(set(features),expected)
        self.assertEqual(len(expected),639)

    def test_group_bootstrap_supports_new_predeclared_comparator(self):
        scores={"train_mean":[3.,4.],"chem_kmer":[2.,2.],"chem_esm":[2.,2.]}
        output=bootstrap(scores,[4,5],replicates=64,seed=17,
                         references=("train_mean","chem_kmer"))
        self.assertEqual(output["bootstrap"]["replicates"],64)
        self.assertIn("chem_esm_minus_chem_kmer",output["bootstrap"]["comparisons"])
        self.assertAlmostEqual(
            output["bootstrap"]["comparisons"]["chem_esm_minus_chem_kmer"]["pooled_delta"],0.)

if __name__=="__main__":unittest.main()
