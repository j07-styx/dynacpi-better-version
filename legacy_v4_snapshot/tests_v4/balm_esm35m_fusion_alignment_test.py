"""Frozen 8M / 35M ESM2 representation fusion unit and report tests."""
import hashlib
import json
import unittest

import numpy as np

from dynacpi.balm_public_data_dev import ROOT,get_train_only
from scripts.compare_balm_esm2_8m_vs_35m_alignment_cold_dev import (
    load_35m,fused_features,RULES,MODELS,
)
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import (
    cache_feature_map,
)

class ESM35MFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.older,_=cache_feature_map()
        cls.larger,cls.manifest=load_35m()

    def test_checksum_encoder_and_train_only_source(self):
        self.assertEqual(self.manifest["upstream_pretrained"],"esm2_t12_35M_UR50D")
        self.assertEqual(self.manifest["protein_embeddings"],639)
        self.assertEqual(self.manifest["embedding_dimensions"],480)
        self.assertEqual(self.manifest["max_nterm_aa_tokens"],1022)
        expected={hashlib.sha256(r["protein"].encode()).hexdigest()
                  for r in get_train_only()}
        self.assertEqual(set(self.larger),expected)
        self.assertEqual(set(self.older),expected)

    def test_feature_norm_and_dtype(self):
        larger_matrix=np.stack(list(self.larger.values()))
        self.assertEqual(larger_matrix.shape,(639,480))
        self.assertTrue(np.isfinite(larger_matrix).all())
        np.testing.assert_allclose(np.linalg.norm(larger_matrix,axis=1),1.,
                                   rtol=1e-5,atol=1e-5)

    def test_fusion_preserves_equal_weighting_and_unit_norm(self):
        fused=fused_features(self.older,self.larger)
        self.assertEqual(set(fused),set(self.older))
        key=next(iter(fused))
        self.assertEqual(fused[key].shape,(800,))
        np.testing.assert_allclose(fused[key][:320],
                                   self.older[key]/np.sqrt(2.),atol=1e-7)
        np.testing.assert_allclose(fused[key][320:],
                                   self.larger[key]/np.sqrt(2.),atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(fused[key])),1.,places=5)

    def test_report_reconciles_reference(self):
        for rule in RULES:
            path=ROOT/"research"/f"balm_esm2_8m_vs_35m_fusion_sw_{rule}_scaffold_cold_traincv_dev.json"
            d=json.loads(path.read_text(encoding="utf-8"))
            prior=json.loads((ROOT/"research"/f"balm_esm8m_nested_sw_{rule}_murcko_scaffold_cold_traincv_dev.json").read_text())
            self.assertEqual(len(d["outer_folds"]),4)
            self.assertEqual(sum(f["heldout_rows"] for f in d["outer_folds"]),15538)
            self.assertEqual(sum(f["heldout_alignment_components"] for f in d["outer_folds"]),
                             d["alignment_groups"])
            self.assertTrue(all(f["scaffold_overlap"]==0 for f in d["outer_folds"]))
            self.assertEqual(set(d["out_of_fold"]["pooled_rmse"]),set(MODELS))
            for metric in ("pooled_rmse","component_macro_rmse"):
                self.assertAlmostEqual(d["out_of_fold"][metric]["nested_8m"],
                                       prior["out_of_fold"][metric]["nested_kmer_esm"],places=8)
            self.assertEqual(d["out_of_fold"]["bootstrap"]["replicates"],5000)

if __name__=="__main__":
    unittest.main()
