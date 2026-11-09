"""Unit and model smoke tests: research SMILES/protein affinity predictor."""
import hashlib,json,unittest
from pathlib import Path

import joblib
import numpy as np

from dynacpi.balm_public_data_dev import ROOT,get_train_only
from dynacpi.balm_memory_predictor_dev import (
    AA_ALPHABET,clean_protein,canonical_smiles,predict,predict_with_embedding,
    sequence_hash,
)

CHECKPOINT=ROOT/"data/models/balm_affinity_memory_prototype_exploratory_20261108.joblib"
MANIFEST=ROOT/"research/balm_affinity_memory_research_inference_manifest_20261108.json"

class ExperimentalMemoryPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads(MANIFEST.read_text(encoding="utf8"))
        cls.model=joblib.load(CHECKPOINT)

    def test_source_and_checkpoint_hash_integrity(self):
        self.assertEqual(
            self.meta["checkpoint_sha256"],hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
        self.assertEqual(self.meta["train_original_pairs"],15538)
        self.assertEqual(self.meta["train_unique_protein_smiles_pairs"],12315)
        self.assertEqual(self.model["train_fingerprints"].shape,(12315,2048))
        self.assertEqual(self.model["train_embeddings"].shape,(12315,320))
        self.assertEqual(self.model["train_labels"].shape,(12315,))

    def test_unapproved_gate_and_fixed_memory_architecture(self):
        self.assertEqual(self.meta["scientific_gate_status"]["status"],
                         "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")
        self.assertEqual(self.model["memory_spec"].name,"joint_esm8_k64")
        self.assertEqual(self.model["beta"],1.)
        self.assertEqual(self.model["memory_mix"],.25)

    def test_sequence_and_smiles_validation(self):
        self.assertEqual(clean_protein(" ma cd "*5),"MACD"*5)
        self.assertEqual(clean_protein(" A "*20),"A"*20)
        self.assertEqual(canonical_smiles("OCC"),canonical_smiles("CCO"))
        with self.assertRaises(ValueError):
            clean_protein("123")
        with self.assertRaises(ValueError):
            clean_protein("MAACD")
        with self.assertRaises(ValueError):
            canonical_smiles("not-a-smiles")

    def test_known_protein_recreates_training_smoke_without_evaluation_claim(self):
        rows=get_train_only()
        observed=[]
        for r in rows:
            protein=r["protein"]
            if len(protein)>=20 and set(protein).issubset(AA_ALPHABET):
                result=predict(self.model,r["smiles"],protein)
                direct=predict_with_embedding(
                    self.model,r["smiles"],protein,
                    self.model["known_embeddings"][sequence_hash(protein)])
                self.assertEqual(result["status"],"EXPLORATORY_UNVALIDATED_RESEARCH_ONLY")
                self.assertAlmostEqual(result["predicted_source_affinity_label"],
                                       direct["predicted_source_affinity_label"])
                self.assertTrue(np.isfinite(result["predicted_source_affinity_label"]))
                observed.append(result["predicted_source_affinity_label"])
                if len(observed)==3:break
        self.assertEqual(len(observed),3)
        for got,smoke in zip(observed,self.meta["smoke_predictions_train_only"]):
            self.assertAlmostEqual(got,smoke["estimated_source_p_affinity"],places=7)

if __name__=="__main__":
    unittest.main()

