"""Unit tests for chemical-similarity gated CPI blending."""
import unittest
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from scripts.train_balm_bindingdb_chemical_domain_gated_ensemble_dev import (
    nearest_chemical_cosine, blend, MODES
)

class ChemicalDomainGateTests(unittest.TestCase):
    def test_fixed_gate_is_exact_quarter(self):
        ridge=np.array([1.,2.,3.])
        boost=np.array([5.,6.,7.])
        sim=np.array([0.,.5,1.])
        np.testing.assert_allclose(blend(ridge,boost,sim,"fixed25"), .75*ridge+.25*boost)
    def test_opposed_novelty_gates(self):
        ridge=np.zeros(2)
        boost=np.full(2,10.)
        nearest=np.array([0.,1.])
        np.testing.assert_allclose(blend(ridge,boost,nearest,"low_similarity_higher_boost"),[4.,1.])
        np.testing.assert_allclose(blend(ridge,boost,nearest,"high_similarity_higher_boost"),[1.,4.])
    def test_gate_rejects_undefined_mode(self):
        with self.assertRaises(ValueError):
            blend(np.zeros(1),np.ones(1),np.zeros(1),"undefined")
    def test_batches_do_not_overwrite_first_queries(self):
        gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
        # Same valid compounds repeated in nontrivial order, batch size one.
        tr=[{"smiles":"CCO"},{"smiles":"c1ccccc1"}]
        va=[{"smiles":"CCN"},{"smiles":"CCO"},{"smiles":"c1ccccc1"}]
        a=nearest_chemical_cosine(tr,va,gen,{},batch=1)
        b=nearest_chemical_cosine(tr,va,gen,{},batch=192)
        np.testing.assert_allclose(a,b)
        self.assertEqual(len(a),len(va))
    def test_self_match_and_reverse_order(self):
        gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
        tr=[{"smiles":"CCO"},{"smiles":"c1ccccc1"},{"smiles":"CC(=O)O"}]
        va=[{"smiles":"CCO"},{"smiles":"CCN"},{"smiles":"c1ccccc1"}]
        result=nearest_chemical_cosine(tr,va,gen,{})
        self.assertAlmostEqual(result[0],1.,places=6)
        self.assertAlmostEqual(result[2],1.,places=6)
        self.assertTrue(np.all((result>=0)&(result<=1)))
        reversed_result=nearest_chemical_cosine(list(reversed(tr)),list(reversed(va)),gen,{})
        np.testing.assert_allclose(result[::-1],reversed_result)

if __name__=="__main__": unittest.main()

