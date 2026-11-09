"""Synthetic equal-support Perceiver controls; no real BindingDB evaluation."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import sys
from pathlib import Path
import unittest

from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.episode_data import PredictionEpisode, SupportObservation, QueryCompound
from dynacpi.perceiver_equal_support_controls import (
    perceiver_equal_support_controls,PerceiverSupportSettings,
)


def make_episode(k=2):
    smiles=["CCO","CCN","CCC","CCCl"]
    canonical=[Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=True)
               for s in smiles]
    ids=[hashlib.sha256(s.encode("utf-8")).hexdigest() for s in canonical]
    support=tuple(SupportObservation(ids[i],canonical[i],7.0+i,"fictional_assay")
                  for i in range(k))
    query=tuple(QueryCompound(ids[i],canonical[i]) for i in range(2,4))
    episode=PredictionEpisode(task_id="synthetic_only",sequence_sha256="synthetic",
        protein_component_id="synthetic_component",assay_key="fictional_assay",
        support=support,queries=query,status="PRE_SPLIT_UNAPPROVED")
    pred={obj.compound_id:6.0 for obj in (*support,*query)}
    return episode,pred


class PerceiverEqualSupportTests(unittest.TestCase):
    def test_zero_support_all_controls_exact_baseline(self):
        ep,p=make_episode(k=0)
        out=perceiver_equal_support_controls(ep,p)
        self.assertEqual(len(out),2)
        self.assertTrue(all(x.p0==x.global_offset==x.local_offset==
                            x.kernel_ridge==6.0 and x.support_used==0 for x in out))

    def test_same_labels_for_each_support_aware_control(self):
        ep,p=make_episode()
        out=perceiver_equal_support_controls(ep,p)
        self.assertEqual(len(out),2)
        for x in out:
            self.assertEqual(x.p0,6.0)
            self.assertEqual(x.support_used,2)
            self.assertGreater(x.global_offset,6.0)
            self.assertGreaterEqual(x.local_offset,6.0)
            self.assertTrue(5.0<=x.kernel_ridge<=7.0)

    def test_support_order_invariance(self):
        ep,p=make_episode()
        x=perceiver_equal_support_controls(ep,p)
        y=perceiver_equal_support_controls(replace(ep,support=ep.support[::-1]),p)
        for a,b in zip(x,y):
            self.assertAlmostEqual(a.global_offset,b.global_offset,places=12)
            self.assertAlmostEqual(a.local_offset,b.local_offset,places=12)
            self.assertAlmostEqual(a.kernel_ridge,b.kernel_ridge,places=12)

    def test_caps_with_extreme_fake_support_residuals(self):
        ep,p=make_episode()
        large=replace(ep,support=tuple(replace(obs,pKd=1000.0) for obs in ep.support))
        out=perceiver_equal_support_controls(large,p)
        for o in out:
            self.assertLessEqual(abs(o.global_offset-o.p0),1.0000001)
            self.assertLessEqual(abs(o.local_offset-o.p0),1.0000001)
            self.assertLessEqual(abs(o.kernel_ridge-o.p0),1.0000001)

    def test_query_labels_never_accepted_as_prediction_input(self):
        ep,p=make_episode()
        self.assertFalse(hasattr(ep.queries[0],"pKd"))
        out=perceiver_equal_support_controls(ep,p)
        self.assertEqual(len(out),len(ep.queries))

    def test_exact_prediction_cache_required(self):
        ep,p=make_episode()
        p.pop(ep.support[0].compound_id)
        with self.assertRaisesRegex(ValueError,"cover exactly"):
            perceiver_equal_support_controls(ep,p)

    def test_wrong_hash_or_support_query_overlap_rejected(self):
        ep,p=make_episode()
        broken=replace(ep,queries=(replace(ep.queries[0],compound_id="invalid"),))
        with self.assertRaises(ValueError):
            perceiver_equal_support_controls(broken,{**p,"invalid":6.0})
        duplicate=replace(ep,queries=(QueryCompound(ep.support[0].compound_id,
                                                   ep.support[0].smiles),))
        with self.assertRaisesRegex(ValueError,"Support/query overlap"):
            perceiver_equal_support_controls(duplicate,p)

    def test_nonfinite_support_and_predictions_rejected(self):
        ep,p=make_episode()
        p[ep.queries[0].compound_id]=float("nan")
        with self.assertRaisesRegex(ValueError,"Nonfinite Perceiver"):
            perceiver_equal_support_controls(ep,p)
        ep,p=make_episode()
        broken=replace(ep,support=(replace(ep.support[0],pKd=float("nan")),ep.support[1]))
        with self.assertRaisesRegex(ValueError,"Nonfinite support"):
            perceiver_equal_support_controls(broken,p)

    def test_reject_negative_regularization_or_tampered_kernel(self):
        ep,p=make_episode()
        with self.assertRaisesRegex(ValueError,"Positive finite"):
            perceiver_equal_support_controls(ep,p,
                PerceiverSupportSettings(lambda_kernel=-1))
        with self.assertRaisesRegex(ValueError,"Tanimoto"):
            perceiver_equal_support_controls(ep,p,
                PerceiverSupportSettings(tanimoto_power=2))

if __name__=="__main__":
    unittest.main(verbosity=2)
