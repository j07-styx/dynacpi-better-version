"""Entirely fabricated tests for strong Perceiver-KRR-anchored geometry guard."""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import math
import sys
from pathlib import Path
import unittest

from rdkit import Chem

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.episode_data import PredictionEpisode,QueryCompound,SupportObservation
from dynacpi.perceiver_equal_support_controls import perceiver_equal_support_controls
from dynacpi.perceiver_krr_geometry_guard import (
    KRRGeometrySettings,compare_krr_geometry_with_support_validation
)
from dynacpi.perceiver_geometry_negative_control import shuffle_compound_disagreement

SMILES=["CCO","CCCO","CCCCO","CC(C)O","CCN","CCCCCO","CCCN","CC(C)CO"]
DIFF=[-.5,-.35,-.1,.3,.5,.65,-.3,.45]

def fiction(k=6,mode="correlated"):
    canon=[Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=True)
           for s in SMILES]
    ids=[sha256(s.encode("utf8")).hexdigest() for s in canon]
    per={cid:6.0 for cid in ids}
    geometry={cid:per[cid]+d for cid,d in zip(ids,DIFF)}
    if mode=="constant_geometry":
        geometry={cid:per[cid]+.3 for cid in ids}
    support=[]
    for i in range(k):
        residual=.4+(
            .9*DIFF[i] if mode=="correlated" else
            -.9*DIFF[i] if mode=="anticorrelated" else
            0.0
        )
        support.append(SupportObservation(ids[i],canon[i],6.0+residual,"synthetic-only"))
    query=tuple(QueryCompound(ids[i],canon[i]) for i in range(6,8))
    episode=PredictionEpisode(task_id="fiction",sequence_sha256="fiction",
        protein_component_id="fiction",assay_key="synthetic-only",
        support=tuple(support),queries=query,status="PRE_SPLIT_UNAPPROVED")
    used={x.compound_id for x in (*support,*query)}
    return episode,{k:per[k] for k in used},{k:geometry[k] for k in used}


def run(episode,per,geometry,available=True,**kwargs):
    return compare_krr_geometry_with_support_validation(
        episode,per,geometry if available else None,available,**kwargs)


class KRRGeometryGuardTests(unittest.TestCase):
    def test_exact_perceiver_krr_baseline_preservation(self):
        e,p,g=fiction()
        old=perceiver_equal_support_controls(e,p)
        result=run(e,p,g)
        self.assertEqual([z.perceiver_krr_pKd for z in result],
                         [z.kernel_ridge for z in old])
        self.assertEqual([z.compound_id for z in result],
                         [z.compound_id for z in e.queries])

    def test_geometry_missing_exact_krr_fallback(self):
        e,p,g=fiction()
        out=run(e,p,g,available=False)
        self.assertTrue(all(z.geometry_krr_pKd==z.perceiver_krr_pKd and
                            z.geometry_delta==z.gate==0 for z in out))

    def test_k0_k1_k2_k3_k4_exact_no_geometry_change(self):
        for k in range(5):
            e,p,g=fiction(k)
            output=run(e,p,g)
            self.assertTrue(all(z.geometry_krr_pKd==z.perceiver_krr_pKd
                                for z in output))

    def test_constant_disagreement_cannot_add_geometry(self):
        e,p,g=fiction(mode="constant_geometry")
        rows=run(e,p,g)
        self.assertTrue(all(z.geometry_delta==0.0 for z in rows))

    def test_pure_calibration_intercept_cannot_add_geometry(self):
        e,p,g=fiction(mode="intercept")
        rows=run(e,p,g)
        self.assertTrue(all(abs(z.geometry_delta)<1e-10 for z in rows))

    def test_permutation_of_support_does_not_change_prediction(self):
        e,p,g=fiction()
        a=run(e,p,g)
        b=run(replace(e,support=e.support[::-1]),p,g)
        for left,right in zip(a,b):
            self.assertAlmostEqual(left.geometry_krr_pKd,right.geometry_krr_pKd,places=9)
            self.assertAlmostEqual(left.loo_relative_gain,right.loo_relative_gain,places=9)
            self.assertAlmostEqual(left.gate,right.gate,places=9)

    def test_delta_always_within_predeclared_cap(self):
        e,p,g=fiction()
        for cid in (q.compound_id for q in e.queries):
            g[cid]+=10000
        result=run(e,p,g)
        self.assertTrue(all(abs(z.geometry_delta)<=.5 for z in result))

    def test_invalid_or_unaligned_frozen_score_cache_rejected(self):
        e,p,g=fiction()
        broken=dict(g)
        broken.pop(e.support[0].compound_id)
        with self.assertRaisesRegex(ValueError,"cover exact"):
            run(e,p,broken)
        broken=dict(g)
        broken[e.support[0].compound_id]=float("nan")
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            run(e,p,broken)
        with self.assertRaisesRegex(ValueError,"Absent geometry"):
            compare_krr_geometry_with_support_validation(e,p,g,False)

    def test_invalid_guard_settings_and_availability_rejected(self):
        e,p,g=fiction()
        with self.assertRaises(TypeError):
            compare_krr_geometry_with_support_validation(e,p,g,1)
        with self.assertRaises(ValueError):
            run(e,p,g,geometry_settings=KRRGeometrySettings(geometry_ridge=0))
        with self.assertRaises(ValueError):
            run(e,p,g,geometry_settings=KRRGeometrySettings(min_support=2))
        with self.assertRaises(ValueError):
            run(e,p,g,geometry_settings=KRRGeometrySettings(min_relative_loo_gain=1))

    def test_query_labels_not_accessible_at_prediction_boundary(self):
        e,p,g=fiction()
        self.assertTrue(all(not hasattr(q,"pKd") for q in e.queries))
        out=run(e,p,g)
        self.assertEqual(len(out),2)
        self.assertTrue(all(math.isfinite(x.geometry_krr_pKd) for x in out))

    def test_shuffled_disagreement_keeps_krr_baseline_same(self):
        e,p,g=fiction()
        shuffled=shuffle_compound_disagreement(e,p,g,shuffle_seed=2601)
        a=run(e,p,g)
        b=run(e,p,shuffled)
        self.assertEqual([x.perceiver_krr_pKd for x in a],
                         [x.perceiver_krr_pKd for x in b])

    def test_loo_gate_recorded_and_bounds(self):
        e,p,g=fiction()
        result=run(e,p,g)
        self.assertTrue(all(math.isfinite(z.loo_krr_mse) and
                            math.isfinite(z.loo_geometry_mse) for z in result))
        self.assertTrue(all(0<=z.gate<=1 for z in result))
        self.assertEqual(len({z.gate for z in result}),1)

if __name__=="__main__":
    unittest.main(verbosity=2)
