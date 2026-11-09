"""Mathematical stress tests for fair PerceiverCPI vs DynaCPI K-shot arms.

Only fictional targets and fabricated affinities; no real BindingDB labels
or prospective-test data are read, decoded, trained or scored.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import math
import sys
from pathlib import Path
import unittest

from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.episode_data import PredictionEpisode, SupportObservation, QueryCompound
from dynacpi.perceiver_disagreement_fewshot import (
    compare_perceiver_fewshot, GeometryDisagreementSettings,
)

SMILES=["CCO","CCCO","CCCCO","CC(C)O","CCN","CCCCCO","CCCN"]
def comp(s):
    smi=Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=True)
    return hashlib.sha256(smi.encode("utf8")).hexdigest(),smi

def fiction(k=5,geometry=True,constant_disagreement=False,constant_residual=False):
    c=[comp(s) for s in SMILES]
    support=[]
    seq={i:6.0 for i,_ in c}
    diff=[-.4,-.2,.05,.3,.45,.5,-.3]
    if constant_disagreement:
        diff=[.2 for _ in c]
    geom={c[i][0]:seq[c[i][0]]+diff[i] for i in range(len(c))}
    for i in range(k):
        y=6.0+.35+(0.0 if constant_residual else .75*diff[i])
        support.append(SupportObservation(c[i][0],c[i][1],y,"fake_assay"))
    queries=tuple(QueryCompound(cid,smiles) for cid,smiles in c[5:7])
    ep=PredictionEpisode(task_id="fake_target",sequence_sha256="fictional_seq",
       protein_component_id="fictional_component",assay_key="fake_assay",
       support=tuple(support),queries=queries,status="PRE_SPLIT_UNAPPROVED")
    keys={x.compound_id for x in (*ep.support,*ep.queries)}
    if not geometry:
        geom=None
    else:
        geom={i:geom[i] for i in keys}
    return ep,{i:seq[i] for i in keys},geom


class FairFewShotComparisonTests(unittest.TestCase):
    def test_all_arms_aligned_queries_and_same_support(self):
        ep,p0,geom=fiction()
        rows=compare_perceiver_fewshot(ep,p0,geom,True)
        self.assertEqual([x.compound_id for x in rows],
                         [x.compound_id for x in ep.queries])
        self.assertEqual([x.support_used for x in rows],[5,5])
        self.assertTrue(all(x.p0==6.0 for x in rows))
        self.assertTrue(all(x.d5_no_geometry==x.p5_global for x in rows))
        self.assertTrue(all(0.0<=x.geometry_reliability<=1.0 for x in rows))

    def test_default_settings_allow_informative_support_disagreement(self):
        ep,p0,geom=fiction()
        rows=compare_perceiver_fewshot(ep,p0,geom,True)
        self.assertTrue(all(x.geometry_active for x in rows))
        self.assertTrue(any(abs(x.d5_geometry_delta)>1e-8 for x in rows))
        self.assertTrue(all(abs(x.d5_geometry_delta)<=0.5 for x in rows))

    def test_manufactured_geometry_signal_can_beat_fair_controls(self):
        # This tiny truth vector is fabricated from the same formula as the
        # synthetic support labels. It proves algebraic capacity, NOT accuracy.
        ep,p0,geom=fiction()
        rows=compare_perceiver_fewshot(ep,p0,geom,True)
        fake_truth=[6.0+.35+.75*.5,6.0+.35+.75*(-.3)]
        def rmse(field):
            return math.sqrt(sum((getattr(row,field)-y)**2
                                 for row,y in zip(rows,fake_truth))/len(rows))
        self.assertLess(rmse("d5_geometry"),
                        min(rmse(k) for k in (
                            "p0","p5_global","p5_local","p5_kernel_ridge")))
        self.assertEqual([x.d5_no_geometry for x in rows],
                         [x.p5_global for x in rows])

    def test_insufficient_effective_support_disables_geometry(self):
        ep,p0,geom=fiction()
        rows=compare_perceiver_fewshot(ep,p0,geom,True,
            geometry_settings=GeometryDisagreementSettings(
                min_effective_support=3.0))
        self.assertTrue(all(x.d5_geometry==x.p5_global and
                            not x.geometry_active for x in rows))

    def test_informative_disagreement_can_modify_global_control(self):
        ep,p0,geom=fiction()
        rows=compare_perceiver_fewshot(ep,p0,geom,True,
            geometry_settings=GeometryDisagreementSettings(
                ridge_fraction=.001,min_effective_support=1.5))
        self.assertTrue(any(x.geometry_active for x in rows))
        self.assertTrue(any(abs(x.d5_geometry_delta)>1e-8 for x in rows))
        self.assertTrue(all(abs(x.d5_geometry_delta)<=0.5 for x in rows))

    def test_missing_geometry_exact_same_support_global_control(self):
        ep,p0,_=fiction(geometry=False)
        rows=compare_perceiver_fewshot(ep,p0,None,False)
        self.assertTrue(all(x.d5_geometry==x.p5_global==x.d5_no_geometry
                            for x in rows))
        self.assertTrue(all(x.theta==0 and not x.geometry_active and
                            x.geometry_reliability==0 for x in rows))

    def test_zero_support_exact_baseline(self):
        ep,p0,_=fiction(k=0,geometry=False)
        rows=compare_perceiver_fewshot(ep,p0,None,False)
        self.assertTrue(all(x.p0==x.p5_local==x.p5_kernel_ridge==
                            x.d5_geometry for x in rows))
        ep,p0,geom=fiction(k=0)
        rows=compare_perceiver_fewshot(ep,p0,geom,True)
        self.assertTrue(all(x.d5_geometry==x.p0 for x in rows))

    def test_low_support_cannot_derive_geometry_slope(self):
        for k in (1,2):
            ep,p0,geom=fiction(k)
            rows=compare_perceiver_fewshot(ep,p0,geom,True)
            self.assertTrue(all(x.d5_geometry==x.p5_global and
                                not x.geometry_active for x in rows))

    def test_constant_disagreement_is_non_identifiable(self):
        ep,p0,geom=fiction(constant_disagreement=True)
        rows=compare_perceiver_fewshot(ep,p0,geom,True)
        self.assertTrue(all(x.d5_geometry==x.p5_global and
                            x.disagreement_variation<1e-6 for x in rows))

    def test_constant_support_residual_does_not_masquerade_as_geometry(self):
        ep,p0,geom=fiction(constant_residual=True)
        rows=compare_perceiver_fewshot(ep,p0,geom,True)
        self.assertTrue(all(abs(x.d5_geometry_delta)<1e-12 for x in rows))

    def test_constant_assay_offset_changes_calibration_not_geometry_effect(self):
        ep,p0,geom=fiction()
        base=compare_perceiver_fewshot(ep,p0,geom,True)
        shifted=replace(ep,support=tuple(replace(s,pKd=s.pKd+1.0) for s in ep.support))
        new=compare_perceiver_fewshot(shifted,p0,geom,True)
        for b,n in zip(base,new):
            self.assertAlmostEqual(b.theta,n.theta,places=10)
            self.assertAlmostEqual(b.d5_geometry_delta,n.d5_geometry_delta,places=10)

    def test_constant_geometry_bias_cancels_via_centering(self):
        ep,p0,geom=fiction()
        original=compare_perceiver_fewshot(ep,p0,geom,True)
        shifted={cid:pred+3.0 for cid,pred in geom.items()}
        altered=compare_perceiver_fewshot(ep,p0,shifted,True)
        for a,b in zip(original,altered):
            self.assertAlmostEqual(a.d5_geometry_delta,b.d5_geometry_delta,places=9)
            self.assertAlmostEqual(a.theta,b.theta,places=9)

    def test_extreme_query_disagreement_cannot_escape_delta_cap(self):
        ep,p0,geom=fiction()
        for q in ep.queries:
            geom[q.compound_id]+=10000
        rows=compare_perceiver_fewshot(ep,p0,geom,True)
        self.assertTrue(all(abs(x.d5_geometry_delta)<=0.5 for x in rows))
        self.assertTrue(all(x.d5_geometry==x.p5_global+x.d5_geometry_delta for x in rows))

    def test_support_permutation_cannot_change_outputs(self):
        ep,p0,geom=fiction()
        x=compare_perceiver_fewshot(ep,p0,geom,True)
        swapped=replace(ep,support=ep.support[::-1])
        y=compare_perceiver_fewshot(swapped,p0,geom,True)
        for a,b in zip(x,y):
            for key in ("p5_local","p5_kernel_ridge","d5_geometry","theta"):
                self.assertAlmostEqual(getattr(a,key),getattr(b,key),places=9)

    def test_geometry_cache_must_cover_exact_same_task(self):
        ep,p0,geom=fiction()
        geom.pop(ep.support[0].compound_id)
        with self.assertRaisesRegex(ValueError,"cover exactly"):
            compare_perceiver_fewshot(ep,p0,geom,True)
        ep,p0,geom=fiction()
        with self.assertRaisesRegex(ValueError,"requires no challenger"):
            compare_perceiver_fewshot(ep,p0,geom,False)

    def test_nonfinite_geometry_predictions_rejected(self):
        ep,p0,geom=fiction()
        geom[ep.queries[0].compound_id]=float("nan")
        with self.assertRaisesRegex(ValueError,"non-finite"):
            compare_perceiver_fewshot(ep,p0,geom,True)

    def test_geometry_availability_must_be_explicit_bool(self):
        ep,p0,geom=fiction()
        with self.assertRaisesRegex(TypeError,"task-level bool"):
            compare_perceiver_fewshot(ep,p0,geom,1)

    def test_query_labels_cannot_enter_predictor(self):
        ep,p0,geom=fiction()
        self.assertFalse(any(hasattr(q,"pKd") for q in ep.queries))
        self.assertEqual(len(compare_perceiver_fewshot(ep,p0,geom,True)),len(ep.queries))

    def test_invalid_settings_rejected(self):
        ep,p0,geom=fiction()
        with self.assertRaises(ValueError):
            compare_perceiver_fewshot(ep,p0,geom,True,
                geometry_settings=GeometryDisagreementSettings(ridge_fraction=0))
        with self.assertRaises(ValueError):
            compare_perceiver_fewshot(ep,p0,geom,True,
                geometry_settings=GeometryDisagreementSettings(min_support=1))

if __name__=="__main__":
    unittest.main(verbosity=2)
