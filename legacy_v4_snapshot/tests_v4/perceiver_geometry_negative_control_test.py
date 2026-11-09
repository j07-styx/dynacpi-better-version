"""Source-independent synthetic stress tests for geometry disagreement shuffle.

All molecules here are fictional examples and all 'affinities' are invented.
No BindingDB files, holdout labels, or scientific benchmark are accessed.
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
from dynacpi.episode_data import (
    PredictionEpisode,QueryCompound,SupportObservation,
)
from dynacpi.perceiver_geometry_negative_control import (
    shuffle_compound_disagreement,compare_geometry_negative_control,
)
from dynacpi.perceiver_disagreement_fewshot import compare_perceiver_fewshot


def fiction(k=5):
    compounds=["CCO","CCCO","CCCCO","CC(C)O","CCN","CCCCCO","CCCN"]
    smi=[Chem.MolToSmiles(Chem.MolFromSmiles(x),isomericSmiles=True,canonical=True)
         for x in compounds]
    ids=[hashlib.sha256(s.encode()).hexdigest() for s in smi]
    per={cid:6.0+0.01*i for i,cid in enumerate(ids)}
    diff=[-0.4,-0.2,0.05,0.3,0.45,0.5,-0.3]
    geometry={cid:per[cid]+diff[i] for i,cid in enumerate(ids)}
    support=tuple(SupportObservation(ids[i],smi[i],per[ids[i]]+.35+.75*diff[i],
                                     "synthetic_assay") for i in range(k))
    queries=tuple(QueryCompound(ids[i],smi[i]) for i in range(5,7))
    episode=PredictionEpisode(
        task_id="fictional_task",sequence_sha256="fake_protein",
        protein_component_id="fake_component",assay_key="synthetic_assay",
        support=support,queries=queries,status="PRE_SPLIT_UNAPPROVED")
    used={x.compound_id for x in (*support,*queries)}
    return episode,{cid:per[cid] for cid in used},{cid:geometry[cid] for cid in used}


class GeometryNegativeControlTests(unittest.TestCase):
    def test_deterministic_derangement_preserves_residual_distribution(self):
        episode,p,g=fiction()
        shuffle=shuffle_compound_disagreement(episode,p,g)
        shuffle2=shuffle_compound_disagreement(episode,p,g)
        self.assertEqual(shuffle,shuffle2)
        before=sorted(round(g[cid]-p[cid],7) for cid in p)
        after=sorted(round(shuffle[cid]-p[cid],7) for cid in p)
        self.assertEqual(before,after)
        # The toy residuals are distinct, so no item can retain its own.
        self.assertTrue(all(not math.isclose(shuffle[cid],g[cid],
                                               abs_tol=1e-10) for cid in p))

    def test_shuffle_changes_when_seed_changes_without_touching_support(self):
        episode,p,g=fiction()
        s1=shuffle_compound_disagreement(episode,p,g,shuffle_seed=2601)
        s2=shuffle_compound_disagreement(episode,p,g,shuffle_seed=2602)
        self.assertTrue(any(abs(s1[c]-s2[c])>1e-8 for c in p))
        self.assertEqual(episode.support,episode.support)

    def test_permutation_invariant_to_support_tuple_order(self):
        episode,p,g=fiction()
        alternate=replace(episode,support=episode.support[::-1],
                          queries=episode.queries[::-1])
        self.assertEqual(shuffle_compound_disagreement(episode,p,g),
                         shuffle_compound_disagreement(alternate,p,g))

    def test_label_perturbation_does_not_change_shuffle(self):
        episode,p,g=fiction()
        shifted=replace(episode,support=tuple(
            replace(s,pKd=s.pKd+100) for s in episode.support))
        self.assertEqual(shuffle_compound_disagreement(episode,p,g),
                         shuffle_compound_disagreement(shifted,p,g))

    def test_all_arms_have_identical_support_and_p5_controls(self):
        episode,p,g=fiction()
        output=compare_geometry_negative_control(episode,p,g,True)
        self.assertEqual(len(output.original),len(output.shuffled))
        for original,shuffled,null in zip(output.original,output.shuffled,output.no_geometry):
            self.assertEqual(original.compound_id,shuffled.compound_id)
            self.assertEqual(original.compound_id,null.compound_id)
            for a in ("p0","p5_global","p5_local","p5_kernel_ridge"):
                self.assertEqual(getattr(original,a),getattr(shuffled,a))
                self.assertEqual(getattr(original,a),getattr(null,a))
            self.assertEqual(null.d5_geometry,original.p5_global)
            self.assertEqual(null.d5_geometry_delta,0.0)
            self.assertFalse(null.geometry_active)

    def test_no_geometry_task_has_exact_control_fallback(self):
        episode,p,_=fiction()
        report=compare_geometry_negative_control(episode,p,None,False)
        self.assertEqual(report.original,report.shuffled)
        self.assertEqual(report.original,report.no_geometry)
        self.assertTrue(all(x.d5_geometry==x.p5_global for x in report.original))

    def test_original_matches_unmodified_adapter(self):
        episode,p,g=fiction()
        reference=compare_perceiver_fewshot(episode,p,g,True)
        report=compare_geometry_negative_control(episode,p,g,True)
        self.assertEqual(reference,report.original)

    def test_excessive_seed_or_invalid_type_rejected(self):
        episode,p,g=fiction()
        for seed in (-1,True,1.5):
            with self.assertRaises(ValueError):
                shuffle_compound_disagreement(episode,p,g,shuffle_seed=seed)
            with self.assertRaises(ValueError):
                compare_geometry_negative_control(episode,p,g,True,shuffle_seed=seed)

    def test_exact_frozen_prediction_cache_and_finite_required(self):
        episode,p,g=fiction()
        broken=dict(g)
        broken.pop(next(iter(broken)))
        with self.assertRaisesRegex(ValueError,"exact episode"):
            shuffle_compound_disagreement(episode,p,broken)
        broken=dict(g)
        broken[next(iter(broken))]=float("nan")
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            shuffle_compound_disagreement(episode,p,broken)

    def test_k0_stays_unadapted_and_query_labels_inaccessible(self):
        episode,p,g=fiction(k=0)
        result=compare_geometry_negative_control(episode,p,g,True)
        self.assertTrue(all(not hasattr(q,"pKd") for q in episode.queries))
        for row in (*result.original,*result.shuffled):
            self.assertEqual(row.d5_geometry,row.p0)

    def test_identical_disagreement_is_no_geometry_effect(self):
        episode,p,g=fiction()
        flat={cid:p[cid]+0.4 for cid in p}
        output=compare_geometry_negative_control(episode,p,flat,True)
        for row in (*output.original,*output.shuffled):
            self.assertEqual(row.d5_geometry,row.p5_global)
            self.assertEqual(row.d5_geometry_delta,0.0)

if __name__=="__main__":
    unittest.main(verbosity=2)
