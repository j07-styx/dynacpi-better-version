"""Support-validated geometry augmentation on the strongest Perceiver KRR control.

Experimental DynaCPI hypothesis: add geometry to equal-support chemical
kernel ridge ONLY when leave-one-support-out error improves on the same task.
NO experimental data is loaded, NO model is trained, and NO scientific cohort
is approved. The input episode must come from a separately approved protocol
for biological claims. Query labels are structurally absent from the API.

This is not a final estimator or a claim of improvement: the 5-observation
support gate can be noisy and should be judged against KRR and shuffled
geometry on independent source-family-heldout data.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import math

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from dynacpi.episode_data import CandidateEpisodeStore, PredictionEpisode
from dynacpi.perceiver_equal_support_controls import (
    PerceiverSupportSettings, perceiver_equal_support_controls, _fingerprint,
)


@dataclass(frozen=True)
class KRRGeometrySettings:
    geometry_ridge: float = 0.25
    max_geometry_slope: float = 1.0
    max_geometry_delta: float = 0.5
    min_support: int = 5
    min_relative_loo_gain: float = 0.05
    min_disagreement_variation: float = 1e-6
    min_absolute_loo_gain: float = 1e-8


@dataclass(frozen=True)
class KRRAugmentedQuery:
    compound_id: str
    perceiver_pKd: float
    perceiver_krr_pKd: float
    geometry_krr_pKd: float
    geometry_delta: float
    support_count: int
    loo_krr_mse: float
    loo_geometry_mse: float
    loo_relative_gain: float
    gate: float
    full_geometry_slope: float
    geometry_active: bool


def _gram(fps, query_ids, support_ids, power):
    return np.asarray([
        [float(DataStructs.TanimotoSimilarity(fps[q],fps[s]))**power
         for s in support_ids] for q in query_ids
    ],dtype=np.float64)


def _geom_delta(K, kq, residual, difference, dq, chemical_ridge, s:KRRGeometrySettings):
    """Ridge-residualized geometry, with a separate intercept via centering."""
    n=len(residual)
    mean_diff=float(np.mean(difference))
    centered=difference-mean_diff
    spread=float(centered @ centered)
    if spread<s.min_disagreement_variation:
        return 0.0,0.0
    A=K
    inv_r=np.linalg.solve(A,residual)
    inv_d=np.linalg.solve(A,centered)
    Mr=residual-(A-np.eye(n)*chemical_ridge) @ inv_r
    Md=centered-(A-np.eye(n)*chemical_ridge) @ inv_d
    denominator=float(centered @ Md)+s.geometry_ridge
    theta=float(np.clip((centered @ Mr)/denominator,
                        -s.max_geometry_slope,s.max_geometry_slope))
    raw_q=float(dq-mean_diff)
    # Do not invent arbitrarily large extrapolated geometry effects.
    raw_q=float(np.clip(raw_q, -np.max(abs(centered)), np.max(abs(centered))))
    delta=float(np.clip(theta*(raw_q-float(kq @ inv_d)),
                        -s.max_geometry_delta,s.max_geometry_delta))
    return delta,theta


def compare_krr_geometry_with_support_validation(
    episode: PredictionEpisode,
    frozen_perceiver_pKd: Mapping[str,float],
    frozen_geometry_pKd: Mapping[str,float] | None,
    geometry_available: bool,
    *,
    control_settings: PerceiverSupportSettings = PerceiverSupportSettings(),
    geometry_settings: KRRGeometrySettings = KRRGeometrySettings(),
) -> tuple[KRRAugmentedQuery,...]:
    """Compare Perceiver-KRR vs geometry-augmented KRR, same K support labels."""
    CandidateEpisodeStore.validate_prediction_input(episode)
    if not isinstance(geometry_available,bool):
        raise TypeError("Geometry availability must be task-level bool")
    c=control_settings
    s=geometry_settings
    if c.tanimoto_power!=4:
        raise ValueError("Protocol requires Morgan Tanimoto exponent four")
    if any(not math.isfinite(v) or v<=0 for v in (
        s.geometry_ridge,s.max_geometry_slope,s.max_geometry_delta,
        s.min_disagreement_variation,s.min_absolute_loo_gain
    )):
        raise ValueError("Invalid positive geometry stability parameter")
    if not 0<s.min_relative_loo_gain<1 or not isinstance(s.min_support,int) or s.min_support<3:
        raise ValueError("Invalid LOO guard or minimum support")
    original=perceiver_equal_support_controls(episode,frozen_perceiver_pKd,c)
    all_ids={x.compound_id for x in (*episode.support,*episode.queries)}
    if geometry_available:
        if frozen_geometry_pKd is None or set(frozen_geometry_pKd)!=all_ids:
            raise ValueError("Geometry predictor must cover exact support/query compound IDs")
        g={cid:float(frozen_geometry_pKd[cid]) for cid in all_ids}
        if any(not math.isfinite(v) for v in g.values()):
            raise ValueError("Nonfinite frozen geometry prediction")
    else:
        if frozen_geometry_pKd is not None:
            raise ValueError("Absent geometry must have no geometry prediction cache")
        g=None
    n=len(episode.support)
    if g is None or n<s.min_support:
        return tuple(KRRAugmentedQuery(
            q.compound_id,q.p0,q.kernel_ridge,q.kernel_ridge,0.0,n,
            0.0,0.0,0.0,0.0,0.0,False
        ) for q in original)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,
                                                   includeChirality=True)
    fps={}
    for obj in (*episode.support,*episode.queries):
        _,fp=_fingerprint(obj.smiles,gen)
        fps[obj.compound_id]=fp
    support_ids=[x.compound_id for x in episode.support]
    y=np.asarray([float(x.pKd)-float(frozen_perceiver_pKd[x.compound_id])
                  for x in episode.support],dtype=np.float64)
    diff=np.asarray([g[cid]-float(frozen_perceiver_pKd[cid])
                     for cid in support_ids],dtype=np.float64)
    K=_gram(fps,support_ids,support_ids,c.tanimoto_power)
    ridge=float(c.lambda_kernel)
    # Fixed chemical ridge in both the original Perceiver control and
    # geometry augmented fits; do NOT optimize it on current query labels.
    loo_base=[]
    loo_full=[]
    for i in range(n):
        other=np.asarray([j for j in range(n) if j!=i],dtype=np.int64)
        A=K[np.ix_(other,other)]+ridge*np.eye(n-1)
        w=K[i,other]
        base=float(np.clip(w@np.linalg.solve(A,y[other]),
                           -c.residual_cap,c.residual_cap))
        geom,_=_geom_delta(A,w,y[other],diff[other],diff[i],ridge,s)
        loo_base.append((y[i]-base)**2)
        loo_full.append((y[i]-base-geom)**2)
    mse_base=float(np.mean(loo_base))
    mse_full=float(np.mean(loo_full))
    gain=(mse_base-mse_full)/(mse_base+1e-12)
    if (not math.isfinite(gain) or gain<s.min_relative_loo_gain or
        mse_base-mse_full<s.min_absolute_loo_gain):
        gate=0.0
    else:
        gate=float(np.clip(gain,0.0,1.0))
    A=K+ridge*np.eye(n)
    output=[]
    for q in original:
        geometry_delta=0.0
        slope=0.0
        if gate>0:
            qid=q.compound_id
            weights=_gram(fps,[qid],support_ids,c.tanimoto_power)[0]
            query_difference=g[qid]-float(frozen_perceiver_pKd[qid])
            raw,slope=_geom_delta(A,weights,y,diff,query_difference,ridge,s)
            geometry_delta=float(np.clip(gate*raw,
                                         -s.max_geometry_delta,s.max_geometry_delta))
        output.append(KRRAugmentedQuery(
            q.compound_id,q.p0,q.kernel_ridge,
            q.kernel_ridge+geometry_delta,geometry_delta,n,
            mse_base,mse_full,gain,gate,slope,gate>0 and slope!=0,
        ))
    return tuple(output)
