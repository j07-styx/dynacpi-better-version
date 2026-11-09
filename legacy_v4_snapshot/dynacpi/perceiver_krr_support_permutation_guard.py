"""Support-only permutation null guard for Perceiver-KRR geometry augmentation.

This is a proposed inference-only experiment, not a scientifically validated
predictor. Query labels never enter this module. Frozen geometry must be from
an independently prefit model, and actual data use requires separate primary
source/cohort approval. The permutation diagnostic only tests support-geometry
alignment conditional on this assay; exchangeability may fail for real
correlated chemistry and must NOT be treated as an exact clinical p-value.

Hypothesis: geometry correction is enabled only when BOTH the existing
leave-one-support-out improvement gate and a prespecified support-label
permutation comparison are favorable. All null variants use exactly the same
support labels, frozen Perceiver predictions and chemical KRR kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import random
from typing import Mapping

import numpy as np
from rdkit.Chem import rdFingerprintGenerator

from dynacpi.episode_data import PredictionEpisode, CandidateEpisodeStore
from dynacpi.perceiver_equal_support_controls import PerceiverSupportSettings, _fingerprint
from dynacpi.perceiver_krr_geometry_guard import (
    KRRGeometrySettings, KRRAugmentedQuery,
    compare_krr_geometry_with_support_validation, _geom_delta, _gram,
)


@dataclass(frozen=True)
class PermutationGuardSettings:
    shuffle_seed: int = 2611
    n_null_permutations: int = 49
    max_pvalue: float = 0.05


@dataclass(frozen=True)
class PermutationGuardQuery:
    compound_id: str
    perceiver_krr_pKd: float
    unguarded_geometry_pKd: float
    permutation_guarded_pKd: float
    applied_geometry_delta: float
    original_loo_gate: float
    observed_loo_mse_gain: float
    null_pvalue: float | None
    null_count_at_least_as_good: int
    n_null_permutations_used: int
    support_count: int
    permutation_gate_passed: bool


def _settings_valid(settings: PermutationGuardSettings):
    if (
        not isinstance(settings.shuffle_seed,int)
        or isinstance(settings.shuffle_seed,bool)
        or settings.shuffle_seed<0
        or not isinstance(settings.n_null_permutations,int)
        or isinstance(settings.n_null_permutations,bool)
        or not 39<=settings.n_null_permutations<=199
        or not math.isfinite(settings.max_pvalue)
        or not 0<settings.max_pvalue<1
        or settings.max_pvalue<1/(settings.n_null_permutations+1)
    ):
        raise ValueError("Invalid prespecified support permutation parameters")


def _support_null_permutations(n:int,seed:int,count:int):
    """Deterministic, unique nonidentity permutations of disagreement values.

    A regular permutation (rather than strict derangement) is preferable for
    a uniform Monte Carlo permutation null. The identity is excluded because
    the observed sample is included explicitly in the add-one numerator.
    """
    # Exhaust the finite null when the requested Monte Carlo budget is larger.
    # In particular K=3/4 and K=5 with a budget of 199 remain valid.
    count=min(count,math.factorial(n)-1)
    rng=random.Random(seed)
    observed=tuple(range(n))
    shuffles=[]
    seen=set()
    attempts=0
    while len(shuffles)<count:
        attempts+=1
        if attempts>100000:
            raise ValueError("Could not sample enough distinct support permutations")
        seq=list(observed)
        rng.shuffle(seq)
        candidate=tuple(seq)
        if candidate==observed or candidate in seen:
            continue
        seen.add(candidate)
        shuffles.append(candidate)
    return tuple(shuffles)


def _permuted_support_mse(
    chemical_k:np.ndarray,
    residual:np.ndarray,
    disagreement:np.ndarray,
    *,
    chemical_ridge:float,
    cap:float,
    geometry_settings:KRRGeometrySettings,
):
    n=len(residual)
    errors=[]
    for i in range(n):
        ix=np.asarray([j for j in range(n) if j!=i],dtype=np.int64)
        A=chemical_k[np.ix_(ix,ix)]+chemical_ridge*np.eye(n-1)
        k_i=chemical_k[i,ix]
        raw=float(k_i@np.linalg.solve(A,residual[ix]))
        base=float(np.clip(raw,-cap,cap))
        geo,_=_geom_delta(A,k_i,residual[ix],disagreement[ix],
                          disagreement[i],chemical_ridge,geometry_settings)
        errors.append((residual[i]-base-geo)**2)
    return float(np.mean(errors))


def _permuted_support_mses(
    chemical_k, residual, disagreements, *,
    chemical_ridge, cap, geometry_settings,
):
    """Evaluate all null columns with shared LOO linear systems.

    Same centering, clipping and ridge algebra as _permuted_support_mse.
    Only support-sized matrices are allocated; no query labels are accepted.
    """
    d=np.asarray(disagreements,dtype=np.float64)
    n=len(residual)
    if d.ndim!=2 or d.shape[0]!=n or d.shape[1]==0:
        raise ValueError("Expected nonempty support-by-permutation matrix")
    errors=np.zeros(d.shape[1],dtype=np.float64)
    s=geometry_settings
    for i in range(n):
        ix=np.asarray([j for j in range(n) if j!=i],dtype=np.int64)
        kernel=chemical_k[np.ix_(ix,ix)]
        A=kernel+chemical_ridge*np.eye(n-1)
        means=d[ix].mean(axis=0)
        centered=d[ix]-means
        # One multiple-RHS solve per fold instead of three per permutation.
        solved=np.linalg.solve(A,np.column_stack([residual[ix],centered]))
        inv_r,inv_d=solved[:,0],solved[:,1:]
        mr=residual[ix]-kernel@inv_r
        md=centered-kernel@inv_d
        spread=np.sum(centered*centered,axis=0)
        denominator=np.sum(centered*md,axis=0)+s.geometry_ridge
        theta=np.clip(centered.T@mr/denominator,-s.max_geometry_slope,s.max_geometry_slope)
        extent=np.max(np.abs(centered),axis=0)
        raw=np.clip(d[i]-means,-extent,extent)
        geo=np.clip(theta*(raw-chemical_k[i,ix]@inv_d),-s.max_geometry_delta,s.max_geometry_delta)
        geo=np.where(spread<s.min_disagreement_variation,0.0,geo)
        base=float(np.clip(chemical_k[i,ix]@inv_r,-cap,cap))
        errors+=(residual[i]-base-geo)**2
    return errors/n


def compare_krr_with_support_permutation_guard(
    episode: PredictionEpisode,
    frozen_perceiver_pKd: Mapping[str,float],
    frozen_geometry_pKd: Mapping[str,float] | None,
    geometry_available: bool,
    *,
    control_settings:PerceiverSupportSettings=PerceiverSupportSettings(),
    geometry_settings:KRRGeometrySettings=KRRGeometrySettings(),
    permutation_settings:PermutationGuardSettings=PermutationGuardSettings(),
) -> tuple[PermutationGuardQuery,...]:
    """A stronger K-shot geometry challenge: Perceiver-KRR exact fallback.

    Crucially, permutations are generated using ONLY task ID, assay ID,
    support compound hashes and prespecified seed. There is no query label,
    observed improvement, prospective split role, or TEST outcome in the RNG.
    """
    _settings_valid(permutation_settings)
    CandidateEpisodeStore.validate_prediction_input(episode)
    original=compare_krr_geometry_with_support_validation(
        episode,frozen_perceiver_pKd,frozen_geometry_pKd,
        geometry_available,control_settings=control_settings,
        geometry_settings=geometry_settings)
    if not original:
        raise ValueError("No query predictions")
    n=len(episode.support)
    original_gate=original[0].gate
    # If the original LOO improvement threshold fails, or 3D is absent,
    # no permutation work is needed; fall back exactly to strongest KRR.
    if original_gate<=0 or not geometry_available:
        return tuple(PermutationGuardQuery(
            compound_id=q.compound_id,perceiver_krr_pKd=q.perceiver_krr_pKd,
            unguarded_geometry_pKd=q.geometry_krr_pKd,
            permutation_guarded_pKd=q.perceiver_krr_pKd,
            applied_geometry_delta=0.0,original_loo_gate=q.gate,
            observed_loo_mse_gain=q.loo_krr_mse-q.loo_geometry_mse,
            null_pvalue=None,null_count_at_least_as_good=0,
            n_null_permutations_used=0,support_count=n,
            permutation_gate_passed=False,
        ) for q in original)

    # Order-invariant support canonicalization makes the null depend on IDs
    # rather than arbitrary input row order or label values.
    ordered=sorted(episode.support,key=lambda x:x.compound_id)
    support_ids=[x.compound_id for x in ordered]
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,
                                                   includeChirality=True)
    fps={}
    for s in ordered:
        _,fps[s.compound_id]=_fingerprint(s.smiles,gen)
    k=_gram(fps,support_ids,support_ids,control_settings.tanimoto_power)
    r=np.asarray([float(s.pKd)-float(frozen_perceiver_pKd[s.compound_id])
                  for s in ordered],dtype=np.float64)
    d=np.asarray([float(frozen_geometry_pKd[cid])-float(frozen_perceiver_pKd[cid])
                  for cid in support_ids],dtype=np.float64)
    actual=float(original[0].loo_krr_mse-original[0].loo_geometry_mse)
    # Use a digest of public, label-independent fields, not pKd values.
    null_seed=int.from_bytes(sha256((
        f"support_permutation_v1|{permutation_settings.shuffle_seed}|"
        f"{episode.task_id}|{episode.assay_key}|"+",".join(support_ids)
    ).encode("utf-8")).digest()[:8],"big")
    permutations=_support_null_permutations(
        n,null_seed,permutation_settings.n_null_permutations)
    null_mses=_permuted_support_mses(
        k,r,d[np.asarray(permutations,dtype=np.int64)].T,
        chemical_ridge=float(control_settings.lambda_kernel),
        cap=float(control_settings.residual_cap),
        geometry_settings=geometry_settings)
    null_gains=original[0].loo_krr_mse-null_mses
    extreme=int(np.sum(null_gains>=actual-1e-12))
    pvalue=(extreme+1)/(len(permutations)+1)
    accepted=pvalue<=permutation_settings.max_pvalue
    return tuple(PermutationGuardQuery(
        compound_id=q.compound_id,perceiver_krr_pKd=q.perceiver_krr_pKd,
        unguarded_geometry_pKd=q.geometry_krr_pKd,
        permutation_guarded_pKd=(
            q.geometry_krr_pKd if accepted else q.perceiver_krr_pKd),
        applied_geometry_delta=(q.geometry_delta if accepted else 0.0),
        original_loo_gate=q.gate,
        observed_loo_mse_gain=actual,null_pvalue=pvalue,
        null_count_at_least_as_good=extreme,
        n_null_permutations_used=len(permutations),support_count=n,
        permutation_gate_passed=accepted,
    ) for q in original)
