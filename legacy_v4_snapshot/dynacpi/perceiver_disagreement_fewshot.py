"""Cold-target few-shot correction anchored to the ORIGINAL frozen PerceiverCPI.

This is a proposal for a *source-APPROVED* future K-shot experiment, not
authorization to train/evaluate on the unreviewed BindingDB cohort.

Input only: K same-assay support labels, compounds, cached frozen Perceiver
and separate frozen geometry-aware predictions. Query labels are not accepted.
The D5 experimental arm is explicitly P5-GLOBAL + centered geometry-
disagreement correction, with missing-geometry EXACT P5-GLOBAL fallback.

The centered regression prevents a constant assay calibration offset from
looking like useful geometry; low diversity/effective K also disables it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Tuple
import math

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from dynacpi.episode_data import CandidateEpisodeStore, PredictionEpisode
from dynacpi.perceiver_equal_support_controls import (
    PerceiverSupportSettings, perceiver_equal_support_controls,
    _fingerprint,
)


@dataclass(frozen=True)
class GeometryDisagreementSettings:
    ridge_fraction: float = 0.25
    theta_cap: float = 1.0
    delta_cap: float = 0.5
    min_support: int = 3
    min_effective_support: float = 1.5
    min_disagreement_variation: float = 1e-6


@dataclass(frozen=True)
class FairFewShotPrediction:
    compound_id: str
    p0: float
    p5_global: float
    p5_local: float
    p5_kernel_ridge: float
    d5_geometry: float
    d5_geometry_delta: float
    d5_no_geometry: float
    support_used: int
    effective_support: float
    disagreement_variation: float
    theta: float
    geometry_reliability: float
    geometry_active: bool


def compare_perceiver_fewshot(
    episode: PredictionEpisode,
    frozen_perceiver_pKd: Mapping[str, float],
    frozen_geometry_pKd: Mapping[str, float] | None,
    geometry_available: bool,
    *,
    control_settings: PerceiverSupportSettings = PerceiverSupportSettings(),
    geometry_settings: GeometryDisagreementSettings = GeometryDisagreementSettings(),
) -> Tuple[FairFewShotPrediction, ...]:
    """Make jointly aligned P0/P5/D5 predictions without exposing query labels.

    The geometry predictor is an independent, *pre-frozen* source. The caller
    has to establish checkpoint provenance and scientific cohort approval
    before interpreting any later benchmark results.
    """
    CandidateEpisodeStore.validate_prediction_input(episode)
    if not isinstance(geometry_available, bool):
        raise TypeError("geometry_available must be a task-level bool")
    z = geometry_settings
    values=(z.ridge_fraction,z.theta_cap,z.delta_cap,z.min_effective_support,
            z.min_disagreement_variation)
    if not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("All geometry regularization and safeguards must be positive finite")
    if not isinstance(z.min_support,int) or z.min_support<2:
        raise ValueError("min_support must be an integer >=2")
    if not isinstance(episode.queries, tuple) or not episode.queries:
        raise ValueError("Episode must have query compounds")

    controls=perceiver_equal_support_controls(episode,frozen_perceiver_pKd,
                                               settings=control_settings)
    all_ids={x.compound_id for x in (*episode.support,*episode.queries)}
    support=list(episode.support)
    if geometry_available:
        if frozen_geometry_pKd is None or set(frozen_geometry_pKd)!=all_ids:
            raise ValueError("Frozen geometry predictions must cover exactly the support/query IDs")
        challenger={cid:float(frozen_geometry_pKd[cid]) for cid in all_ids}
        if not all(math.isfinite(x) for x in challenger.values()):
            raise ValueError("Frozen geometry predictor has non-finite scores")
    else:
        if frozen_geometry_pKd is not None:
            raise ValueError("Unavailable geometry requires no challenger prediction cache")
        challenger=None

    # K0, missing geometry, or too few support examples: do not fit a
    # disagreement correction. The equal-support P5 control remains valid.
    if not geometry_available or len(support)<z.min_support:
        return tuple(FairFewShotPrediction(
            compound_id=c.compound_id,p0=c.p0,p5_global=c.global_offset,
            p5_local=c.local_offset,p5_kernel_ridge=c.kernel_ridge,
            d5_geometry=c.global_offset,d5_geometry_delta=0.0,
            d5_no_geometry=c.global_offset,support_used=c.support_used,
            effective_support=0.0,disagreement_variation=0.0,theta=0.0,
            geometry_reliability=0.0,geometry_active=False,
        ) for c in controls)

    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,
                                                   includeChirality=True)
    fp={}
    for obs in (*episode.support,*episode.queries):
        canonical,fp[obs.compound_id]=_fingerprint(obs.smiles,gen)
        # This check is also inside the P5 controls; redundant by design.
        if not canonical:
            raise ValueError("Empty canonical compound identity")
    supp_id=[s.compound_id for s in support]
    disp=np.asarray([challenger[c]-float(frozen_perceiver_pKd[c])
                     for c in supp_id],dtype=np.float64)
    residual=np.asarray([float(s.pKd)-float(frozen_perceiver_pKd[s.compound_id])
                         for s in support],dtype=np.float64)
    out=[]
    for c,query in zip(controls,episode.queries):
        weights=np.asarray([
            float(DataStructs.TanimotoSimilarity(fp[query.compound_id],fp[cid]))**
            control_settings.tanimoto_power
            for cid in supp_id
        ],dtype=np.float64)
        W=float(weights.sum())
        W2=float(weights @ weights)
        n_eff=W*W/W2 if W2>0 else 0.0
        V=0.0
        theta=0.0
        delta=0.0
        reliability=0.0
        active=False
        if W>0 and n_eff>=z.min_effective_support:
            dc=float(weights @ disp/W)
            rc=float(weights @ residual/W)
            centered=disp-dc
            V=float(weights @ (centered*centered))
            if V>=z.min_disagreement_variation:
                # Unlike uncentered residual regression this is insensitive
                # to a constant assay-wide calibration shift.
                centered_residual=residual-rc
                # Dimensionless shrinkage avoids a fixed absolute prior
                # overwhelming modest yet real protein-disagreement variation.
                # Small V has already been screened above; cap safeguards remain.
                theta=float(np.clip(weights @ (centered*centered_residual)/
                                    (V*(1.0+z.ridge_fraction)),-z.theta_cap,z.theta_cap))
                # Don't extrapolate disagreement beyond its observed support.
                max_dist=float(np.max(np.abs(centered)))
                dq=challenger[query.compound_id]-float(frozen_perceiver_pKd[query.compound_id])
                projected=float(np.clip(dq-dc,-max_dist,max_dist))
                # A local query explained by few effective support molecules
                # receives a smaller correction even if the fitted slope is high.
                reliability=float(np.clip(n_eff/len(support),0.0,1.0))
                delta=float(np.clip(theta*projected*reliability,
                                    -z.delta_cap,z.delta_cap))
                active=True
        out.append(FairFewShotPrediction(
            compound_id=c.compound_id,p0=c.p0,
            p5_global=c.global_offset,p5_local=c.local_offset,
            p5_kernel_ridge=c.kernel_ridge,
            d5_geometry=c.global_offset+delta,d5_geometry_delta=delta,
            d5_no_geometry=c.global_offset,support_used=c.support_used,
            effective_support=n_eff,disagreement_variation=V,theta=theta,
            geometry_reliability=reliability,geometry_active=active,
        ))
    return tuple(out)
