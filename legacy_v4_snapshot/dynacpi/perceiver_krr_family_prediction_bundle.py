"""Inference-only seven-arm extension: strong Perceiver-KRR challengers.

Produces aligned KRR LOO, KRR support-permutation, and chemically misaligned
geometry disagreement controls using the same K support observations and
frozen Perceiver scores. No query-affinity labels are accepted here.
No cohort or assay is ever loaded or approved by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

from dynacpi.episode_data import CandidateEpisodeStore,PredictionEpisode
from dynacpi.perceiver_equal_support_controls import PerceiverSupportSettings
from dynacpi.perceiver_krr_geometry_guard import (
    KRRGeometrySettings,KRRAugmentedQuery,
    compare_krr_geometry_with_support_validation,
)
from dynacpi.perceiver_krr_support_permutation_guard import (
    PermutationGuardSettings,PermutationGuardQuery,
    compare_krr_with_support_permutation_guard,
)
from dynacpi.perceiver_geometry_negative_control import (
    shuffle_compound_disagreement,
)


@dataclass(frozen=True)
class KRRFamilyPredictionBundle:
    loo:tuple[KRRAugmentedQuery,...]
    permutation:tuple[PermutationGuardQuery,...]
    shuffled_permutation:tuple[PermutationGuardQuery,...]
    geometry_available:bool


def predict_krr_family_challengers(
    episode:PredictionEpisode,
    frozen_perceiver_pKd:Mapping[str,float],
    frozen_geometry_pKd:Mapping[str,float]|None,
    geometry_available:bool,
    *,
    shuffle_seed:int=2601,
    control_settings:PerceiverSupportSettings=PerceiverSupportSettings(),
    geometry_settings:KRRGeometrySettings=KRRGeometrySettings(),
    permutation_settings:PermutationGuardSettings=PermutationGuardSettings(),
)->KRRFamilyPredictionBundle:
    """Return prediction-only bundle; no query labels or validation outcomes."""
    CandidateEpisodeStore.validate_prediction_input(episode)
    if not isinstance(geometry_available,bool):
        raise TypeError("geometry_available must be an explicit task bool")
    if not isinstance(shuffle_seed,int) or isinstance(shuffle_seed,bool) or shuffle_seed<0:
        raise ValueError("Invalid prespecified shuffle seed")
    loo=compare_krr_geometry_with_support_validation(
        episode,frozen_perceiver_pKd,frozen_geometry_pKd,geometry_available,
        control_settings=control_settings,geometry_settings=geometry_settings)
    original=compare_krr_with_support_permutation_guard(
        episode,frozen_perceiver_pKd,frozen_geometry_pKd,geometry_available,
        control_settings=control_settings,geometry_settings=geometry_settings,
        permutation_settings=permutation_settings)
    if geometry_available:
        synthetic_geometry=shuffle_compound_disagreement(
            episode,frozen_perceiver_pKd,frozen_geometry_pKd,
            shuffle_seed=shuffle_seed)
    else:
        synthetic_geometry=None
    shuffled=compare_krr_with_support_permutation_guard(
        episode,frozen_perceiver_pKd,synthetic_geometry,geometry_available,
        control_settings=control_settings,geometry_settings=geometry_settings,
        permutation_settings=permutation_settings)
    query_ids=tuple(q.compound_id for q in episode.queries)
    for rows in (loo,original,shuffled):
        if tuple(row.compound_id for row in rows)!=query_ids:
            raise AssertionError("Different query compound IDs across KRR arms")
        if any(row.support_count!=len(episode.support) for row in rows):
            raise AssertionError("KRR arms received different support budgets")
    return KRRFamilyPredictionBundle(
        loo=loo,permutation=original,
        shuffled_permutation=shuffled,geometry_available=geometry_available)
