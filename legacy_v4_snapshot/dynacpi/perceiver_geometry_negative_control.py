"""Deterministic compound-disagreement negative control for Perceiver-anchored K-shot CPI.

Shuffles only the FROZEN geometry-minus-Perceiver prediction residual within
one (task, assay) episode. Never swaps support labels or compound IDs.
This diagnoses chemical alignment of the challenger disagreement signal;
it is NOT a substitute for genuine geometry-feature ablation, independent
protein family split review, or a scientific benchmark.

Inputs are in-memory frozen predictions and PredictionEpisode metadata only.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Mapping

from dynacpi.episode_data import CandidateEpisodeStore, PredictionEpisode
from dynacpi.perceiver_disagreement_fewshot import (
    GeometryDisagreementSettings,
    FairFewShotPrediction,
    compare_perceiver_fewshot,
)
from dynacpi.perceiver_equal_support_controls import PerceiverSupportSettings


@dataclass(frozen=True)
class NegativeControlComparison:
    original: tuple[FairFewShotPrediction, ...]
    shuffled: tuple[FairFewShotPrediction, ...]
    no_geometry: tuple[FairFewShotPrediction, ...]
    shuffle_seed: int
    control_name: str = "within_assay_compound_disagreement_derangement"


def shuffle_compound_disagreement(
    episode: PredictionEpisode,
    frozen_perceiver_pKd: Mapping[str, float],
    frozen_geometry_pKd: Mapping[str, float],
    *,
    shuffle_seed: int = 2601,
) -> dict[str, float]:
    """A deterministic cyclic derangement, not a change to any assay value.

    Keep the multiset of geometry - Perceiver disagreement unchanged while
    assigning every disagreement to a DIFFERENT compound. The order depends
    only on task/assay/compound IDs and the public prespecified seed.
    """
    CandidateEpisodeStore.validate_prediction_input(episode)
    if not isinstance(shuffle_seed, int) or isinstance(shuffle_seed, bool) or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    identifiers=[x.compound_id for x in (*episode.support, *episode.queries)]
    if len(identifiers)<2:
        raise ValueError("Need at least two distinct compounds for derangement")
    ids=set(identifiers)
    if set(frozen_perceiver_pKd)!=ids or set(frozen_geometry_pKd)!=ids:
        raise ValueError("Frozen cache IDs must match exact episode compound set")
    p={cid:float(frozen_perceiver_pKd[cid]) for cid in ids}
    g={cid:float(frozen_geometry_pKd[cid]) for cid in ids}
    if not all(isfinite(x) for x in (*p.values(),*g.values())):
        raise ValueError("Nonfinite frozen model prediction")
    ordering=sorted(ids,key=lambda cid:sha256(
        f"geometry_negative_v1|{shuffle_seed}|{episode.task_id}|{episode.assay_key}|{cid}"
        .encode("utf-8")).hexdigest())
    diffs={cid:g[cid]-p[cid] for cid in ids}
    # Rolling once guarantees no original compound receives its own residual.
    return {cid: p[cid]+diffs[ordering[(j+1)%len(ordering)]]
            for j,cid in enumerate(ordering)}


def compare_geometry_negative_control(
    episode: PredictionEpisode,
    frozen_perceiver_pKd: Mapping[str, float],
    frozen_geometry_pKd: Mapping[str, float] | None,
    geometry_available: bool,
    *,
    shuffle_seed: int = 2601,
    control_settings: PerceiverSupportSettings = PerceiverSupportSettings(),
    geometry_settings: GeometryDisagreementSettings = GeometryDisagreementSettings(),
) -> NegativeControlComparison:
    """Evaluate original, shuffled and no-geometry on the SAME support panel.

    Only the challenger disagreement assignment changes; support labels,
    frozen Perceiver values and chemical fingerprints are identical.
    """
    if not isinstance(shuffle_seed,int) or isinstance(shuffle_seed,bool) or shuffle_seed<0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    original=compare_perceiver_fewshot(
        episode,frozen_perceiver_pKd,frozen_geometry_pKd,geometry_available,
        control_settings=control_settings,geometry_settings=geometry_settings)
    no_geometry=compare_perceiver_fewshot(
        episode,frozen_perceiver_pKd,None,False,
        control_settings=control_settings,geometry_settings=geometry_settings)
    if not geometry_available:
        return NegativeControlComparison(
            original=original,shuffled=original,no_geometry=no_geometry,
            shuffle_seed=shuffle_seed)
    fake_geom=shuffle_compound_disagreement(
        episode,frozen_perceiver_pKd,frozen_geometry_pKd,shuffle_seed=shuffle_seed)
    shuffled=compare_perceiver_fewshot(
        episode,frozen_perceiver_pKd,fake_geom,True,
        control_settings=control_settings,geometry_settings=geometry_settings)
    for a,b in zip(original,shuffled):
        if a.compound_id!=b.compound_id:
            raise AssertionError("Negative control changed query compound ID")
        if (a.p0,a.p5_global,a.p5_local,a.p5_kernel_ridge)!=(b.p0,b.p5_global,b.p5_local,b.p5_kernel_ridge):
            raise AssertionError("Negative control changed Perceiver support baseline")
    return NegativeControlComparison(
        original=original,shuffled=shuffled,no_geometry=no_geometry,
        shuffle_seed=shuffle_seed)
