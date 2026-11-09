"""Evaluator-facing label join for the new Perceiver KRR challenger arms.

Compute BOTH prediction bundles before obtaining EvaluationLabels. This
adapter is the first place query outcomes appear. It never trains models or
certifies a study source / assay construct. Biological evaluation requires
independently approved source provenance and a frozen new split.
"""
from __future__ import annotations
import math

from dynacpi.episode_data import EvaluationLabels,PredictionEpisode
from dynacpi.perceiver_geometry_negative_control import NegativeControlComparison
from dynacpi.perceiver_trial_record_adapter import evaluation_rows_from_predictions
from dynacpi.perceiver_krr_family_prediction_bundle import KRRFamilyPredictionBundle


def evaluation_rows_with_krr_challengers(
    episode:PredictionEpisode,
    labels:EvaluationLabels,
    standard_comparison:NegativeControlComparison,
    krr_predictions:KRRFamilyPredictionBundle,
    *,
    geometry_available:bool,
    source_family_sha256:str,
    construct_id:str,
    seed:int,
    draw:int,
)->tuple[dict,...]:
    """Join strictly aligned seven legacy arms + three stronger KRR challengers.

    NOTE: validates *internal* consistency, not external source provenance.
    The labels object must never be given to an inference-only function.
    """
    if not isinstance(krr_predictions,KRRFamilyPredictionBundle):
        raise TypeError("Expected an inference-only KRR family prediction bundle")
    if not isinstance(geometry_available,bool) or (
        krr_predictions.geometry_available is not geometry_available
    ):
        raise ValueError("Geometry availability differs between matched arms")
    base_rows=evaluation_rows_from_predictions(
        episode,labels,standard_comparison,
        geometry_available=geometry_available,
        source_family_sha256=source_family_sha256,
        construct_id=construct_id,seed=seed,draw=draw)
    all_predictions=(
        krr_predictions.loo,
        krr_predictions.permutation,
        krr_predictions.shuffled_permutation,
    )
    expected_ids=tuple(row["query_id"] for row in base_rows)
    if any(tuple(z.compound_id for z in predictions)!=expected_ids
           for predictions in all_predictions):
        raise ValueError("KRR arms must use the identical ordered query IDs")
    support_count=len(episode.support)
    for predictions in all_predictions:
        if any(z.support_count!=support_count for z in predictions):
            raise ValueError("KRR arms received a different support information budget")
    if len({z.original_loo_gate for z in krr_predictions.permutation})>1 or (
        len({z.permutation_gate_passed for z in krr_predictions.permutation})>1
    ):
        raise ValueError("KRR permutation gate must be fixed per support episode")
    if len({z.permutation_gate_passed for z in krr_predictions.shuffled_permutation})>1:
        raise ValueError("Shuffled KRR permutation gate must be fixed per episode")
    rows=[]
    for row,loo,permutation,shuffled in zip(base_rows,*all_predictions):
        krr=row["p5_kernel_ridge"]
        for z in (loo,permutation,shuffled):
            if not math.isfinite(float(z.perceiver_krr_pKd)) or (
                abs(z.perceiver_krr_pKd-krr)>1e-10
            ):
                raise ValueError("KRR arm changed the frozen equal-support Perceiver baseline")
        if (
            abs(permutation.unguarded_geometry_pKd-loo.geometry_krr_pKd)>1e-10
            or abs(permutation.original_loo_gate-loo.gate)>1e-10
        ):
            raise ValueError("KRR LOO challenger differs between matched guard arms")
        for prediction in (permutation,shuffled):
            expect=(prediction.unguarded_geometry_pKd
                    if prediction.permutation_gate_passed
                    else prediction.perceiver_krr_pKd)
            if abs(prediction.permutation_guarded_pKd-expect)>1e-10:
                raise ValueError("Permutation decision does not match the prediction")
            if abs(prediction.permutation_guarded_pKd-krr)>0.50000001:
                raise ValueError("KRR permutation arm exceeds geometry cap")
            if prediction.null_pvalue is None and prediction.permutation_gate_passed:
                raise ValueError("Cannot activate geometry without support null evidence")
            if prediction.null_pvalue is not None and (
                not math.isfinite(prediction.null_pvalue)
                or not 0< prediction.null_pvalue<=1
            ):
                raise ValueError("Invalid support permutation tail fraction")
        if any(not math.isfinite(z) for z in (
            loo.geometry_krr_pKd,permutation.permutation_guarded_pKd,
            shuffled.permutation_guarded_pKd,
        )):
            raise ValueError("Nonfinite KRR challenger prediction")
        if abs(loo.geometry_krr_pKd-krr)>0.50000001:
            raise ValueError("KRR LOO arm exceeds geometry correction cap")
        if not geometry_available and any(abs(value-krr)>1e-10 for value in (
            loo.geometry_krr_pKd,permutation.permutation_guarded_pKd,
            shuffled.permutation_guarded_pKd,
        )):
            raise ValueError("Unavailable geometry did not fall back to Perceiver-KRR")
        rows.append({
            **row,
            "krr_loo_geometry":loo.geometry_krr_pKd,
            "krr_permutation_geometry":permutation.permutation_guarded_pKd,
            "krr_shuffled_permutation_geometry":shuffled.permutation_guarded_pKd,
        })
    return tuple(rows)
