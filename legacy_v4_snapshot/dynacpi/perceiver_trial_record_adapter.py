"""Separate evaluated-query LABEL boundary for an approved future K-shot trial.

This module ONLY combines precomputed inference-only predictions with
query labels in an evaluator-facing row format. It neither predicts nor
trains, and it cannot certify a source-reviewed cohort or protein group.

NEVER supply query labels to compare_geometry_negative_control or any
inference model. Invoke that function first; then call this adapter in an
evaluation-only scope after explicit independent cohort/split approval.
"""
from __future__ import annotations
from hashlib import sha256
import json
import math
import re

from dynacpi.episode_data import (
    CandidateEpisodeStore, EvaluationLabels, PredictionEpisode,
)
from dynacpi.perceiver_geometry_negative_control import NegativeControlComparison

HASH=re.compile(r"^[0-9a-f]{64}$")


def evaluation_rows_from_predictions(
    episode: PredictionEpisode,
    labels: EvaluationLabels,
    comparison: NegativeControlComparison,
    *,
    geometry_available: bool,
    source_family_sha256: str,
    construct_id: str,
    seed: int,
    draw: int,
) -> tuple[dict, ...]:
    """Strict output mapping: all arms share one query, panel and known label.

    The caller's source family identity MUST be verified externally against
    the independently frozen source-family graph and actual assay constructs.
    This adapter cannot authorize the confidential user data or final test.
    """
    CandidateEpisodeStore.validate_prediction_input(episode)
    if not isinstance(labels,EvaluationLabels) or labels.task_id!=episode.task_id:
        raise ValueError("Evaluator-only query labels must match the task")
    if not isinstance(comparison,NegativeControlComparison):
        raise TypeError("Expected evaluated negative-control prediction bundle")
    if not isinstance(geometry_available,bool):
        raise TypeError("Geometry availability must be a predeclared task-level bool")
    if not isinstance(source_family_sha256,str) or not HASH.fullmatch(source_family_sha256):
        raise ValueError("Unverified/malformed source-family SHA256")
    if not isinstance(construct_id,str) or not construct_id.strip():
        raise ValueError("Missing assayed construct ID")
    for name,value in (("seed",seed),("draw",draw)):
        if not isinstance(value,int) or isinstance(value,bool) or value<0:
            raise ValueError("Invalid "+name)
    queries=[q.compound_id for q in episode.queries]
    if tuple(queries)!=labels.query_ids or len(labels.pKd_values)!=len(queries):
        raise ValueError("Evaluator query labels are missing, additional, or reordered")
    if not all(math.isfinite(float(y)) for y in labels.pKd_values):
        raise ValueError("Evaluator query label is non-finite")
    if not queries:
        raise ValueError("No queries")
    if any(len(items)!=len(queries) for items in (
        comparison.original,comparison.shuffled,comparison.no_geometry
    )):
        raise ValueError("Model arms do not cover exactly the query panel")
    for arr in (comparison.original,comparison.shuffled,comparison.no_geometry):
        if tuple(x.compound_id for x in arr)!=tuple(queries):
            raise ValueError("Model arms disagree on exact ordered query IDs")
    if len({s.compound_id for s in episode.support})!=len(episode.support):
        raise ValueError("Duplicate support IDs")
    support=[(s.compound_id,format(float(s.pKd),".17g"))
             for s in sorted(episode.support,key=lambda x:x.compound_id)]
    if any(not math.isfinite(float(s.pKd)) for s in episode.support):
        raise ValueError("Invalid support label")
    panel_payload={
        "version":"perceiver_kshot_panel_v1",
        "task_id":episode.task_id,
        "assay_key":episode.assay_key,
        "support":support,
    }
    panel_sha=sha256(json.dumps(panel_payload,sort_keys=True,separators=(",",":"))
                     .encode("utf-8")).hexdigest()
    rows=[]
    for query,truth,original,shuffled,no_geom in zip(
        episode.queries, labels.pKd_values,
        comparison.original,comparison.shuffled,comparison.no_geometry
    ):
        if original.support_used!=len(support):
            raise ValueError("Original arm support budget changed")
        if not (shuffled.support_used==no_geom.support_used==original.support_used):
            raise ValueError("One model arm has a different support budget")
        for key in ("p0","p5_global","p5_local","p5_kernel_ridge"):
            values=(getattr(original,key),getattr(shuffled,key),getattr(no_geom,key))
            if any(not math.isfinite(float(v)) for v in values) or not all(
                abs(float(v)-float(values[0]))<=1e-10 for v in values
            ):
                raise ValueError("Some arms do not use identical frozen Perceiver controls")
        if abs(no_geom.d5_geometry-original.p5_global)>1e-10:
            raise ValueError("D5 no-geometry does not equal P5-global")
        predictions=(
            original.p0,original.p5_global,original.p5_local,
            original.p5_kernel_ridge,no_geom.d5_geometry,
            shuffled.d5_geometry,original.d5_geometry,
        )
        if any(not math.isfinite(float(v)) for v in predictions):
            raise ValueError("Non-finite model arm")
        rows.append({
            "task_id":episode.task_id,
            "construct_id":construct_id,
            "protein_component_id":episode.protein_component_id,
            "source_family_sha256":source_family_sha256,
            "assay_key":episode.assay_key,
            "seed":seed,"draw":draw,
            "query_id":query.compound_id,
            "support_panel_sha256":panel_sha,
            "support_count":len(support),
            "geometry_available":geometry_available,
            "y":float(truth),
            "p0":original.p0,
            "p5_global":original.p5_global,
            "p5_local":original.p5_local,
            "p5_kernel_ridge":original.p5_kernel_ridge,
            "d5_no_geometry":no_geom.d5_geometry,
            "d5_shuffled":shuffled.d5_geometry,
            "d5_geometry":original.d5_geometry,
        })
    return tuple(rows)
