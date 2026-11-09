"""Evaluator-only source-family-clustered seven- or ten-arm K-shot comparisons.

This module accepts already-created paired predictions and separately held
evaluation labels. It NEVER constructs episodes, trains models or reads files.
Its caller MUST pass independently reviewed original source-family membership
from a gate-approved scientific cohort before biological interpretation.

Selecting the strongest support-aware Perceiver comparator by looking at the
evaluation/test outcomes is disallowed: reference_arm is an explicit fixed
input chosen on an approved TRAIN/validation partition beforehand.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
import random
import re
import statistics

ARMS=(
    "p0", "p5_global", "p5_local", "p5_kernel_ridge",
    "d5_no_geometry", "d5_shuffled", "d5_geometry",
)
SUPPORT_CONTROLS=("p5_global","p5_local","p5_kernel_ridge")
KRR_CHALLENGERS=(
    "krr_loo_geometry","krr_permutation_geometry",
    "krr_shuffled_permutation_geometry",
)
REQUIRED=(
    "task_id","construct_id","protein_component_id","source_family_sha256",
    "assay_key","seed","draw","query_id","support_panel_sha256",
    "support_count","geometry_available","y",
)+ARMS
HASH=re.compile(r"^[0-9a-f]{64}$")


def _mean(values):
    return statistics.mean(values)


def _interpolate_quantile(seq,proportion):
    arr=sorted(seq)
    at=(len(arr)-1)*proportion
    i=int(at)
    j=min(i+1,len(arr)-1)
    return arr[i]*(j-at)+arr[j]*(at-i)


def _average_dicts(rows):
    if not rows:
        raise ValueError("No rows to average")
    return {k:_mean(x[k] for x in rows) for k in rows[0]}


def evaluate_source_family_fewshot(
    records,
    *,
    reference_arm: str,
    expected_queries: int=10,
    expected_support: int=5,
    bootstrap_draws: int=2000,
    bootstrap_seed: int=1109,
    min_independent_families: int=5,
    include_krr_challengers: bool=False,
):
    """Calculate paired differences on source-independent family clusters.

    Family-macro: query RMSE -> episode -> task -> construct -> protein
    component -> publication-linked SOURCE FAMILY -> macro source families.
    Pooled RMSE is reported separately (one weight per prediction row).
    The function never authorizes source-family labels or scientific approval.
    """
    if reference_arm not in SUPPORT_CONTROLS:
        raise ValueError("reference_arm must be a PRESELECTED same-support Perceiver control")
    if not isinstance(include_krr_challengers,bool):
        raise TypeError("include_krr_challengers must be an explicit bool")
    if include_krr_challengers and reference_arm!="p5_kernel_ridge":
        raise ValueError("KRR challenge requires PRESELECTED p5_kernel_ridge reference")
    arms=ARMS+(KRR_CHALLENGERS if include_krr_challengers else ())
    required=REQUIRED+(KRR_CHALLENGERS if include_krr_challengers else ())
    for name,val,least in (
        ("expected_queries",expected_queries,1),
        ("expected_support",expected_support,0),
        ("bootstrap_draws",bootstrap_draws,100),
        ("min_independent_families",min_independent_families,5),
    ):
        if not isinstance(val,int) or isinstance(val,bool) or val<least:
            raise ValueError("Invalid "+name)
    if not isinstance(bootstrap_seed,int) or isinstance(bootstrap_seed,bool) or bootstrap_seed<0:
        raise ValueError("Invalid bootstrap_seed")

    episodes=defaultdict(list)
    task_meta={}
    construct_component={}
    component_family={}
    support_panels={}
    query_truth={}
    queries_by_task=defaultdict(set)
    seen=set()
    pooled_sum_squares={a:0.0 for a in arms}
    pooled_sum_absolutes={a:0.0 for a in arms}
    total=0
    for row in records:
        if any(k not in row for k in required):
            raise ValueError("Incomplete paired source-family evaluation row")
        task,construct,component,family,assay=[
            str(row[key]) for key in (
                "task_id","construct_id","protein_component_id",
                "source_family_sha256","assay_key"
            )]
        if not all((task,construct,component,family,assay)):
            raise ValueError("Empty task/construct/source identity")
        if not HASH.fullmatch(family):
            raise ValueError("Source family must be a stable 64-hex ID")
        if not isinstance(row["geometry_available"],bool):
            raise ValueError("Geometry availability must be task-level bool")
        try:
            seed=int(row["seed"]);draw=int(row["draw"])
        except (TypeError,ValueError):
            raise ValueError("Invalid seed/draw") from None
        if seed<0 or draw<0 or isinstance(row["seed"],bool) or isinstance(row["draw"],bool):
            raise ValueError("Invalid seed/draw")
        if seed!=row["seed"] or draw!=row["draw"]:
            raise ValueError("Seed/draw must be integral")
        query=str(row["query_id"])
        if not query:
            raise ValueError("Empty query ID")
        if row["support_count"]!=expected_support or isinstance(row["support_count"],bool):
            raise ValueError("Support count does not match prespecified information budget")
        digest=row["support_panel_sha256"]
        if not isinstance(digest,str) or not HASH.fullmatch(digest):
            raise ValueError("Invalid support panel SHA256")
        identity=(task,construct,component,family,assay,row["geometry_available"])
        if task in task_meta and task_meta[task]!=identity:
            raise ValueError("Task changed assay, construct, protein, family or geometry availability")
        task_meta[task]=identity
        if construct in construct_component and construct_component[construct]!=component:
            raise ValueError("Construct spans multiple protein components")
        construct_component[construct]=component
        if component in component_family and component_family[component]!=family:
            raise ValueError("Protein component spans source families")
        component_family[component]=family
        ep=(task,seed,draw)
        if ep in support_panels and support_panels[ep]!=digest:
            raise ValueError("A paired episode used inconsistent support panels")
        support_panels[ep]=digest
        key=(task,seed,draw,query)
        if key in seen:
            raise ValueError("Duplicate paired query row")
        seen.add(key)
        y=float(row["y"])
        predictions={a:float(row[a]) for a in arms}
        if not math.isfinite(y) or any(not math.isfinite(x) for x in predictions.values()):
            raise ValueError("Nonfinite label or model prediction")
        if (task,query) in query_truth and abs(y-query_truth[(task,query)])>1e-10:
            raise ValueError("Query's evaluator-only label changed across draws")
        query_truth[(task,query)]=y
        queries_by_task[task].add(query)
        if abs(predictions["d5_no_geometry"]-predictions["p5_global"])>1e-10:
            raise ValueError("D5-no-geometry must exactly match P5-global")
        if not row["geometry_available"] and (
            abs(predictions["d5_geometry"]-predictions["p5_global"])>1e-10 or
            abs(predictions["d5_shuffled"]-predictions["p5_global"])>1e-10
        ):
            raise ValueError("Unavailable geometry must fall back exactly to P5-global")
        if include_krr_challengers:
            # All KRR challengers are bounded, opt-in refinements of exactly
            # the SAME frozen Perceiver KRR reference, never new label budgets.
            for name in KRR_CHALLENGERS:
                if abs(predictions[name]-predictions["p5_kernel_ridge"])>0.50000001:
                    raise ValueError("KRR challenger escaped predeclared geometry delta cap")
            if not row["geometry_available"] and any(
                abs(predictions[name]-predictions["p5_kernel_ridge"])>1e-10
                for name in KRR_CHALLENGERS
            ):
                raise ValueError("Unavailable geometry must fall back exactly to P5-KRR")
        episodes[ep].append((query,y,predictions))
        for a,value in predictions.items():
            pooled_sum_squares[a]+=(y-value)**2
            pooled_sum_absolutes[a]+=abs(y-value)
        total+=1
    if not total:
        raise ValueError("No paired query rows")
    grids=defaultdict(set)
    task_scores=defaultdict(list)
    for (task,seed,draw),items in episodes.items():
        if len(items)!=expected_queries or len({q for q,_,_ in items})!=expected_queries:
            raise ValueError("Every episode must have fixed distinct queries")
        if {q for q,_,_ in items}!=queries_by_task[task]:
            raise ValueError("Same task has mismatched query set across draws")
        grids[task].add((seed,draw))
        row_scores={}
        for arm in arms:
            squares=[(y-p[arm])**2 for _,y,p in items]
            row_scores[arm]=math.sqrt(_mean(squares))
        task_scores[task].append(row_scores)
    all_grids=list(grids.values())
    if any(g!=all_grids[0] for g in all_grids[1:]):
        raise ValueError("Tasks do not share the same fixed seed/support-draw grid")
    tasks_by_construct=defaultdict(list)
    for task,scores in task_scores.items():
        _,construct,_,_,_,_=task_meta[task]
        tasks_by_construct[construct].append(_average_dicts(scores))
    constructs_by_component=defaultdict(list)
    for construct,scores in tasks_by_construct.items():
        constructs_by_component[construct_component[construct]].append(_average_dicts(scores))
    components_by_family=defaultdict(list)
    for component,scores in constructs_by_component.items():
        components_by_family[component_family[component]].append(_average_dicts(scores))
    if len(components_by_family)<min_independent_families:
        raise ValueError("Not enough independent source families for family-cluster uncertainty")
    family_scores={fam:_average_dicts(scores) for fam,scores in components_by_family.items()}
    macro=_average_dicts(list(family_scores.values()))
    pooled={a:math.sqrt(pooled_sum_squares[a]/total) for a in arms}
    pooled_mae={a:pooled_sum_absolutes[a]/total for a in arms}

    family_values=list(family_scores.values())
    rng=random.Random(bootstrap_seed)
    # Use the SAME source-family resample indices for every paired contrast.
    resamples=[
        [rng.randrange(len(family_values)) for _ in range(len(family_values))]
        for _ in range(bootstrap_draws)
    ]
    comparators=(reference_arm,"d5_no_geometry","d5_shuffled")
    comparison={}
    for ref in comparators:
        differences=[row[ref]-row["d5_geometry"] for row in family_values]
        observed=_mean(differences)
        draws=[_mean(differences[j] for j in sample) for sample in resamples]
        comparison["d5_vs_"+ref]={
            "family_macro_RMSE_reduction":observed,
            "relative_reduction_percent":observed/macro[ref]*100 if macro[ref]>0 else None,
            "paired_family_bootstrap_CI95":[
                _interpolate_quantile(draws,0.025),_interpolate_quantile(draws,0.975)
            ],
            "family_count":len(family_values),
        }
    if include_krr_challengers:
        krr_ref="p5_kernel_ridge"
        krr_candidate="krr_permutation_geometry"
        for comparison_arm in (
            krr_ref,"krr_loo_geometry","krr_shuffled_permutation_geometry"
        ):
            differences=[
                row[comparison_arm]-row[krr_candidate]
                for row in family_values
            ]
            draws=[_mean(differences[j] for j in sample) for sample in resamples]
            observed=_mean(differences)
            comparison["krr_perm_vs_"+comparison_arm]={
                "family_macro_RMSE_reduction":observed,
                "relative_reduction_percent":(
                    observed/macro[comparison_arm]*100
                    if macro[comparison_arm]>0 else None
                ),
                "paired_family_bootstrap_CI95":[
                    _interpolate_quantile(draws,0.025),
                    _interpolate_quantile(draws,0.975)
                ],
                "family_count":len(family_values),
            }
    return {
        "status":"UNVERIFIED_DATA_ORIGIN_EVALUATOR_ONLY_NO_SCIENTIFIC_APPROVAL",
        "include_krr_challengers":include_krr_challengers,
        "reference_arm_preselected":reference_arm,
        "expected_support":expected_support,
        "n_source_families":len(family_values),
        "n_protein_components":len(component_family),
        "n_constructs":len(construct_component),
        "n_tasks":len(task_meta),
        "n_episodes":len(episodes),
        "n_prediction_rows":total,
        "source_family_macro_episode_RMSE":macro,
        "pooled_query_RMSE":pooled,
        "pooled_query_MAE":pooled_mae,
        "paired_comparisons":comparison,
        "bootstrap_draws":bootstrap_draws,
        "bootstrap_seed":bootstrap_seed,
        "aggregation":"query -> episode RMSE -> task -> construct -> component -> connected source family -> macro source families",
        "warning":"Input source-family provenance and assay approval are not verified by this standalone mathematics module. Synthetic tests cannot establish model superiority. No model promotion without source-family gate and preregistered held-out evaluation.",
    }
