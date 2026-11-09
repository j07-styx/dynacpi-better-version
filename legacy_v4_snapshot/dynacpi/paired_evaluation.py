"""Paired protein-cluster-balanced regression evaluation (generic).

Predeclared primary metric:
- RMSE across queries for each (task, seed, support draw);
- average episode RMSE per task, then per independent protein component;
- macro-average component errors, ensuring giant components do not dominate;
- paired bootstrap resampling components, preserving all tasks/seeds/draws.

A single record contains paired predictions, so models necessarily share
the same exact query, label, seed, support draw and protein component.
This module is evaluator-only. Never pass y labels to the prediction model.
"""
from collections import defaultdict
from math import sqrt,isfinite
import random
import statistics

REQUIRED=("task_id","construct_id","component_id","seed","draw","query_id","y","baseline","candidate")

def _quantile(values,p):
    a=sorted(values)
    pos=(len(a)-1)*p
    low=int(pos);high=min(low+1,len(a)-1)
    return a[low]*(high-pos)+a[high]*(pos-low)

def evaluate_paired(records,expected_queries=10,bootstrap_draws=2000,bootstrap_seed=1105):
    if expected_queries<1 or bootstrap_draws<100:
        raise ValueError("Insufficient query or bootstrap setting")
    groups=defaultdict(list)
    component_for_task={}
    construct_for_task={}
    component_for_construct={}
    truth={}
    query_sets=defaultdict(set)
    seen=set()
    n=0
    for r in records:
        for name in REQUIRED:
            if name not in r:raise ValueError("Missing field "+name)
        task,construct,comp=str(r["task_id"]),str(r["construct_id"]),str(r["component_id"])
        if not task or not comp or not construct:raise ValueError("Empty task, construct or component")
        if task in construct_for_task and construct_for_task[task]!=construct:
            raise ValueError("Task appeared in multiple protein constructs")
        if construct in component_for_construct and component_for_construct[construct]!=comp:
            raise ValueError("Construct appeared in multiple protein components")
        construct_for_task[task]=construct
        component_for_construct[construct]=comp
        seed,draw,query=int(r["seed"]),int(r["draw"]),str(r["query_id"])
        if seed<0 or draw<0 or not query:raise ValueError("Invalid episode identifier")
        key=(task,seed,draw,query)
        if key in seen:raise ValueError("Duplicate paired query row")
        seen.add(key)
        if task in component_for_task and component_for_task[task]!=comp:
            raise ValueError("Task appeared in multiple protein components")
        component_for_task[task]=comp
        y,b,c=[float(r[k]) for k in ("y","baseline","candidate")]
        if not all(map(isfinite,(y,b,c))):
            raise ValueError("Non-finite observed/predicted value")
        refkey=(task,query)
        if refkey in truth and abs(truth[refkey]-y)>1e-10:
            raise ValueError("A query label changed between paired episodes")
        truth[refkey]=y
        groups[(task,seed,draw)].append((query,y,b,c))
        query_sets[task].add(query)
        n+=1
    if not n:raise ValueError("No paired queries")
    episodes_by_task=defaultdict(set)
    for task,seed,draw in groups:
        episodes_by_task[task].add((seed,draw))
    grids=list(episodes_by_task.values())
    if any(g != grids[0] for g in grids[1:]):
        raise ValueError("Tasks do not share the same fixed seed/support-draw grid")
    by_task=defaultdict(list)
    for (task,seed,draw),items in groups.items():
        if len(items)!=expected_queries or len({i[0] for i in items})!=expected_queries:
            raise ValueError("Episode must contain exactly the fixed query count")
        if {i[0] for i in items}!=query_sets[task]:
            raise ValueError("Support draws do not use matching fixed queries")
        eb=sqrt(sum((y-b)**2 for _,y,b,c in items)/len(items))
        ec=sqrt(sum((y-c)**2 for _,y,b,c in items)/len(items))
        ab=sum(abs(y-b) for _,y,b,c in items)/len(items)
        ac=sum(abs(y-c) for _,y,b,c in items)/len(items)
        by_task[task].append((eb,ec,ab,ac))
    by_construct=defaultdict(list)
    for task,episodes in by_task.items():
        task_vector=tuple(statistics.mean(ep[k] for ep in episodes) for k in range(4))
        by_construct[construct_for_task[task]].append(task_vector)
    by_component=defaultdict(list)
    for construct,task_vectors in by_construct.items():
        construct_vector=tuple(statistics.mean(t[k] for t in task_vectors) for k in range(4))
        by_component[component_for_construct[construct]].append(construct_vector)
    component_scores={
        comp:tuple(statistics.mean(construct[k] for construct in constructs) for k in range(4))
        for comp,constructs in by_component.items()}
    if len(component_scores)<5:
        raise ValueError("Too few independent protein components for a cluster-bootstrap estimate")
    values=list(component_scores.values())
    rmse_b=statistics.mean(v[0] for v in values)
    rmse_c=statistics.mean(v[1] for v in values)
    mae_b=statistics.mean(v[2] for v in values)
    mae_c=statistics.mean(v[3] for v in values)
    if rmse_b<=0:raise ValueError("Baseline RMSE must be positive for relative gain")
    rng=random.Random(bootstrap_seed)
    gains=[];rel=[]
    for _ in range(bootstrap_draws):
        sample=[values[rng.randrange(len(values))] for j in range(len(values))]
        b=statistics.mean(v[0] for v in sample)
        c=statistics.mean(v[1] for v in sample)
        gains.append(b-c)
        rel.append((b-c)/b*100)
    baseline_minus_candidate=rmse_b-rmse_c
    delta_interval=[_quantile(gains,0.025),_quantile(gains,0.975)]
    return {
       "status":"EVALUATOR_STATISTICS_ONLY",
       "n_protein_components":len(component_scores),
       "n_constructs":len(by_construct),"n_tasks":len(by_task),"n_episodes":len(groups),"n_prediction_rows":n,
       "macro_rmse_baseline":rmse_b,"macro_rmse_candidate":rmse_c,
       "macro_mae_baseline":mae_b,"macro_mae_candidate":mae_c,
       "paired_rmse_reduction":baseline_minus_candidate,
       "relative_rmse_reduction_percent":baseline_minus_candidate/rmse_b*100,
       "paired_rmse_reduction_CI95":delta_interval,
       "relative_rmse_reduction_CI95_percent":[_quantile(rel,.025),_quantile(rel,.975)],
       "CI_excludes_zero":delta_interval[0]>0 or delta_interval[1]<0,
       "bootstrap_draws":bootstrap_draws,"bootstrap_seed":bootstrap_seed,
       "aggregation":"query RMSE -> mean episodes per assay task -> mean tasks per construct -> mean constructs per protein component -> macro components",
       "warning":"Cluster-bootstrap CIs rely on independent components and an approved cohort. Do not treat a positive test on synthetic data as biological evidence."
    }
