"""Factorial adapter smoke with real RDKit chemistry, SYNTHETIC labels/predictions.

The BindingDB pre-split task identity and SMILES are read, but all observed
affinity labels are discarded and replaced by synthetic values. This is NOT
a model evaluation, an approved assay, or a scientific training experiment.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from dynacpi.episode_data import CandidateEpisodeStore
from dynacpi.frozen_adapter import FrozenMeans,AdapterSettings,adapt_episode

store=CandidateEpisodeStore(root/"data/audit")
task_ids=store.task_ids()
counts={"tasks_tested":0,"queries_tested":0,"K0_checks":0,
        "K5_checks":0,"missing_geometry_checks":0,"permutation_checks":0,
        "synthetic_query_label_invariance":0}

def deterministic(cid,offset):
    v=int.from_bytes(hashlib.sha256((cid+"|"+str(offset)).encode()).digest()[:6],"big")/2**48
    return v

for task_id in task_ids[:12]:
    raw=store.prediction_episode(task_id,k=5)
    syn_support=tuple(replace(s,pKd=5.0+2*deterministic(s.compound_id,1))
                      for s in raw.support)
    ep=replace(raw,support=syn_support)
    full={i.compound_id for i in (*ep.support,*ep.queries)}
    means={cid:FrozenMeans(
          sequence_pKd=5.7+deterministic(cid,2),
          geometry_pKd=5.7+deterministic(cid,2)+0.65*(deterministic(cid,3)-0.5),
          geometry_available=True) for cid in full}
    p=adapt_episode(ep,means)
    assert len(p)==10
    for o in p:
        assert abs(o.D-o.B-o.differential)<1e-11
        assert abs(o.D-o.C-o.offset)<1e-11
        assert abs(o.D-o.A)<=3+1e-12
        assert 0<=o.W<=5.0
        assert 0<=o.support_used==5
        assert math.isfinite(o.V)
    counts["K5_checks"]+=1
    counts["queries_tested"]+=len(p)
    flipped=replace(ep,support=tuple(reversed(ep.support)))
    q=adapt_episode(flipped,means)
    for a,b in zip(p,q):
        assert abs(a.D-b.D)<1e-11
        assert abs(a.B-b.B)<1e-11
    counts["permutation_checks"]+=1

    ep0=replace(ep,support=tuple())
    means0={q.compound_id:means[q.compound_id] for q in ep0.queries}
    z=adapt_episode(ep0,means0)
    assert all(o.A==o.B==o.C==o.D and o.W==0 for o in z)
    counts["K0_checks"]+=1
    no_geometry={cid:FrozenMeans(sequence_pKd=means[cid].sequence_pKd,
                                 geometry_pKd=means[cid].sequence_pKd,
                                 geometry_available=False) for cid in full}
    ng=adapt_episode(ep,no_geometry)
    assert all(o.C==o.A and o.D==o.B and o.theta==0 and o.differential==0 for o in ng)
    counts["missing_geometry_checks"]+=1

    # Query labels live ONLY in EvaluationLabels and must have no impact.
    ref=store.query_label_reference(task_id)
    key=(task_id,ref.query_ids[0])
    before=adapt_episode(ep,means)
    val=store._measurements[key]
    store._measurements[key]=(val[0],val[1]+123)
    after=adapt_episode(ep,means)
    store._measurements[key]=val
    assert before==after
    counts["synthetic_query_label_invariance"]+=1

    counts["tasks_tested"]+=1

# Negative tests: missing structure cannot pass a modified geometry estimate;
# and the cache cannot include an unknown query ID.
bad=dict(means)
cid=next(iter(bad))
bad[cid]=FrozenMeans(sequence_pKd=6,geometry_pKd=8,geometry_available=False)
try:
    adapt_episode(ep,bad)
except ValueError:pass
else:raise AssertionError("Mixed or invalid geometry cache passed")
try:
    adapt_episode(ep,{**means,"unexpected_compound":FrozenMeans(5,5,True)})
except ValueError as exc:
    assert "exact episode" in str(exc)
else:raise AssertionError("Cache with unrequested compound accepted")

report={"status":"PASS_ENGINEERING_ONLY",
        "data":"Real pre-split BindingDB SMILES and task IDs, SYNTHETIC values for ALL support labels and frozen model predictions",
        "counts":counts,
        "checks":["factorial A/B/C/D identity","Tanimoto^4 kernel bounds","permutation invariance",
                  "K0 exact fallback","missing 3D differential zero","separate query-label interface",
                  "invalid geometry/cache rejection"],
        "warning":"No experimental BindingDB labels were used for adaptation; no model accuracy comparison."}
dest=root/"data/audit/frozen_adapter_real_chemistry_smoke.json"
dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
