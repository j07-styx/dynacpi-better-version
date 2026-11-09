"""Identity and support-label isolation tests over real pre-split BindingDB manifests."""
from dataclasses import replace
from pathlib import Path
import json
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from dynacpi.episode_data import CandidateEpisodeStore, SupportObservation

store=CandidateEpisodeStore(root/"data/audit")
totals={"task_count":0,"episodes_checked":0,"K20_tasks":0,
        "support_query_overlaps":0,"cross_assay_support":0,
        "query_labels_exposed_to_prediction_input":0}
for task in store.task_ids():
    totals["task_count"]+=1
    e0=store.prediction_episode(task,k=0)
    e5=store.prediction_episode(task,k=5)
    assert e0.support == tuple()
    assert e0.queries == e5.queries
    assert e5.task_id==task
    assert len(e5.support)==5 and len(e5.queries)==10
    assert not any(hasattr(q,"pKd") for q in e5.queries)
    assert not hasattr(e5,"query_pKd")
    assert all(p.assay_key==e5.assay_key for p in e5.support)
    assert len({p.compound_id for p in e5.support})==5
    assert not ({p.compound_id for p in e5.support}&{q.compound_id for q in e5.queries})
    store.validate_prediction_input(e5)
    reference=store.query_label_reference(task)
    assert reference.task_id==task
    assert reference.query_ids==tuple(q.compound_id for q in e5.queries)
    assert len(reference.pKd_values)==10
    for draw in (1,2):
        e=store.prediction_episode(task,k=5,draw=draw)
        assert [q.compound_id for q in e.queries]==list(reference.query_ids)
        assert [p.compound_id for p in e.support]==[
            p.compound_id for p in store.prediction_episode(task,k=5,draw=draw).support]
        store.validate_prediction_input(e)
        totals["episodes_checked"]+=1
    totals["episodes_checked"]+=2
    try:
        k20=store.prediction_episode(task,k=20)
        assert len(k20.support)==20
        store.validate_prediction_input(k20)
        totals["K20_tasks"]+=1
        totals["episodes_checked"]+=1
    except ValueError as exc:
        if "Insufficient support" not in str(exc):
            raise

# Negative controls prove input validator rejects unsafe episodes.
t=store.task_ids()[0]
episode=store.prediction_episode(t)
s0=episode.support[0]
try:
    store.validate_prediction_input(replace(episode,support=(replace(s0,assay_key="other_assay"),)+episode.support[1:]))
except ValueError as exc:
    assert "Cross-assay" in str(exc)
else:
    raise AssertionError("Failed to reject cross-assay support")
try:
    qry=episode.queries[0]
    store.validate_prediction_input(replace(episode,support=(replace(s0,compound_id=qry.compound_id),)+episode.support[1:]))
except ValueError as exc:
    assert "Support/query" in str(exc)
else:
    raise AssertionError("Failed to reject query support identity reuse")

# Regression test: query label changes cannot alter any prediction-facing field.
qid=episode.queries[0].compound_id
key=(t,qid)
orig=store._measurements[key]
before=store.prediction_episode(t,5,1)
store._measurements[key]=(orig[0],orig[1]+7.0)
after=store.prediction_episode(t,5,1)
assert before==after
assert store.query_label_reference(t).pKd_values[0]==orig[1]+7.0
store._measurements[key]=orig

report={"status":"PASS","scope":"Real pre-split BindingDB candidate identities, no scientific split or model training",
        "counts":totals,
        "negative_controls":["reject cross-assay support","reject reused query compound",
                             "query-label perturbation cannot affect prediction inputs",
                             "query labels available only in evaluation reference"],
        "warning":"A software boundary does not substitute for final provenance, splitting or scientific approval."}
(root/"data/audit/bindingdb_full_kd_episode_loader_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
