"""Leakage-guarded DynaCPI v4 support/query data boundary.

These PRE_SPLIT manifests are TECHNICAL CANDIDATES, NOT SCIENCE-APPROVED.
Do not use this loader to fit published benchmark models until the approved
construct/assay cohort, protein-group split and endpoint policy are frozen.

The prediction-facing Episode exposes support labels only. Query labels reside
exclusively inside an EvaluationLabels object created via a separate method.
Only labels from the selected assay task may enter the support list.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Dict, Tuple


@dataclass(frozen=True)
class SupportObservation:
    compound_id: str
    smiles: str
    pKd: float
    assay_key: str


@dataclass(frozen=True)
class QueryCompound:
    compound_id: str
    smiles: str


@dataclass(frozen=True)
class PredictionEpisode:
    task_id: str
    sequence_sha256: str
    protein_component_id: str
    assay_key: str
    support: Tuple[SupportObservation, ...]
    queries: Tuple[QueryCompound, ...]
    status: str = "PRE_SPLIT_UNAPPROVED"


@dataclass(frozen=True)
class EvaluationLabels:
    """Never hand this object to the model forward pass or router."""
    task_id: str
    query_ids: Tuple[str, ...]
    pKd_values: Tuple[float, ...]


class CandidateEpisodeStore:
    def __init__(self, audit_directory: Path):
        audit_directory = Path(audit_directory)
        self._tasks = {}
        self._episodes = {}
        self._measurements = {}
        for ln in (audit_directory / "bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl").open(encoding="utf-8"):
            entry = json.loads(ln)
            if entry["task_id"] in self._tasks:
                raise ValueError("Duplicate task_id")
            if entry["status"] != "PRE_SPLIT_TECHNICAL_CANDIDATE_NOT_ASSAY_VERIFIED":
                raise ValueError("Unexpected task approval status")
            self._tasks[entry["task_id"]] = entry
        for ln in (audit_directory / "bindingdb_full_kd_compound_median_candidates.jsonl").open(encoding="utf-8"):
            item = json.loads(ln)
            key = (item["task_id"], item["compound_sha256"])
            if key in self._measurements:
                raise ValueError("Duplicate measurement")
            self._measurements[key] = (item["smiles_isomeric"], float(item["median_pKd"]))
        for ln in (audit_directory / "bindingdb_full_kd_candidate_episodes_PRE_SPLIT.jsonl").open(encoding="utf-8"):
            item = json.loads(ln)
            task_id = item["task_id"]
            if task_id in self._episodes:
                raise ValueError("Duplicate episode task")
            if task_id not in self._tasks:
                raise ValueError("Episode not in task manifest")
            if item["status"] != "DO_NOT_USE_FOR_FINAL_BENCHMARK_UNTIL_ASSAY_QC_AND_SPLIT_LOCK":
                raise ValueError("Unexpected episode approval status")
            self._episodes[task_id] = item

    @property
    def candidate_task_count(self) -> int:
        return len(self._episodes)

    def task_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._episodes))

    def query_label_reference(self, task_id: str) -> EvaluationLabels:
        """Evaluator-only. Must never be passed to prediction functions."""
        source = self._episodes[task_id]
        q = tuple(source["query_ids"])
        return EvaluationLabels(task_id, q, tuple(self._measurements[(task_id, cid)][1] for cid in q))

    def prediction_episode(self, task_id: str, k: int = 5, draw: int = 0) -> PredictionEpisode:
        """Return an inference-only episode for one task.

        draw=0 reproduces predeclared SHA prefix support ordering.
        draw>0 uses stable SHA ranking of only the same task's candidates.
        Both sampling choices are independent of affinity labels.
        """
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if not isinstance(draw, int) or draw < 0:
            raise ValueError("draw must be nonnegative integer")
        source = self._episodes[task_id]
        task = self._tasks[task_id]
        support_ids = source["candidate_support_ids"]
        if k > len(support_ids):
            raise ValueError("Insufficient support candidates for requested K")
        if draw == 0:
            chosen = list(support_ids[:k])
        else:
            ranked = sorted(support_ids,
                key=lambda cid: hashlib.sha256((f"draw_v4|{draw}|{task_id}|{cid}").encode()).hexdigest())
            chosen = ranked[:k]
        query_ids = list(source["query_ids"])
        if set(chosen).intersection(query_ids):
            raise AssertionError("Query/support compound overlap")
        if len(query_ids)!=len(set(query_ids)) or len(chosen)!=len(set(chosen)):
            raise AssertionError("Repeated compound identity in episode")
        assay = task["assay_key"]
        support=tuple(
            SupportObservation(cid, self._measurements[(task_id,cid)][0],
                               self._measurements[(task_id,cid)][1], assay)
            for cid in chosen)
        queries=tuple(
            QueryCompound(cid,self._measurements[(task_id,cid)][0])
            for cid in query_ids)
        if any(obs.assay_key != assay for obs in support):
            raise AssertionError("Cross-assay support")
        if len(support)!=k or len(queries)!=10:
            raise AssertionError("Unexpected episode size")
        return PredictionEpisode(
            task_id=task_id,sequence_sha256=task["protein_seq_sha256"],
            protein_component_id=task["protein_component_id"],assay_key=assay,
            support=support,queries=queries)

    @staticmethod
    def validate_prediction_input(episode: PredictionEpisode) -> None:
        if episode.status != "PRE_SPLIT_UNAPPROVED":
            raise ValueError("Unsupported episode status; approval is separate")
        query_ids = [q.compound_id for q in episode.queries]
        support_ids = [x.compound_id for x in episode.support]
        if set(query_ids).intersection(support_ids):
            raise ValueError("Support/query overlap")
        if len(set(query_ids))!=len(query_ids) or len(set(support_ids))!=len(support_ids):
            raise ValueError("Duplicate query or support identities")
        if any(x.assay_key!=episode.assay_key for x in episode.support):
            raise ValueError("Cross-assay support measurement")
