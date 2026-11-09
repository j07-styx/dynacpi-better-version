"""Fail-closed research cohort and split authorization checks.

This module performs artifact and metadata integrity checks only. It does not
independently verify any biological measurement. Its acceptance is necessary,
never sufficient, for the scientific relevance of a cohort.

An executable scientific training entrypoint must call require_approved_cohort
before reading any affinity labels. The old engineering pilot stays separate.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
from collections import Counter
from dynacpi.approved_source_family_gate import (
    SOURCE_MANIFEST_NAME, require_frozen_source_family_independence,
    reject_duplicate_json_keys,
)

ROLES=("base_train","base_validation","adapter_train","adapter_validation","test")
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for ch in iter(lambda:f.read(1024*1024),b""):
            h.update(ch)
    return h.hexdigest()

def require_approved_cohort(cohort_path, split_path):
    cohort_path=Path(cohort_path)
    split_path=Path(split_path)
    if not cohort_path.is_file() or not split_path.is_file():
        raise PermissionError("Blocked: independently reviewed approved cohort and frozen split are required.")
    try:
        cohort=json.loads(cohort_path.read_text(encoding="utf-8"),
                          object_pairs_hook=reject_duplicate_json_keys)
        split=json.loads(split_path.read_text(encoding="utf-8"),
                         object_pairs_hook=reject_duplicate_json_keys)
    except (ValueError, UnicodeError, OSError) as exc:
        raise PermissionError("Blocked: frozen cohort or split JSON invalid.") from exc
    if cohort.get("status")!="APPROVED_EXPERIMENTAL_COHORT_SOURCE_REVIEWED":
        raise PermissionError("Blocked: source-reviewed approval status missing.")
    if split.get("status")!="FROZEN_PROTEIN_GROUP_SPLIT":
        raise PermissionError("Blocked: protein-group split is not frozen.")
    if split.get("cohort_sha256")!=sha256(cohort_path):
        raise PermissionError("Blocked: cohort changed since split was frozen.")
    if not all(cohort.get(k) for k in ("review_policy_id","reviewer_identity","review_evidence_manifest_sha256")):
        raise PermissionError("Blocked: approval provenance is incomplete.")
    if not re.fullmatch(r"[0-9a-f]{64}",str(cohort["review_evidence_manifest_sha256"])):
        raise PermissionError("Blocked: approved-cohort review evidence SHA256 malformed.")
    if not all(split.get(k) for k in ("split_policy_id","freeze_timestamp_utc","frozen_by")):
        raise PermissionError("Blocked: split freeze metadata incomplete.")
    try:
        freeze_date=datetime.fromisoformat(split["freeze_timestamp_utc"].replace("Z","+00:00"))
        if freeze_date.tzinfo is None:raise ValueError("Freeze must include time zone")
    except (TypeError, ValueError) as exc:
        raise PermissionError("Blocked: invalid freeze timestamp.") from exc
    rows=cohort.get("tasks")
    assigned=split.get("task_roles")
    if not isinstance(rows,list) or not isinstance(assigned,dict):
        raise PermissionError("Blocked: cohort task records and split assignments absent.")
    if not rows or len(rows)!=len({x["task_id"] for x in rows}):
        raise PermissionError("Blocked: no tasks or repeated task IDs.")
    taskmap={x["task_id"]:x for x in rows}
    if set(taskmap)!=set(assigned):
        raise PermissionError("Blocked: split assignments differ from approved cohort.")
    componentrole={}
    role_counts=Counter()
    for tid,task in taskmap.items():
        role=assigned[tid]
        if role not in ROLES:raise PermissionError("Blocked: invalid role.")
        comp=task.get("protein_component_id")
        if not comp or not task.get("source_review_record_id"):
            raise PermissionError("Blocked: missing protein component or source adjudication record.")
        if int(task.get("n_unique_compounds",0))<15:
            raise PermissionError("Blocked: insufficient chemistry to form K5 candidate episodes.")
        if comp in componentrole and componentrole[comp]!=role:
            raise PermissionError("Blocked: same protein component assigned multiple roles.")
        componentrole[comp]=role
        role_counts[role]+=1
    if not all(role_counts[r]>=1 for r in ROLES):
        raise PermissionError("Blocked: one or more roles are empty.")
    if len(componentrole)<25 or len({c for c in componentrole})<25:
        raise PermissionError("Blocked: too few independent protein groups.")
    comp_counts=Counter(componentrole.values())
    if any(comp_counts[r]<3 for r in ROLES):
        raise PermissionError("Blocked: fewer than three protein groups in a study role.")
    # A protein-only split is insufficient: the source-family graph must also
    # be reviewed, frozen, bound by checksums and disjoint across ALL roles.
    source_proof=require_frozen_source_family_independence(
        cohort_path,split_path,split_path.with_name(SOURCE_MANIFEST_NAME))
    return {"status":"ARTIFACT_GATE_PASSED_NOT_BIOLOGICAL_VALIDATION",
            "tasks":len(rows),"components":len(componentrole),
            "role_counts":dict(role_counts),"component_role_counts":dict(comp_counts),
            "source_family_artifact_proof":source_proof}

def current_status(root):
    """Never infer approval from PRE_SPLIT filename or an unreviewed review CSV."""
    root=Path(root)
    audit=root/"data/audit"
    cohort=audit/"bindingdb_source_approved_cohort.json"
    split=audit/"bindingdb_frozen_protein_group_split.json"
    try:
        status=require_approved_cohort(cohort,split)
    except (PermissionError,FileNotFoundError,ValueError,KeyError,TypeError) as exc:
        status={"status":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY","reason":str(exc)}
    return status
