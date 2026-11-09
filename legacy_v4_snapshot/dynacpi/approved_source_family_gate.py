"""Additional fail-closed PUBLICATION-SOURCE FAMILY freeze preflight.

Consumes only separately reviewed/frozen artifact metadata. Does NOT validate
original assay tables, experimental constructs, or reviewer expertise. It is a
necessary additional check for a future scientific cohort, never approval.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import json
import re

from dynacpi.source_publication_split_guard import (
    SourceFamilySplitError,
    require_source_disjoint_roles,
)


SOURCE_MANIFEST_NAME = "bindingdb_frozen_source_families.json"
SOURCE_STATUS = "SOURCE_FAMILY_GRAPH_FROZEN_AND_SOURCE_REVIEWED"
FROZEN_STATUS = "FROZEN_PROTEIN_GROUP_SPLIT"
COHORT_STATUS = "APPROVED_EXPERIMENTAL_COHORT_SOURCE_REVIEWED"
ROLES = ("base_train", "base_validation", "adapter_train", "adapter_validation", "test")


def reject_duplicate_json_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate key in frozen JSON artifact")
        obj[key] = value
    return obj


def file_digest(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_frozen_source_family_independence(cohort_path, split_path, source_path):
    """Check frozen source provenance, complete membership, and role independence.

    The caller must also run the full scientific-cohort gate. This verifies
    artifact integrity only; it cannot independently certify assay evidence.
    """
    cohort_path, split_path, source_path = map(Path, (cohort_path, split_path, source_path))
    if not all(path.is_file() for path in (cohort_path, split_path, source_path)):
        raise PermissionError("Blocked: source-reviewed cohort, split and frozen source-family map required")
    try:
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"),
                            object_pairs_hook=reject_duplicate_json_keys)
        split = json.loads(split_path.read_text(encoding="utf-8"),
                           object_pairs_hook=reject_duplicate_json_keys)
        manifest = json.loads(source_path.read_text(encoding="utf-8"),
                              object_pairs_hook=reject_duplicate_json_keys)
    except (OSError, UnicodeError, ValueError) as exc:
        raise PermissionError("Blocked: source-family evidence JSON invalid") from exc

    if (cohort.get("status") != COHORT_STATUS or
            split.get("status") != FROZEN_STATUS or
            manifest.get("status") != SOURCE_STATUS):
        raise PermissionError("Blocked: missing reviewed frozen source-family statuses")
    if manifest.get("cohort_sha256") != file_digest(cohort_path):
        raise PermissionError("Blocked: source-family graph does not match approved cohort checksum")
    if split.get("source_family_manifest_sha256") != file_digest(source_path):
        raise PermissionError("Blocked: frozen split not bound to source-family graph checksum")
    if not all(manifest.get(k) for k in (
        "source_linkage_policy_id", "reviewer_identity",
        "review_evidence_manifest_sha256", "freeze_timestamp_utc"
    )):
        raise PermissionError("Blocked: source-family review provenance fields absent")
    evidence_hash = manifest["review_evidence_manifest_sha256"]
    if not isinstance(evidence_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", evidence_hash):
        raise PermissionError("Blocked: source-family review evidence digest invalid")
    try:
        stamp = manifest["freeze_timestamp_utc"]
        frozen = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if frozen.tzinfo is None:
            raise ValueError("Timezone required")
    except (TypeError, AttributeError, ValueError) as exc:
        raise PermissionError("Blocked: source-family freeze timestamp must have a timezone") from exc
    mapped = manifest.get("source_family_by_protein_component")
    tasks = cohort.get("tasks")
    roles = split.get("task_roles")
    if not isinstance(mapped, dict) or not isinstance(tasks, list) or not isinstance(roles, dict):
        raise PermissionError("Blocked: source-family mapping or task roles absent")
    if len(tasks) == 0 or any(not isinstance(task, dict) for task in tasks):
        raise PermissionError("Blocked: source-reviewed task list missing")
    components = {r.get("protein_component_id") for r in tasks}
    if None in components or "" in components or set(mapped) != components:
        raise PermissionError("Blocked: source-family mapping must cover exactly all approved protein components")
    if any(not isinstance(family, str) or not re.fullmatch(r"[0-9a-f]{64}", family)
           for family in mapped.values()):
        raise PermissionError("Blocked: malformed source-family identity hash")
    try:
        check = require_source_disjoint_roles(roles, tasks, mapped)
    except (SourceFamilySplitError, ValueError, TypeError, KeyError) as exc:
        raise PermissionError("Blocked: known DOI/PMID/patent-connected family assigned across roles") from exc

    counts = Counter()
    family_by_role = {}
    for task in tasks:
        comp = task["protein_component_id"]
        role = roles[task["task_id"]]
        family_by_role[mapped[comp]] = role
    counts.update(family_by_role.values())
    # Source families, rather than merely protein components, are the unit
    # of independence. Require adequate number per role for any future split.
    if any(counts[role] < 3 for role in ROLES):
        raise PermissionError("Blocked: fewer than three independent source families in a study role")
    if len(family_by_role) < 25:
        raise PermissionError("Blocked: fewer than 25 source-independent study families")
    return {
        "status": "SOURCE_FAMILY_ARTIFACT_GATE_PASSED_NOT_SCIENTIFIC_VALIDATION",
        "source_family_count": len(family_by_role),
        "role_source_family_counts": dict(sorted(counts.items())),
        "source_family_manifest_sha256": file_digest(source_path),
        "limitation": "Artifact membership and source metadata consistency only, never biological/assay approval",
    }
