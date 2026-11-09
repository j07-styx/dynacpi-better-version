"""Fail-closed lock for the retired, unapproved TransferCPI pilot train entrypoint.

The historical five-arm DEV comparison remains reproducible from saved artifacts
but new runs on the provisional, publication-linked split are not authorized.
This check reads metadata JSONs only and must run before any label-loader or GPU.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path

from dynacpi.scientific_cohort_gate import current_status
from scripts.audit_provisional_bindingdb_source_split_dev import generate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("research/bindingdb_publication_connected_protein_review_preflight_20261108.json")
SPLIT = Path("research/bindingdb_frozen_component_split_manifest_dev.json")
REPORT = Path("research/bindingdb_provisional_role_source_overlap_audit_20261108.json")


class ProvisionalRetrainingBlocked(PermissionError):
    pass


def classify_retraining_request(analysis: dict, scientific_status: dict) -> str:
    """Pure decision. Never returns ALLOW: old provisional DEV pilot is retired."""
    if analysis.get("families_crossing_two_or_more_roles", 0) > 0:
        return "PUBLICATION_SOURCE_LINKS_CROSS_PROVISIONAL_ROLES"
    if not analysis.get("publication_family_disjoint"):
        return "NO_VERIFIABLE_SOURCE_FAMILY_INDEPENDENCE"
    if scientific_status.get("status") != "ARTIFACT_GATE_PASSED_NOT_BIOLOGICAL_VALIDATION":
        return "UNAPPROVED_EXPERIMENTAL_COHORT"
    return "RETIRED_PROVISIONAL_PROTOCOL_REQUIRES_NEW_APPROVED_COHORT_AND_SPLIT"


def refuse_provisional_transfer_retraining(root: Path = ROOT) -> None:
    root = Path(root)
    source, split, report = root / SOURCE, root / SPLIT, root / REPORT
    if not all(p.is_file() for p in (source, split, report)):
        raise ProvisionalRetrainingBlocked("Missing source audit, split manifest, or immutable report: refuse provisional retraining")
    try:
        original = json.loads(report.read_text(encoding="utf-8"))
        for path, expected in (
            (source, original["original_source_linkage_sha256"]),
            (split, original["original_provisional_split_sha256"]),
        ):
            if sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Audit source or frozen provisional split SHA256 changed")
        recomputed = generate(source, split)
        if original != recomputed:
            raise ValueError("Recorded metadata-source audit differs from current inputs")
        reason = classify_retraining_request(recomputed, current_status(root))
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        raise ProvisionalRetrainingBlocked("Provisional source/role guard cannot be verified; fail closed") from exc
    raise ProvisionalRetrainingBlocked(
        f"{reason}: New training on the historical BindingDB provisional "
        "train/inner-DEV split is blocked; previously saved runs remain archival. "
        "Independent source adjudication, a newly registered study-disjoint protocol "
        "and a separate approved training entrypoint are required."
    )
