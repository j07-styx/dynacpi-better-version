"""Reconstruct complete source-ID lineage for the seven conflict families.

Public output: aggregates only. Private output: full DOI/PMID/patent and
component/role provenance; must NEVER commit. No affinities or prediction
fields accessed, stored, compared or printed. Existing 'gather' parses raw
technical JSONL and projects task, eligible reactant ID and bibliographic
metadata only, with its original technical screening and integrity constraints.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from scripts.audit_bindingdb_source_linked_protein_groups_dev import source_ids
from scripts.prioritize_bindingdb_component_source_review_dev import gather

ROOT=Path(__file__).resolve().parents[1]
PREFLIGHT=ROOT/"research/bindingdb_publication_connected_protein_review_preflight_20261108.json"
QUEUE=ROOT/"data/audit/PROVISIONAL_SOURCE_FAMILY_EVIDENCE_QUEUE_PRIVATE_20261108.json"
SOURCE_LEDGER=ROOT/"data/audit/SOURCE_FAMILY_PRIMARY_ACCESS_LEDGER_PRIVATE_20261108.json"
PRIVATE_OUT=ROOT/"data/audit/SOURCE_FAMILY_COMPLETE_ELIGIBLE_PROVENANCE_PRIVATE_20261108.json"
PUBLIC_OUT=ROOT/"research/bindingdb_seven_family_eligible_source_provenance_audit_20261108.json"


def reconstruct(tasks: dict, source_by_task: dict, preflight: dict,
                seven_families: list[dict]) -> tuple[list[dict], dict]:
    """Pure projection of original eligible TASK source bibliographies; never labels."""
    if preflight.get("scientific_gate")!="BLOCKED_UNAPPROVED_NOT_TRAINING_READY":
        raise PermissionError("Input source linkage is not quarantined")
    mapping=preflight["linkage"]["source_family_by_protein_component"]
    members=preflight["linkage"]["family_members"]
    if set(source_by_task)!=set(tasks):
        raise ValueError("Missing or additional bibliographic task provenance")
    if len(set(mapping.values()))!=len(members):
        raise ValueError("Source-family maps do not agree")
    all_src_components=defaultdict(set)
    indexed=defaultdict(list)
    all_comp=set()
    for key, task in tasks.items():
        comp=task["protein_component_id"]
        if comp not in mapping:
            raise ValueError("Task protein component absent from frozen source preflight")
        all_comp.add(comp)
        identifiers=sorted(source_ids(source_by_task[key]))
        family=mapping[comp]
        indexed[family].append((task["task_id"],comp,identifiers))
        for ident in identifiers:
            all_src_components[ident].add(comp)
    if set(mapping)!=all_comp:
        raise ValueError("Source preflight protein component coverage mismatched")
    for ident,comps in all_src_components.items():
        if len({mapping[comp] for comp in comps})!=1:
            raise ValueError("Source identifier unexpectedly crosses source-family graph")
    if len(all_src_components)!=preflight["linkage"]["source_identifier_total"]:
        raise ValueError("All-source identifier total does not match frozen preflight")
    shared={k:v for k,v in all_src_components.items() if len(v)>1}
    if len(shared)!=preflight["linkage"]["shared_source_identifier_count"]:
        raise ValueError("Shared-publication identifier total does not match frozen preflight")
    sampled=preflight["linkage"]["shared_identifier_examples"]
    if set(sampled)!=set(shared):
        raise ValueError("Frozen shared identifier index is incomplete or inconsistent")
    for ident,comps in shared.items():
        expected=sampled[ident]
        if expected["components"]!=len(comps) or expected["component_prefixes"]!=[
            x[:12] for x in sorted(comps)[:8]
        ]:
            raise ValueError("Shared-publication original component witnesses disagree")

    if len(seven_families)!=7:
        raise ValueError("Expected seven originally flagged conflict families")
    selected=set()
    details=[]
    by_type=Counter()
    unsourced=0
    for entry in sorted(seven_families,key=lambda r:r["source_family_sha256"]):
        fam=entry["source_family_sha256"]
        if fam not in members or fam in selected:
            raise ValueError("Unknown/duplicated conflicted source family")
        selected.add(fam)
        if entry.get("review_status")!="UNREVIEWED_METADATA_CONFLICT_NOT_APPROVED":
            raise PermissionError("A conflict family was edited or approved")
        role_counts=entry["affected_provisional_role_component_counts"]
        if len(role_counts)<2:
            raise ValueError("Family no longer crosses roles")
        rows=indexed[fam]
        if not rows:
            raise ValueError("Conflict family lacks eligible tasks")
        observed={component for _,component,_ in rows}
        if observed!=set(members[fam]):
            raise ValueError("Complete family component coverage not observed")
        source_to_tasks=defaultdict(set)
        source_to_components=defaultdict(set)
        no_id_tasks=[]
        for task_id,comp,identifiers in rows:
            if not identifiers:
                no_id_tasks.append(task_id)
            for ident in identifiers:
                source_to_tasks[ident].add(task_id)
                source_to_components[ident].add(comp)
        sources=[]
        for ident in sorted(source_to_tasks):
            kind=ident.split(":",1)[0]
            if kind not in ("doi","pmid","patent"):
                raise ValueError("Unsupported source identifier")
            by_type[kind]+=1
            sources.append({
                "original_source_identifier":ident,
                "protein_component_ids":sorted(source_to_components[ident]),
                "eligible_task_ids":sorted(source_to_tasks[ident]),
                "links_two_or_more_protein_components":len(source_to_components[ident])>1,
                "review_status":"UNREVIEWED_SOURCE_IDENTITY_ONLY",
            })
        unsourced+=len(no_id_tasks)
        details.append({
            "source_family_sha256":fam,
            "affected_roles_and_components_PRIVATE":role_counts,
            "all_original_protein_component_ids":sorted(members[fam]),
            "technically_eligible_task_count":len(rows),
            "all_original_source_identifiers":sources,
            "technically_eligible_tasks_missing_all_DOI_PMID_patent":sorted(no_id_tasks),
            "original_record_table_crosswalk_verified":False,
            "exact_assayed_protein_construct_verified":False,
            "approval_status":"UNREVIEWED_NOT_SCIENTIFICALLY_APPROVED",
            "scope":"All technically eligible >=15-compound tasks underlying frozen preflight, NOT all BindingDB rows",
        })
    public={
        "status":"ELIGIBLE_SOURCE_PROVENANCE_REPLAY_NOT_ASSAY_APPROVAL",
        "scope":"Original source IDs for technically eligible >=15-compound tasks, same inventory as frozen preliminary study-family graph",
        "source_families_with_cross_role_conflicts":len(details),
        "all_eligible_tasks_total":len(tasks),
        "all_eligible_source_identifiers_total":len(all_src_components),
        "all_shared_cross_component_source_identifiers_total":len(shared),
        "all_eligible_source_families_total":len(members),
        "conflict_families_eligible_tasks_total":sum(x["technically_eligible_task_count"] for x in details),
        "conflict_families_distinct_original_source_identifiers":sum(len(x["all_original_source_identifiers"]) for x in details),
        "conflict_families_source_identifier_counts_by_type":dict(sorted(by_type.items())),
        "conflict_families_tasks_without_DOI_PMID_patent":unsourced,
        "conflict_families_with_complete_eligible_task_source_ID_inventory":len(details),
        "families_original_assay_record_crosswalk_verified":0,
        "families_assayed_construct_verified":0,
        "families_scientifically_approved":0,
        "scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "warnings":[
            "An exhaustive bibliography for the PREVIOUSLY SELECTED technically eligible inventory is not exhaustive of all papers, ineligible assays, patents, sources absent from BindingDB, or unresolved source aliases.",
            "The original source technical JSONL contains affinity fields, but the existing provenance projection accessed only metadata keys; no numeric affinity labels, predictions or held-out outcomes were used or emitted.",
            "Source-family membership is conservative DOI/PMID/patent connectivity, not proof of an identical assay, batch, construct or independently reviewed protein-homology relationship.",
            "All original source IDs and source-to-component/role links are CONFIDENTIAL, private and never committed.",
            "No source-table/record matching, reviewer approval, change in split membership, retraining, or reserved-test metric was performed."
        ],
    }
    return details,public


def main():
    if PRIVATE_OUT.exists() or PUBLIC_OUT.exists():
        raise FileExistsError("Refusing to overwrite immutable provenance audit")
    preflight=json.loads(PREFLIGHT.read_text(encoding="utf-8-sig"))
    queue=json.loads(QUEUE.read_text(encoding="utf-8"))
    tier=json.loads(SOURCE_LEDGER.read_text(encoding="utf-8"))
    if queue.get("status")!="PRIVATE_HUMAN_REVIEW_QUEUE_UNAPPROVED":
        raise PermissionError("Required private conflict review queue is not unreviewed")
    if tier.get("status")!="PRIVATE_SOURCE_ACCESS_REVIEW_REQUIRED":
        raise PermissionError("Source access ledger does not state unreviewed")
    if {x["source_family_sha256"] for x in tier["families"]}!={
        x["source_family_sha256"] for x in queue["families"]
    }:
        raise ValueError("Changed seven-family review identities")
    # Preserve the exact historical eligible-inventory source and technical
    # selection versions. These are file digests, never affinity observations.
    for filename,expected in preflight["data_source_integrity_sha256"].items():
        h=sha256()
        with (ROOT/"data/audit"/filename).open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""):
                h.update(chunk)
        if h.hexdigest()!=expected:
            raise ValueError("Pre-split technical source inventory checksum changed")
    tasks,source,assay,counts=gather()
    # The original gather() explicitly accesses ONLY bibliographic and construct
    # metadata from eligible source rows. It does parse technical JSONL files
    # with other fields present, but never extracts or scores potency values.
    private,public=reconstruct(tasks,source,preflight,queue["families"])
    if public["all_eligible_tasks_total"]!=1146 or public["all_eligible_source_identifiers_total"]!=729:
        raise ValueError("Replayed technical inventory unexpectedly changed")
    if public["all_shared_cross_component_source_identifiers_total"]!=26:
        raise ValueError("Replayed source-link graph unexpectedly changed")
    raw_source=PREFLIGHT.read_bytes()
    doc={
        "status":"CONFIDENTIAL_COMPLETE_ELIGIBLE_SOURCE_IDS_NO_APPROVAL",
        "original_frozen_preflight_sha256":sha256(raw_source).hexdigest(),
        "original_private_conflict_queue_sha256":sha256(QUEUE.read_bytes()).hexdigest(),
        "families":private,
    }
    with PRIVATE_OUT.open("x",encoding="utf-8") as f:
        json.dump(doc,f,indent=2)
        f.write("\n")
    with PUBLIC_OUT.open("x",encoding="utf-8") as f:
        json.dump(public,f,indent=2)
        f.write("\n")
    print(json.dumps(public,indent=2))


if __name__=="__main__":
    main()
