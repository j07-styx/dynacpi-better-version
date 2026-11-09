"""Fail-closed publication/protein group split preflight (NOT assay approval).

Input: a role for each candidate task, original protein component per task,
and an already-audited conservative source-family mapping. Ensure no known
DOI/PMID/patent-connected source family crosses between role assignments.

Do not use this as an approval mechanism. Missing source metadata can cause
unknown shared publications, and protein groups remain provisional.
"""


ALLOWED_ROLES=frozenset(
    ("base_train","base_validation","adapter_train","adapter_validation","test")
)


class SourceFamilySplitError(PermissionError):
    pass


def require_source_disjoint_roles(task_roles,task_metadata,
                                  source_family_by_protein_component):
    """Reject source-connected component overlap, missing mappings or roles.

    task_roles: dict(task_id -> assigned study role)
    task_metadata: iterable/dict values with task_id, protein_component_id
    family_by_component: audited identity map including ALL candidate protein
                         components to study source-family identifiers.
    """
    if not isinstance(task_roles,dict) or not task_roles:
        raise SourceFamilySplitError("Missing task-to-role mapping")
    if not isinstance(source_family_by_protein_component,dict) or not source_family_by_protein_component:
        raise SourceFamilySplitError("Missing reviewed source-family mapping")
    records=list(task_metadata.values()) if isinstance(task_metadata,dict) else list(task_metadata)
    tasks={}
    for record in records:
        task_id=record.get("task_id")
        comp=record.get("protein_component_id")
        if not task_id or not comp:
            raise SourceFamilySplitError("Task ID or component ID absent")
        if task_id in tasks:
            raise SourceFamilySplitError("Duplicate task ID in metadata")
        tasks[task_id]=comp
    if set(tasks)!=set(task_roles):
        raise SourceFamilySplitError("Role map does not exactly cover the audited tasks")
    families={}
    counts={role:0 for role in sorted(ALLOWED_ROLES)}
    for task_id,component in tasks.items():
        role=task_roles[task_id]
        if role not in ALLOWED_ROLES:
            raise SourceFamilySplitError("Unknown split role")
        family=source_family_by_protein_component.get(component)
        if not isinstance(family,str) or len(family)!=64:
            raise SourceFamilySplitError("Missing or invalid source-family evidence")
        previous=families.get(family)
        if previous is not None and previous!=role:
            raise SourceFamilySplitError(
                "Same protein/source connected family assigned multiple study roles"
            )
        families[family]=role
        counts[role]+=1
    # Does NOT require positive quota in every role: evaluating one partial
    # worklist isn't a frozen cohort. Formal scientific gate does that.
    return {
        "status":"METADATA_ONLY_PROTEIN_AND_PUBLICATION_SOURCE_DISJOINT_NO_APPROVAL",
        "tasks_checked":len(tasks),
        "source_connected_families_checked":len(families),
        "role_task_counts":counts,
        "guarantees":"No known source-family ID crosses roles within these audited task assignments",
        "does_not_guarantee":[
            "No unknown publication batch leakage when DOI or PMID metadata is absent",
            "Independently verified protein homology",
            "Assay, compound, construct, target or source record scientific approval",
            "Approved or frozen scientific training/test split",
        ]
    }
