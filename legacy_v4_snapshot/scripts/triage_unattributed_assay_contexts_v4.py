"""Metadata-only assay-context triage of 62 citation-unattributed tasks.

The original source-neighbor PRIVATE review and two frozen PRE_SPLIT metadata
files are required. Reads no raw binding measurements, molecular structures,
original TEST outcomes, support/query affinities or provisional split roles.

Classifications are method-review leads, NOT source or endpoint verification.
The public report has aggregate categories only, no task, assay, sequence,
protein-component, publication, or source-family identifiers.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/"data/audit/bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl"
CONTEXT=ROOT/"data/audit/bindingdb_full_kd_assay_contexts.jsonl"
PREFLIGHT=ROOT/"research/bindingdb_publication_connected_protein_review_preflight_20261108.json"
NEIGHBORS=ROOT/"data/audit/UNATTRIBUTED_TASK_NEIGHBOR_LEADS_PRIVATE_20261109.json"
PRIVATE_OUT=ROOT/"data/audit/UNATTRIBUTED_TASK_ASSAY_CONTEXT_PRIVATE_20261109.json"
PUBLIC_OUT=ROOT/"research/bindingdb_unattributed_assay_context_20261109.json"

TASK_FIELDS=frozenset((
    "assay_description_sha256","assay_key","n_discordant_replicate_groups",
    "n_duplicate_compounds","n_multicomponent_smiles","n_unique_compounds",
    "protein_component_id","protein_seq_sha256","quality_review_flags",
    "status","task_id",
))
CONTEXT_FIELDS=frozenset((
    "article_doi_count","assay_key","description","name",
    "preliminary_size_band","raw_record_count","sequence_sha256","signals"
))
SIGNAL_KEYS=frozenset(("biochemical","cellular","ic50","kd","ki",
                       "kinetics","ph","temperature"))
KINASE_NAME="Kinase Inhibitor Selectivity Profiling Assay"
CATEGORIES=(
    "GENERIC_KINASE_SELECTIVITY_PROFILING",
    "D3R_CHALLENGE_SERIES",
    "OCTETRED_LABELED_METHOD",
    "THERMOFLUOR_LABELED_METHOD",
    "OTHER_UNRESOLVED_ASSAY_CONTEXT",
)


def classify(name):
    if name==KINASE_NAME:return CATEGORIES[0]
    if re.fullmatch(r"D3R20[0-9]{2}",name or ""):return CATEGORIES[1]
    if re.fullmatch(r"OctetRed(?:_Method[0-9]+)?",name or ""):
        return CATEGORIES[2]
    if name=="Thermofluor":return CATEGORIES[3]
    return CATEGORIES[4]


def read_jsonl_metadata(path,checksum,allowed):
    if sha256(Path(path).read_bytes()).hexdigest()!=checksum:
        raise PermissionError("Frozen pre-split metadata source checksum changed")
    records=[]
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            if set(row)!=allowed:
                raise PermissionError("Nonmetadata or changed original input schema")
            if allowed==CONTEXT_FIELDS:
                if not isinstance(row["signals"],dict) or (
                    set(row["signals"])!=SIGNAL_KEYS
                ):
                    raise PermissionError("Unexpected assay signal schema")
            records.append(row)
    return records


def audit_contexts(task_rows,assay_contexts,neighbors):
    if neighbors.get("status")!="PRIVATE_REVIEW_ONLY_NO_SOURCE_ASSIGNMENTS":
        raise PermissionError("Unattributed source-lead review is no longer quarantined")
    unresolved=neighbors["records"]
    if len(unresolved)!=62 or len({x["task_id"] for x in unresolved})!=62:
        raise ValueError("Original 62 unresolved tasks changed")
    if any(x["source_record_verified"] or x["assay_approved"]
           or x["training_eligible"] for x in unresolved):
        raise PermissionError("Unattributed tasks cannot be approved")
    task_index={}
    for t in task_rows:
        if int(t["n_unique_compounds"])<15:continue
        if t["task_id"] in task_index:
            raise ValueError("Duplicate eligible task ID")
        task_index[t["task_id"]]=t
    context_index={}
    for c in assay_contexts:
        key=(c["sequence_sha256"],c["assay_key"])
        if key in context_index:
            raise ValueError("Duplicate original assay-context metadata")
        context_index[key]=c
    if len(task_index)!=1146:
        raise ValueError("Original 1146 technically eligible tasks changed")
    private=[]
    counts=Counter()
    components=set()
    for review in unresolved:
        tid=review["task_id"]
        if tid not in task_index:
            raise ValueError("Missing original technical task")
        t=task_index[tid]
        if (
            t["protein_component_id"]!=review["protein_component_id"]
            or t["assay_key"]!=review["assay_key"]
        ):
            raise ValueError("Original task identity and review worklist disagree")
        key=(t["protein_seq_sha256"],t["assay_key"])
        if key not in context_index:
            raise ValueError("Missing exact protein-sequence/assay context")
        c=context_index[key]
        if not isinstance(c["article_doi_count"],int) or c["article_doi_count"]<0:
            raise ValueError("Assay context DOI count is malformed")
        if not isinstance(c["name"],str) or not isinstance(c["description"],str):
            raise ValueError("Assay context description is malformed")
        if any(not isinstance(v,bool) for v in c["signals"].values()):
            raise ValueError("Original assay signals must be boolean metadata")
        category=classify(c["name"])
        counts[category]+=1
        components.add(t["protein_component_id"])
        private.append({
            "task_id":tid,
            "protein_component_id":t["protein_component_id"],
            "protein_sequence_sha256":t["protein_seq_sha256"],
            "assay_key":t["assay_key"],
            "method_context_category":category,
            "exact_assay_description_sha256":t["assay_description_sha256"],
            "assay_name":c["name"],
            "assay_description_sha256":sha256(c["description"].encode("utf-8")).hexdigest(),
            "assay_description_has_text":bool(c["description"].strip()),
            "assay_article_doi_count_metadata_only":c["article_doi_count"],
            "source_endpoint_keyword_flags":dict(c["signals"]),
            "source_record_verified":False,
            "assayed_protein_construct_verified":False,
            "exact_Kd_endpoint_verified":False,
            "original_publication_verified":False,
            "training_eligible":False,
            "warning":"Assay-name metadata neither assigns an original paper nor establishes that any numerical affinity is exact Kd.",
        })
    if set(counts)-set(CATEGORIES):
        raise ValueError("Unlisted assay category")
    # Confirm the observed fixed source-lost task group, never treat the
    # generated counts as confirmation of the scientific quality of a source.
    if len(private)!=62 or len(components)!=2:
        raise ValueError("Frozen source-lost task grouping changed")
    if any(p["assay_article_doi_count_metadata_only"]!=0 for p in private):
        raise ValueError("Original unresolved assay DOI metadata changed")
    if any(p["source_endpoint_keyword_flags"]["kd"]
           or p["source_endpoint_keyword_flags"]["ic50"] for p in private):
        raise ValueError("Original missing-source assay keyword signals changed")
    summary={
        "status":"UNATTRIBUTED_ASSAY_CONTEXT_TRIAGE_NO_EXPERIMENTAL_APPROVAL",
        "original_unattributed_task_count":62,
        "original_pre_split_component_count":len(components),
        "assay_context_exact_task_sequence_matches":len(private),
        "assay_context_categories":{name:counts[name] for name in CATEGORIES},
        "tasks_with_assay_context_DOI_count_zero":sum(
            r["assay_article_doi_count_metadata_only"]==0 for r in private),
        "tasks_with_no_explicit_Kd_or_IC50_keyword_flag":sum(
            not r["source_endpoint_keyword_flags"]["kd"]
            and not r["source_endpoint_keyword_flags"]["ic50"] for r in private),
        "tasks_with_exact_source_record_verified":0,
        "tasks_with_physical_construct_verified":0,
        "tasks_with_exact_Kd_endpoint_verified":0,
        "tasks_with_independent_source_approval":0,
        "tasks_added_to_training_or_test":0,
        "scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "warnings":[
            "Generic assay metadata and keyword flags are not evidence of no actual affinity values; they cannot verify original measurement type or numerical quality.",
            "The 54 generic kinase contexts could share method prose without sharing an original assay source; no new bibliographic family merges are made.",
            "No task, source, protein, compound, sequence, assay or original assay-description identifiers are emitted in the public aggregate.",
            "No raw binding affinity outcome, model prediction, reserved TEST result, original numerical label, or protected role assignment was read.",
        ],
    }
    return private,summary


def main():
    if PRIVATE_OUT.exists() or PUBLIC_OUT.exists():
        raise FileExistsError("Refusing to overwrite immutable source-context review")
    old_pref=PREFLIGHT.read_bytes()
    old_neighbors=NEIGHBORS.read_bytes()
    preflight=json.loads(old_pref)
    checksums=preflight["data_source_integrity_sha256"]
    task_rows=read_jsonl_metadata(TASK,checksums[TASK.name],TASK_FIELDS)
    contexts=read_jsonl_metadata(CONTEXT,checksums[CONTEXT.name],CONTEXT_FIELDS)
    detail,public=audit_contexts(task_rows,contexts,json.loads(old_neighbors))
    if old_pref!=PREFLIGHT.read_bytes() or old_neighbors!=NEIGHBORS.read_bytes():
        raise PermissionError("Frozen source review witnesses changed during audit")
    with PRIVATE_OUT.open("x",encoding="utf-8") as stream:
        json.dump({"status":"PRIVATE_ASSAY_CONTEXT_REVIEW_NO_SOURCE_ASSIGNMENT",
                   "original_preflight_sha256":sha256(old_pref).hexdigest(),
                   "source_neighbor_snapshot_sha256":sha256(old_neighbors).hexdigest(),
                   "records":detail},stream,indent=2)
        stream.write("\n")
    with PUBLIC_OUT.open("x",encoding="utf-8") as stream:
        json.dump(public,stream,indent=2)
        stream.write("\n")
    print(json.dumps(public,indent=2))


if __name__=="__main__":main()
