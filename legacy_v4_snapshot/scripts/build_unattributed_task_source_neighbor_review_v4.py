"""Bibliographic neighbor leads for tasks with no recorded source IDs.

Reads the frozen, eleven-field PRE_SPLIT task METADATA file (no affinities)
and the existing PRIVATE source-family bibliography index. NEVER reads assay
measurements, molecules, model outputs, reserved-test scores, or split roles.

Every suggested bibliography belongs to ANOTHER task. A source neighbor is
not source attribution and cannot unlock training or an evaluation cohort.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
TASK_FILE=ROOT/"data/audit/bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl"
PREFLIGHT=ROOT/"research/bindingdb_publication_connected_protein_review_preflight_20261108.json"
SOURCE=ROOT/"data/audit/SOURCE_FAMILY_COMPLETE_ELIGIBLE_PROVENANCE_PRIVATE_20261108.json"
PRIVATE_OUT=ROOT/"data/audit/UNATTRIBUTED_TASK_NEIGHBOR_LEADS_PRIVATE_20261109.json"
PUBLIC_OUT=ROOT/"research/bindingdb_unattributed_task_neighbor_summary_20261109.json"

ALLOWED=frozenset(("assay_description_sha256","assay_key","n_discordant_replicate_groups",
    "n_duplicate_compounds","n_multicomponent_smiles","n_unique_compounds",
    "protein_component_id","protein_seq_sha256","quality_review_flags","status","task_id"))
TIERS=("SAME_COMPONENT_AND_ASSAY_NEIGHBORS","SAME_COMPONENT_OTHER_ASSAY_NEIGHBORS",
       "SAME_ASSAY_OTHER_COMPONENT_NEIGHBORS","FAMILY_ONLY_NO_NEARBY_BIBLIOGRAPHY")


def read_tasks(path,expected_sha):
    if sha256(Path(path).read_bytes()).hexdigest()!=expected_sha:
        raise PermissionError("Frozen metadata input checksum changed")
    tasks={}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():continue
            row=json.loads(line)
            if set(row)!=ALLOWED:
                raise PermissionError("Unexpected fields: refuse nonmetadata content")
            if int(row["n_unique_compounds"])<15:continue
            tid=row["task_id"]
            if not isinstance(tid,str) or not tid or tid in tasks:
                raise ValueError("Missing or repeated task ID")
            tasks[tid]={"protein_component_id":row["protein_component_id"],
                        "assay_key":row["assay_key"]}
    return tasks


def review_neighbors(tasks,preflight,index,*,strict=True):
    if preflight.get("scientific_gate")!="BLOCKED_UNAPPROVED_NOT_TRAINING_READY":
        raise PermissionError("Source graph not quarantined")
    if index.get("status")!="CONFIDENTIAL_COMPLETE_ELIGIBLE_SOURCE_IDS_NO_APPROVAL":
        raise PermissionError("Original source index is not unapproved")
    component_family=preflight["linkage"]["source_family_by_protein_component"]
    members=preflight["linkage"]["family_members"]
    if {r["protein_component_id"] for r in tasks.values()}!=set(component_family):
        raise ValueError("Protein-component inventory diverged from preflight")
    private=[]
    tiers=Counter()
    eligible_total=0
    seen_families=set()
    for f in index["families"]:
        family=f["source_family_sha256"]
        if family in seen_families or family not in members:
            raise ValueError("Unknown or repeated source family")
        seen_families.add(family)
        comps=set(f["all_original_protein_component_ids"])
        if comps!=set(members[family]):
            raise ValueError("Preflight component witness changed")
        if f["approval_status"]!="UNREVIEWED_NOT_SCIENTIFICALLY_APPROVED":
            raise PermissionError("Source family is not unreviewed")
        expected={t for t,r in tasks.items() if r["protein_component_id"] in comps}
        eligible_total+=len(expected)
        referenced=set()
        by_comp=defaultdict(set)
        by_assay=defaultdict(set)
        ids=set()
        for src in f["all_original_source_identifiers"]:
            sid=src["original_source_identifier"]
            if sid in ids:raise ValueError("Duplicate source ID")
            ids.add(sid)
            linked=set(src["eligible_task_ids"])
            if not linked or not linked<=expected:
                raise ValueError("Source link crossed original family")
            if {tasks[t]["protein_component_id"] for t in linked}!=set(
                src["protein_component_ids"]
            ):
                raise ValueError("Original source-to-component witness mismatch")
            referenced.update(linked)
            for t in linked:
                by_comp[tasks[t]["protein_component_id"]].add(sid)
                by_assay[tasks[t]["assay_key"]].add(sid)
        missing=set(f["technically_eligible_tasks_missing_all_DOI_PMID_patent"])
        if missing&referenced or missing|referenced!=expected:
            raise ValueError("Original sourced and unattributed tasks do not partition")
        if len(expected)!=f["technically_eligible_task_count"]:
            raise ValueError("Family technical task count changed")
        for tid in sorted(missing):
            row=tasks[tid]
            comp=by_comp.get(row["protein_component_id"],set())
            assay=by_assay.get(row["assay_key"],set())
            if comp&assay: tier,candidates=TIERS[0],comp&assay
            elif comp:tier,candidates=TIERS[1],comp
            elif assay:tier,candidates=TIERS[2],assay
            else:tier,candidates=TIERS[3],set()
            tiers[tier]+=1
            private.append({
                "task_id":tid,"protein_component_id":row["protein_component_id"],
                "source_family_sha256":family,"assay_key":row["assay_key"],
                "review_tier":tier,
                "NEIGHBOR_source_identifiers_NOT_ATTRIBUTED":sorted(candidates),
                "other_component_source_count":len(comp),
                "other_assay_source_count":len(assay),
                "all_family_source_count":len(ids),
                "source_record_verified":False,"construct_verified":False,
                "training_eligible":False,"assay_approved":False,
            })
    if len(private)!=len({x["task_id"] for x in private}):
        raise ValueError("Duplicate unattributed task")
    if strict and (
        len(tasks)!=1146 or len(members)!=142 or len(seen_families)!=7
        or eligible_total!=708 or len(private)!=62
    ):
        raise ValueError("Frozen original inventory counts changed")
    private.sort(key=lambda r:(
        TIERS.index(r["review_tier"]),
        len(r["NEIGHBOR_source_identifiers_NOT_ATTRIBUTED"]),r["task_id"]))
    public={
        "status":"UNATTRIBUTED_TASK_BIBLIOGRAPHIC_NEIGHBORS_NOT_SOURCE_APPROVAL",
        "pre_split_conflicted_source_families":len(seen_families),
        "technical_tasks_in_those_families":eligible_total,
        "technical_tasks_with_no_original_DOI_PMID_or_patent":len(private),
        "source_neighbor_tier_counts":{k:tiers[k] for k in TIERS},
        "original_source_records_verified":0,"physical_constructs_verified":0,
        "scientifically_approved_tasks":0,
        "source_imputation_performed":False,"reassigned_split_roles":False,
        "scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "warnings":[
            "A bibliography shared by another task or by the same family is a review lead, NOT evidence that it produced this task's measurements.",
            "Neither the earlier reserved test outcomes nor any numerical affinity measurements or chemical identifiers are used.",
            "Private task, source, assay and component mappings are excluded from this aggregate.",
        ],
    }
    return private,public


def main():
    if PRIVATE_OUT.exists() or PUBLIC_OUT.exists():
        raise FileExistsError("Refusing to overwrite source review artifacts")
    p_bytes=PREFLIGHT.read_bytes();s_bytes=SOURCE.read_bytes()
    preflight=json.loads(p_bytes);source=json.loads(s_bytes)
    expected=preflight["data_source_integrity_sha256"][TASK_FILE.name]
    tasks=read_tasks(TASK_FILE,expected)
    details,public=review_neighbors(tasks,preflight,source)
    if p_bytes!=PREFLIGHT.read_bytes() or s_bytes!=SOURCE.read_bytes():
        raise PermissionError("Private source witnesses changed during review")
    with PRIVATE_OUT.open("x",encoding="utf-8") as f:
        json.dump({"status":"PRIVATE_REVIEW_ONLY_NO_SOURCE_ASSIGNMENTS",
                   "original_index_sha256":sha256(s_bytes).hexdigest(),
                   "records":details},f,indent=2);f.write("\n")
    with PUBLIC_OUT.open("x",encoding="utf-8") as f:
        json.dump(public,f,indent=2);f.write("\n")
    print(json.dumps(public,indent=2))


if __name__=="__main__":main()
