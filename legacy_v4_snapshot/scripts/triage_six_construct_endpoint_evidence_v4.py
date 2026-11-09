"""Privacy-preserving source-method triage for six fixed assay-review tasks.

The actual source/target/assay/registry annotation mapping is PRIVATE at
data/audit/SIX_TASK_METHOD_ANNOTATIONS_PRIVATE_20261109.json. This public
program reads only review TASK METADATA, never original numerical Kd labels,
compounds, sequences, model predictions or prospective-test outcomes.

Registry target "D"/"H" is a database annotation, NOT an independently
verified physical construct. No source/construct/assay is approved here.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import csv
import json

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_REVIEW_QUEUE=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
PRIOR_SIX=ROOT/"research/source_access_v4/six_task_publication_evidence.json"
PRIVATE_ANNOTATIONS=ROOT/"data/audit/SIX_TASK_METHOD_ANNOTATIONS_PRIVATE_20261109.json"
PRIVATE_OUT=ROOT/"data/audit/SIX_TASK_CONSTRUCT_ENDPOINT_METHOD_TRIAGE_PRIVATE_20261109.json"
PUBLIC_OUT=ROOT/"research/six_task_construct_endpoint_method_triage_20261109.json"


def load_private_annotations(path=PRIVATE_ANNOTATIONS):
    file=json.loads(Path(path).read_text(encoding="utf-8"))
    if file.get("status")!="PRIVATE_UNREVIEWED_PUBLIC_METHOD_ANNOTATION_ONLY":
        raise PermissionError("Source annotations are not unreviewed")
    if file.get("scientifically_approved_tasks")!=0:
        raise PermissionError("Private method annotations cannot approve studies")
    entries=file.get("annotations")
    if not isinstance(entries,list) or len(entries)!=6:
        raise ValueError("Expected exactly six private annotations")
    return entries


def assess(entries,original,prior):
    """Pure metadata-only review; output approvals unconditionally FALSE."""
    if len(entries)!=6 or len({x["assay_key"] for x in entries})!=6:
        raise ValueError("Exactly six distinct fixed source method leads required")
    if prior.get("status")!="SIX_TASKS_EXTERNAL_PRIMARY_SOURCE_EVIDENCE_NO_APPROVAL":
        raise PermissionError("Previous original publication leads are not unapproved")
    if prior.get("source_approved_task_count")!=0:
        raise PermissionError("Previous source review has been modified")
    if len(original)!=30 or len({x["task_id"] for x in original})!=30:
        raise ValueError("Canonical human source review queue has changed")
    if any(x.get("curation_decision")!="UNREVIEWED" for x in original):
        raise PermissionError("Cannot replace scientific reviewer decisions")
    if len(prior["records"])!=6 or prior.get("candidate_record_count")!=147:
        raise ValueError("Prior source lead evidence count changed")
    by_key={(r["target_names"],r["assay_key"]):r for r in original}
    by_prior={(r["target"],r["assay_key"]):r for r in prior["records"]}
    if len(by_key)!=30 or len(by_prior)!=6:
        raise ValueError("Duplicate source identity")
    output=[]
    for x in entries:
        if set(x)!={
            "assay_key","target","annotation","target_relationship",
            "construct_description","endpoint_indicator","primary_source_status",
            "source_methods_risk","registry_assay_reference"
        }:
            raise ValueError("Unexpected source evidence schema")
        key=(x["target"],x["assay_key"])
        if key not in by_key or key not in by_prior:
            raise ValueError("Method evidence does not match frozen original task")
        record=by_key[key]
        prev=by_prior[key]
        if (record["task_id"]!=prev["task_id"]
            or int(record["record_count"])!=prev["source_record_count"]
            or record.get("curation_decision")!="UNREVIEWED"):
            raise ValueError("Original task and source record witness mismatch")
        if x["target_relationship"] not in ("D","H","UNKNOWN"):
            raise ValueError("Invalid external registry target-relationship flag")
        if x["target_relationship"]=="H" and "HOMOLOGOUS" not in x["source_methods_risk"]:
            raise ValueError("Homologous registry assignment must be flagged as risk")
        if "KINASE_DOMAIN" in x["construct_description"] and (
            "DOMAIN" not in x["source_methods_risk"]
            and "HOMOLOGOUS" not in x["source_methods_risk"]
        ):
            raise ValueError("Kinase-domain source cannot silently become full length")
        if "CENSORING_PRESENT" in x["endpoint_indicator"] and (
            "KD" not in x["endpoint_indicator"]
        ):
            raise ValueError("Unknown original binding assay endpoint")
        output.append({
            **x,"task_id":record["task_id"],
            "source_record_count_from_prior_review":int(record["record_count"]),
            "scientifically_approved":False,
            "exact_source_record_Kd_crosswalk_verified":False,
            "individual_assayed_protein_construct_verified":False,
            "full_length_sequence_as_construct_authorized":False,
            "training_or_reserved_test_scoring_authorized":False,
            "review_status":"SOURCE_METHOD_METADATA_TRIAGE_ONLY",
        })
    if sum(r["source_record_count_from_prior_review"] for r in output)!=147:
        raise ValueError("Prior source row count changed")
    counts=Counter(r["target_relationship"] for r in output)
    if counts!={"D":1,"H":2,"UNKNOWN":3}:
        raise ValueError("Fixed manual registry categories changed")
    return output,{
        "status":"PUBLIC_METHOD_METADATA_TRIAGE_ZERO_ASSAY_APPROVAL",
        "fixed_original_review_tasks":6,
        "original_candidate_source_record_count":147,
        "target_relationship_annotation_counts":{
            "direct_registry":1,"homologous_registry":2,"unknown_or_unverified":3},
        "source_metadata_indicating_kinase_domain_assay_tasks":3,
        "original_published_kinase_table_with_censoring_warning_tasks":3,
        "SPR_recombinant_protein_public_registry_described_tasks":1,
        "NMR_conditional_method_requires_assay_comparability_review_tasks":1,
        "review_citation_not_primary_assay_tasks":1,
        "tasks_with_exact_original_source_row_crosswalk_verified":0,
        "physically_assayed_protein_constructs_sequence_verified":0,
        "scientifically_approved_tasks":0,
        "full_length_sequence_assay_constructs_authorized":0,
        "training_or_prospective_test_scoring_authorized":False,
        "scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "limitations":[
            "D/H/UNKNOWN describes a public registry target relationship; D does not prove a source-matched physically assayed protein sequence and H requires additional scrutiny.",
            "A published kinase Kd matrix may contain censored/blank values, which cannot be silently substituted as exact binding constants.",
            "Original-source task metadata and publication titles are not original experimental table crosswalks or independent source/construct validation.",
            "The detailed six-task/source-annotation mapping remains confidential; the public projection contains only aggregate method risk counts.",
            "No raw Kd/pKd labels, measured drug affinities, model predictions, training or reserved-test outcomes were accessed.",
        ],
    }


def main():
    if PRIVATE_OUT.exists() or PUBLIC_OUT.exists():
        raise FileExistsError("Refusing to overwrite immutable source triage")
    original_bytes=ORIGINAL_REVIEW_QUEUE.read_bytes()
    prior_bytes=PRIOR_SIX.read_bytes()
    annotation_bytes=PRIVATE_ANNOTATIONS.read_bytes()
    with ORIGINAL_REVIEW_QUEUE.open(encoding="utf-8-sig",newline="") as f:
        original=list(csv.DictReader(f))
    prior=json.loads(prior_bytes)
    if sha256(original_bytes).hexdigest()!=prior.get("original_review_csv_sha256"):
        raise PermissionError("Original source queue changed since fixed publication leads")
    entries=load_private_annotations()
    detail,safe=assess(entries,original,prior)
    if (
        original_bytes!=ORIGINAL_REVIEW_QUEUE.read_bytes()
        or prior_bytes!=PRIOR_SIX.read_bytes()
        or annotation_bytes!=PRIVATE_ANNOTATIONS.read_bytes()
    ):
        raise PermissionError("Private source review evidence changed during triage")
    local={
        "status":"PRIVATE_PUBLIC_SOURCE_METHOD_TRIAGE_NOT_APPROVED",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "original_review_csv_sha256":sha256(original_bytes).hexdigest(),
        "prior_six_leads_sha256":sha256(prior_bytes).hexdigest(),
        "private_annotation_sha256":sha256(annotation_bytes).hexdigest(),
        "source_evidence_objects":detail,
        "public_summary":safe
    }
    with PRIVATE_OUT.open("x",encoding="utf-8") as f:
        json.dump(local,f,indent=2);f.write("\n")
    with PUBLIC_OUT.open("x",encoding="utf-8") as f:
        json.dump(safe,f,indent=2);f.write("\n")
    print(json.dumps(safe,indent=2))


if __name__=="__main__":
    main()
