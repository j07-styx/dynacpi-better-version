"""Single DOI absent from Europe PMC MED: Crossref bibliographic fallback.

Reads PRIVATE identifier lookup summary only. Resolves a DOI's publication
metadata through Crossref exact DOI endpoint, WITHOUT ever reading original
BindingDB affinity values, query labels or provisional study roles.

Both the source identifier and bibliographic article metadata remain private.
The public output contains only aggregate access resolution and explicit
no-assay-approval status. An article DOI does not verify any experimental Kd.
"""
from __future__ import annotations

from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json
import urllib.parse
import urllib.request

from scripts.reconcile_seven_family_bibliographic_aliases_v4 import (
    normalize_doi, PRIVATE_OUTPUT, PUBLIC_OUTPUT,
)

ROOT=Path(__file__).resolve().parents[1]
PRIVATE_NEW=ROOT/"data/audit/SEVEN_FAMILY_CROSSREF_DOI_EXCEPTION_PRIVATE_20261109.json"
PUBLIC_NEW=ROOT/"research/bindingdb_seven_family_crossref_exception_20261109.json"


def exact_crossref_metadata(requested_doi, response):
    if not isinstance(response,dict):
        raise ValueError("Malformed Crossref works response")
    message=response.get("message")
    if not isinstance(message,dict):
        raise ValueError("Malformed Crossref works message")
    cross_doi=normalize_doi(message.get("DOI"))
    if not requested_doi or cross_doi!=normalize_doi(requested_doi):
        raise ValueError("Crossref DOI did not exactly match the unresolved source DOI")
    title=message.get("title")
    if not isinstance(title,list) or not title or not isinstance(title[0],str):
        raise ValueError("Exact Crossref entry has no article title")
    return {
        "exact_crossref_doi_match":True,
        "source_kind":str(message.get("type") or "unknown"),
        "title":title[0][:350],
        "publisher":str(message.get("publisher") or "")[:120],
        "has_volume":bool(message.get("volume")),
        "review_stage":"BIBLIOGRAPHIC_IDENTITY_ONLY_NO_ASSAY_CONSTRUCT_OR_ENDPOINT_APPROVAL",
    }


def unresolved_dois(private_report,public_report):
    if private_report.get("status")!="PRIVATE_RECIPROCAL_METADATA_SOURCE_REVIEW_NO_ASSAY_APPROVAL":
        raise PermissionError("Private source-alias review not in quarantined state")
    if public_report.get("status")!="SEVEN_FAMILY_BIBLIOGRAPHIC_RECONCILIATION_ONLY_NO_ASSAY_APPROVAL":
        raise PermissionError("Public companion report is not the immutable source-only snapshot")
    if public_report.get("unresolved_DOI_identifiers_in_MED")!=1:
        raise ValueError("Expected exactly one PubMed-unindexed DOI from original audit")
    unresolved=[]
    for family in private_report["families"]:
        for record in family["DOI_records_metadata_only"]:
            if record.get("status")!="EXACT_MED_METADATA":
                unresolved.append((family["source_family_sha256"],record["original_doi"]))
    if len(unresolved)!=1:
        raise ValueError("Private and public PubMed exception counts disagree")
    return unresolved


def query_crossref(doi):
    url="https://api.crossref.org/works/"+urllib.parse.quote(doi,safe="/")
    req=urllib.request.Request(url,headers={
        "User-Agent":"DynaCPI-v4-bibliographic-metadata-only/0.1",
        "Accept":"application/json",
    })
    with urllib.request.urlopen(req,timeout=18) as res:
        raw=res.read(512000+1)
    if len(raw)>512000:
        raise ValueError("Crossref metadata response exceeded safety size limit")
    return json.loads(raw.decode("utf-8"))


def main():
    if PRIVATE_NEW.exists() or PUBLIC_NEW.exists():
        raise FileExistsError("Refusing to overwrite existing Crossref metadata review")
    source_bytes=PRIVATE_OUTPUT.read_bytes()
    public_bytes=PUBLIC_OUTPUT.read_bytes()
    private=json.loads(source_bytes)
    public=json.loads(public_bytes)
    [(family,doi)]=unresolved_dois(private,public)
    response=query_crossref(doi)
    found=exact_crossref_metadata(doi,response)
    if source_bytes!=PRIVATE_OUTPUT.read_bytes() or public_bytes!=PUBLIC_OUTPUT.read_bytes():
        raise PermissionError("Source identity reports changed during Crossref lookup")
    local={
        "status":"PRIVATE_CROSSREF_BIBLIOGRAPHIC_IDENTITY_ONLY_NO_ASSAY_APPROVAL",
        "source_alias_snapshot_sha256":sha256(source_bytes).hexdigest(),
        "verified_utc":datetime.now(timezone.utc).isoformat(),
        "source_family_sha256":family,
        "original_doi":doi,
        "metadata":found,
        "bindingdb_records_verified":0,
        "physical_construct_verified":False,
        "scientific_approval":False,
    }
    safe={
        "status":"CROSSREF_ONE_PUBMED_UNINDEXED_DOI_BIBLIOGRAPHY_ONLY",
        "pubmed_unindexed_doi_checked":1,
        "exact_crossref_DOI_identifiers_verified":1,
        "crossref_source_type_counts":{found["source_kind"]:1},
        "additional_bibliographic_DOI_identifiers_resolved":1,
        "original_DOI_total":74,
        "original_pubmed_DOI_identifiers_resolved":73,
        "original_pubmed_PMID_identifiers_resolved":90,
        "reciprocal_pubmed_DOI_PMID_alias_pairs_verified":72,
        "original_unattributed_technical_tasks":62,
        "patent_identifiers_unresolved":56,
        "bindingdb_assay_rows_verified":0,
        "physical_constructs_verified":0,
        "families_scientifically_approved":0,
        "scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "interpretation":"Exactly one original DOI not indexed in the Europe PMC MED search was separately resolved as an exact Crossref bibliographic record. This does not assign any PMID or verify a direct affinity assay.",
        "privacy":"DOI, source-family identity and article title remain solely in private reviewer report.",
    }
    with PRIVATE_NEW.open("x",encoding="utf-8") as f:
        json.dump(local,f,indent=2)
        f.write("\n")
    with PUBLIC_NEW.open("x",encoding="utf-8") as f:
        json.dump(safe,f,indent=2)
        f.write("\n")
    print(json.dumps(safe,indent=2))


if __name__=="__main__":main()
