"""Europe PMC DOI/PMID metadata reconciliation for seven PRIVATE conflicted source families.

NO affinity values, assay numbers, chemical structures, protein sequences,
training labels, provisional split roles, predictions, or held-out outcomes
are accessed by this program. It reads only the immutable PRIVATE
bibliographic identifier index and queries official PubMed metadata.

Detailed DOI/PMID/publication identities and family links stay PRIVATE in
data/audit. Public reports aggregate counts only, never names, IDs or roles.
A verified PMID<->DOI publication identity is NOT verified assay provenance,
construct identity, endpoint, or cohort approval. A patent is not assumed
equal to a PubMed article without independent evidence.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json
import re
import urllib.parse

from scripts.audit_primary_source_access_v4 import fetch,API

ROOT=Path(__file__).resolve().parents[1]
PRIVATE_INPUT=ROOT/"data/audit/SOURCE_FAMILY_COMPLETE_ELIGIBLE_PROVENANCE_PRIVATE_20261108.json"
PRIVATE_OUTPUT=ROOT/"data/audit/SEVEN_FAMILY_BIBLIOGRAPHIC_ALIAS_VERIFICATION_PRIVATE_20261109.json"
PUBLIC_OUTPUT=ROOT/"research/bindingdb_seven_family_bibliographic_alias_20261109.json"
BATCH_SIZE=8
WORKERS=3


def normalize_doi(value):
    if not isinstance(value,str):
        return ""
    v=urllib.parse.unquote(value).strip().lower().removeprefix("doi:")
    for prefix in ("https://doi.org/","http://doi.org/","doi.org/"):
        if v.startswith(prefix): v=v[len(prefix):]
    return v if v.startswith("10.") and "/" in v and not any(c.isspace() for c in v) else ""


def verified_med_record(record):
    if str(record.get("source") or "").upper()!="MED":
        return None
    pid=str(record.get("pmid") or record.get("id") or "")
    if not re.fullmatch(r"[0-9]{6,9}",pid):
        return None
    return {
        "pmid":pid,
        "doi":normalize_doi(record.get("doi")),
        "pmcid":str(record.get("pmcid") or "") or None,
        "open_access":record.get("isOpenAccess")=="Y",
        "first_publication_year":str(record.get("firstPublicationDate") or "")[:4],
    }


def batches(items,size=BATCH_SIZE):
    return [items[i:i+size] for i in range(0,len(items),size)]


def exact_matches(kind,query_values,records):
    """Ignore fuzzy API hits, non-MED sources and malformed metadata."""
    grouped={item:[] for item in query_values}
    for raw in records:
        rec=verified_med_record(raw)
        if rec is None:
            continue
        term=rec["pmid"] if kind=="pmid" else rec["doi"]
        if term in grouped and rec not in grouped[term]:
            grouped[term].append(rec)
    return grouped


def query_med_batch(kind,items):
    if kind not in ("pmid","doi"):
        raise ValueError("Only exact PMID/DOI search is supported")
    if not items:
        raise ValueError("Empty bibliography query batch")
    query_tokens=[
        "EXT_ID:"+id if kind=="pmid" else 'DOI:"'+id.replace('"',"")+'"'
        for id in items
    ]
    expr="("+ " OR ".join(query_tokens) +") AND SRC:MED"
    qs=urllib.parse.urlencode({
        "query":expr,"format":"json","resultType":"lite","pageSize":100
    })
    raw=fetch(API+"search?"+qs,limit=2*1024*1024)
    response=json.loads(raw.decode("utf-8"))
    if int(response.get("hitCount",0))>100:
        raise ValueError("Exact bibliography batch gave too many hits")
    found=response.get("resultList",{}).get("result",[])
    if not isinstance(found,list):
        raise ValueError("Malformed Europe PMC bibliography response")
    return kind,items,exact_matches(kind,items,found)


def load_private_identifiers(path=PRIVATE_INPUT):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status")!="CONFIDENTIAL_COMPLETE_ELIGIBLE_SOURCE_IDS_NO_APPROVAL":
        raise PermissionError("Source indexing input is not the frozen unapproved inventory")
    families=data.get("families")
    if not isinstance(families,list) or len(families)!=7:
        raise ValueError("Expected exactly seven conflicted source families")
    if len({f["source_family_sha256"] for f in families})!=7:
        raise ValueError("Repeated family identity")
    all_ids=[src["original_source_identifier"]
             for fam in families for src in fam["all_original_source_identifiers"]]
    counts=Counter(x.split(":",1)[0] for x in all_ids)
    if counts!={"doi":74,"pmid":90,"patent":56}:
        raise ValueError("Unexpected or tampered original bibliography totals")
    if len(all_ids)!=220 or len(all_ids)!=len(set(all_ids)):
        raise ValueError("Original identifier-to-family assignment changed")
    if sum(len(f["technically_eligible_tasks_missing_all_DOI_PMID_patent"])
           for f in families)!=62:
        raise ValueError("Original unattributed technical-task count changed")
    for src in all_ids:
        kind,val=src.split(":",1)
        if kind=="pmid" and not re.fullmatch(r"[0-9]{6,9}",val):
            raise ValueError("Malformed source PMID")
        if kind=="doi" and not normalize_doi(val):
            raise ValueError("Malformed source DOI")
    return families


def reconcile(families, pmid_results,doi_results):
    # Lookup results are prefiltered EXACT PMID and EXACT canonical DOI.
    reviewed=[]
    index_family={}
    all_doi=set()
    all_pmid=set()
    for fam in families:
        key=fam["source_family_sha256"]
        sources={s["original_source_identifier"] for s in fam["all_original_source_identifiers"]}
        dois={s[4:] for s in sources if s.startswith("doi:")}
        pmids={s[5:] for s in sources if s.startswith("pmid:")}
        all_doi.update(dois);all_pmid.update(pmids)
        for item in sources:
            index_family[item]=key
        reviewed.append((key,dois,pmids))
    if len(all_doi)!=74 or len(all_pmid)!=90:
        raise ValueError("Identifier coverage changed during reconciliation")

    pmid_one={}
    doi_one={}
    for pm in sorted(all_pmid):
        found=pmid_results.get(pm,[])
        pmid_one[pm]=found[0] if len(found)==1 else None
    for doi in sorted(all_doi):
        found=doi_results.get(doi,[])
        doi_one[doi]=found[0] if len(found)==1 else None

    verified_pairs=[]
    one_way=[]
    cross_family_candidate=[]
    for family,dois,pmids in reviewed:
        family_verified=[]
        family_one_way=[]
        family_cross=[]
        for pmid in sorted(pmids):
            result=pmid_one[pmid]
            if not result or not result["doi"]:
                continue
            claimed=result["doi"]
            if claimed in dois:
                doi_response=doi_one.get(claimed)
                if doi_response and doi_response["pmid"]==pmid:
                    family_verified.append({
                        "doi":claimed,"pmid":pmid,"evidence":"EXACT_RECIPROCAL_EUROPE_PMC_MED_METADATA"
                    })
                else:
                    family_one_way.append({
                        "doi":claimed,"pmid":pmid,
                        "evidence":"PMID_TO_DOI_ONLY_OR_RECIPROCAL_CONFLICT"
                    })
            elif "doi:"+claimed in index_family and index_family["doi:"+claimed]!=family:
                family_cross.append({
                    "doi":claimed,"pmid":pmid,
                    "other_family_sha256":index_family["doi:"+claimed],
                    "evidence":"CROSS_PRELIMINARY_FAMILY_BIBLIOGRAPHIC_LINK_REVIEW_REQUIRED"
                })
        verified_pairs.extend(family_verified)
        one_way.extend(family_one_way)
        cross_family_candidate.extend(family_cross)
    if len(verified_pairs)!=len({(x["doi"],x["pmid"]) for x in verified_pairs}):
        raise ValueError("Duplicate publication alias from multiple families")
    verified_dois={x["doi"] for x in verified_pairs}
    verified_pmids={x["pmid"] for x in verified_pairs}
    if len(verified_dois)!=len(verified_pairs) or len(verified_pmids)!=len(verified_pairs):
        # This conservative graph counts only one-to-one publication IDs.
        raise ValueError("DOI/PMID mapping non-bijective; manual verification required")

    groups=[]
    for family,dois,pmids in reviewed:
        family_pairs=[x for x in verified_pairs if x["doi"] in dois and x["pmid"] in pmids]
        groups.append({
            "source_family_sha256":family,
            "verification_status":"BIBLIOGRAPHY_ONLY_NO_ASSAY_OR_CONSTRUCT_APPROVAL",
            "original_DOI_strings":sorted(dois),
            "original_PMID_strings":sorted(pmids),
            "verified_reciprocal_DOI_PMID_aliases":family_pairs,
            "PMID_records_metadata_only":[
                {"original_pmid":p,"status":(
                     "EXACT_PUBMED_METADATA" if pmid_one[p] is not None
                     else "UNRESOLVED_OR_AMBIGUOUS_METADATA"),
                 "reported_doi":pmid_one[p]["doi"] if pmid_one[p] else None,
                 "open_access":pmid_one[p]["open_access"] if pmid_one[p] else None,
                 "pmcid":pmid_one[p]["pmcid"] if pmid_one[p] else None}
                for p in sorted(pmids)
            ],
            "DOI_records_metadata_only":[
                {"original_doi":v,
                 "status":"EXACT_MED_METADATA" if doi_one[v] is not None
                          else "NOT_FOUND_OR_AMBIGUOUS_IN_MED",
                 "reported_pmid":doi_one[v]["pmid"] if doi_one[v] else None}
                for v in sorted(dois)
            ],
            "one_direction_only_alias_leads":[x for x in one_way if x["doi"] in dois],
            "cross_family_link_leads":[x for x in cross_family_candidate if x["pmid"] in pmids],
            "approved_assays":0,"verified_physical_constructs":0,
        })
    public={
        "status":"SEVEN_FAMILY_BIBLIOGRAPHIC_RECONCILIATION_ONLY_NO_ASSAY_APPROVAL",
        "original_conflicted_source_families":7,
        "original_eligible_tasks_in_seven_families":708,
        "original_unattributed_technical_tasks":62,
        "original_source_identifier_strings":220,
        "original_source_identifier_counts_by_kind":{"doi":74,"pmid":90,"patent":56},
        "doi_identifiers_with_exact_MED_metadata":sum(v is not None for v in doi_one.values()),
        "pmid_identifiers_with_exact_MED_metadata":sum(v is not None for v in pmid_one.values()),
        "reciprocal_DOI_PMID_aliases_confirmed_as_same_publication":len(verified_pairs),
        "one_way_only_DOI_PMID_alias_leads_not_promoted":len(one_way),
        "cross_preliminary_family_bibliography_link_leads_require_review":len(cross_family_candidate),
        "unresolved_DOI_identifiers_in_MED":sum(v is None for v in doi_one.values()),
        "unresolved_PMID_identifiers_in_MED":sum(v is None for v in pmid_one.values()),
        "original_patent_strings_unverified_by_pubmed":56,
        "original_assay_record_crosswalks_verified":0,
        "assayed_physical_constructs_verified":0,
        "families_scientifically_approved":0,
        "scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "warnings":[
            "A reciprocal DOI–PMID lookup verifies only bibliographic identity in Europe PMC/MED; it does NOT match BindingDB source assay rows or physical protein constructs.",
            "Neither a DOI nor a PMID guarantees a primary original experiment; some publications may be reviews, screening panels or indirect assays.",
            "Patent IDs are deliberately left as distinct unresolved sources; do not merge them with DOI/PMID without original evidence.",
            "All individual source IDs, family IDs and detailed crosswalks are confined to the private reviewer artifact, never public.",
            "No affinity values, model outcomes, reserved-test metrics, split-role assignments or compound identities were read or changed.",
            "No scientific training/cohort approval or new split freeze is authorized by these results.",
        ],
    }
    return groups,public


def main():
    if PRIVATE_OUTPUT.exists() or PUBLIC_OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite immutable bibliography review")
    before=PRIVATE_INPUT.read_bytes()
    families=load_private_identifiers()
    pmids=sorted({src["original_source_identifier"][5:]
        for fam in families for src in fam["all_original_source_identifiers"]
        if src["original_source_identifier"].startswith("pmid:")})
    dois=sorted({src["original_source_identifier"][4:]
        for fam in families for src in fam["all_original_source_identifiers"]
        if src["original_source_identifier"].startswith("doi:")})
    tasks=[(kind,block) for kind,items in (("pmid",pmids),("doi",dois))
           for block in batches(items)]
    fetched={"pmid":{},"doi":{}}
    errors=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(query_med_batch,kind,block):(kind,block)
                 for kind,block in tasks}
        for future in as_completed(futures):
            kind,block=futures[future]
            try:
                _,items,found=future.result()
                fetched[kind].update(found)
                print(json.dumps({
                    "metadata_batch":kind,"requested":len(items),
                    "exact_europe_pmc_hits":sum(len(v)==1 for v in found.values()),
                    "status":"COMPLETE"
                }),flush=True)
            except Exception as ex:
                errors.append((kind,len(block),type(ex).__name__))
                fetched[kind].update({x:[] for x in block})
                print(json.dumps({
                    "metadata_batch":kind,"requested":len(block),
                    "status":"UNRESOLVED_API_ERROR","error_type":type(ex).__name__
                }),flush=True)
    if sha256(PRIVATE_INPUT.read_bytes()).digest()!=sha256(before).digest():
        raise PermissionError("Source crosswalk changed while public metadata was fetched")
    detail,public=reconcile(families,fetched["pmid"],fetched["doi"])
    public["europe_pmc_exact_metadata_batches_total"]=len(tasks)
    public["europe_pmc_metadata_batches_unresolved_by_network"]=len(errors)
    public["metadata_fetch_errors_by_type"]=dict(sorted(Counter(e[2] for e in errors).items()))
    public["original_private_bibliographic_inventory_sha256"]=sha256(before).hexdigest()
    private={
        "status":"PRIVATE_RECIPROCAL_METADATA_SOURCE_REVIEW_NO_ASSAY_APPROVAL",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "source_inventory_sha256":sha256(before).hexdigest(),
        "metadata_query_provider":"Europe PMC MED lite exact PMID and exact DOI",
        "unresolved_API_batches_metadata_only":errors,
        "families":detail,
        "public_projection":public
    }
    with PRIVATE_OUTPUT.open("x",encoding="utf-8") as f:
        json.dump(private,f,indent=2)
        f.write("\n")
    with PUBLIC_OUTPUT.open("x",encoding="utf-8") as f:
        json.dump(public,f,indent=2)
        f.write("\n")
    print(json.dumps(public,indent=2),flush=True)


if __name__=="__main__":main()
