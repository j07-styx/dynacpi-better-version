"""Fetch public publication metadata for the 30 BindingDB review tasks.

No affinity values read. Europe PMC metadata can confirm an indexed paper's
title/DOI but NOT its assay conditions or protein construct. Patent records are
listed for human primary-source inspection, not automatically verified.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"data/audit"
QUEUE=AUDIT/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
OUT=AUDIT/"bindingdb_assay_review_30_publication_screen.json"
CSV=AUDIT/"bindingdb_assay_review_30_publication_screen.csv"
CACHE=ROOT/"data/reference/europepmc"
CACHE.mkdir(parents=True,exist_ok=True)

def all_values(s):
    return sorted(set(x.strip() for x in (s or "").split("|") if x.strip()))

def get_pubmed(pmid):
    dest=CACHE/(pmid+".json")
    if dest.exists():
        d=json.loads(dest.read_text(encoding="utf-8"))
        if d.get("source_pmid")==pmid:return d
    query=urllib.parse.quote("EXT_ID:"+pmid+" AND SRC:MED")
    url="https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="+query+"&format=json&pageSize=1"
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"DynaCPI-v4-PublicationMetadataAudit/0.1"})
        with urllib.request.urlopen(req,timeout=24) as r:
            d=json.load(r)
        results=d.get("resultList",{}).get("result",[])
        if len(results)!=1 or str(results[0].get("id"))!=pmid:
            result={"source_pmid":pmid,"status":"UNRESOLVED_SOURCE"}
        else:
            t=results[0]
            result={"source_pmid":pmid,"status":"PUBMED_INDEX_FOUND",
                    "title":t.get("title",""),
                    "journal":t.get("journalTitle",""),
                    "year":t.get("pubYear",""),
                    "doi":t.get("doi",""),"authorString":t.get("authorString","")[:250],
                    "pmcid":t.get("pmcid",""),
                    "metadata_url":"https://europepmc.org/article/MED/"+pmid}
        dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
        return result
    except Exception as exc:
        return {"source_pmid":pmid,"status":"METADATA_FETCH_ERROR",
                "error":type(exc).__name__+": "+str(exc)[:150]}

def method_flags(name,description):
    s=(name+" "+description).lower()
    flags=[]
    terms=[
        ("ITC_MENTION",["isothermal titration calorimetry","itc200","microcal itc"]),
        ("SPR_MENTION",["surface plasmon resonance","biacore","spr interaction"]),
        ("FLUORESCENCE_POLARIZATION_MENTION",["fluorescence polarization"]),
        ("COMPETITION_ASSAY_MENTION",["competitive binding","competition binding","competing with","displacement"]),
        ("INHIBITION_ASSAY_MENTION",["inhibition assay","inhibition constant","enzyme inhibition","ic50"]),
        ("DISSOCIATION_MENTION",["dissociation constant"," kd"," k_d"," k(d)","binding constant"]),
        ("KINETICS_MENTION",["kinetic binding","association rate","dissociation rate","kon","koff"]),
        ("UNKNOWN_METHOD",[])
    ]
    for k,v in terms[:-1]:
        if any(word in s for word in v):flags.append(k)
    if not flags:flags.append("UNKNOWN_METHOD")
    return flags

def main():
    with QUEUE.open(encoding="utf-8-sig",newline="") as f:
        tasks=list(csv.DictReader(f))
    if len(tasks)!=30:raise RuntimeError("Unexpected review cohort size")
    refs=sorted(set(x for t in tasks for x in all_values(t["pmids"]) if re.fullmatch(r"\d{4,10}",x)))
    results={}
    with ThreadPoolExecutor(max_workers=4) as ex:
        pending={ex.submit(get_pubmed,x):x for x in refs}
        for fut in as_completed(pending):
            result=fut.result()
            results[pending[fut]]=result
    output=[];counts=Counter()
    for t in tasks:
        pmids=all_values(t["pmids"])
        dois=all_values(t["article_dois"])
        patents=all_values(t["patent_numbers"])
        public=[results.get(p,{"source_pmid":p,"status":"MISSING"}) for p in pmids]
        sources_with_doi=[p for p in public if p.get("doi")]
        matches=any(p["doi"].casefold() in {d.casefold() for d in dois} for p in sources_with_doi)
        mismatch=bool(dois and sources_with_doi and not matches)
        flags=method_flags(t["assay_name"],t["assay_description"])
        for flag in flags:counts[flag]+=1
        if mismatch:counts["DOI_MISMATCH_REVIEW"]+=1
        if not pmids and not dois and patents:counts["PATENT_ONLY"]+=1
        if any(p["status"]=="METADATA_FETCH_ERROR" for p in public):counts["FETCH_ERRORS"]+=1
        entry={
            "task_id":t["task_id"],"assay_key":t["assay_key"],"protein_component_id":t["protein_component_id"],
            "assay_name":t["assay_name"],"data_sources":t["data_sources"],
            "method_text_flags":flags,
            "bindingdb_article_dois":dois,"bindingdb_pmids":pmids,"patents":patents,
            "indexed_publications":public,"doi_mismatch_review_flag":mismatch,
            "publication_identification_status":
                "PUBMED_INDEXED" if any(p["status"]=="PUBMED_INDEX_FOUND" for p in public)
                else ("PATENT_PRIMARY_SOURCE_REQUIRED" if patents
                      else "NO_INDEXED_PUBLICATION_CONFIRMED"),
            "scientific_assay_approved":False
        }
        counts["tasks"]+=1
        output.append(entry)
    report={"status":"SOURCE_METADATA_TRIAGE_ONLY_ZERO_ASSAYS_APPROVED",
            "retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
            "unique_pubmed_ids_requested":len(refs),
            "unique_pubmed_ids_indexed":sum(v["status"]=="PUBMED_INDEX_FOUND" for v in results.values()),
            "counts":dict(counts),
            "tasks":output,
            "limitations":[
                "Publication metadata and method words cannot prove the measured value is exact Kd.",
                "DOI mismatches require human evaluation and may reflect source aggregation.",
                "Patent-only tasks require independent access to the original patent disclosures.",
                "Study may use a construct/variant different from BindingDB's reference sequence.",
                "No source has been adjudicated and no affinity values were used."]}
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    fields=["task_id","assay_key","data_sources","assay_name","method_text_flags",
            "publication_identification_status","article_dois","pmids","patents",
            "indexed_titles","indexed_dois","metadata_links","doi_mismatch_review_flag","reviewer_decision"]
    if CSV.exists():
        raise FileExistsError("Refusing to overwrite possibly reviewed publication sheet: "+str(CSV))
    with CSV.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for t in output:
            ps=t["indexed_publications"]
            writer.writerow({
                "task_id":t["task_id"],"assay_key":t["assay_key"],"data_sources":t["data_sources"],
                "assay_name":t["assay_name"],"method_text_flags":";".join(t["method_text_flags"]),
                "publication_identification_status":t["publication_identification_status"],
                "article_dois":" | ".join(t["bindingdb_article_dois"]),
                "pmids":" | ".join(t["bindingdb_pmids"]),"patents":" | ".join(t["patents"]),
                "indexed_titles":" | ".join(p.get("title","") for p in ps),
                "indexed_dois":" | ".join(p.get("doi","") for p in ps),
                "metadata_links":" | ".join(p.get("metadata_url","") for p in ps),
                "doi_mismatch_review_flag":t["doi_mismatch_review_flag"],
                "reviewer_decision":"UNREVIEWED"})
    print(json.dumps({k:v for k,v in report.items() if k not in ("tasks","limitations")},indent=2),flush=True)
if __name__=="__main__":main()
