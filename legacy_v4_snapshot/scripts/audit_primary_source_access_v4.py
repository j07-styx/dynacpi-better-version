"""Read-only, source-identification aid for DynaCPI 30-task review queue.

Collect official Europe PMC bibliographic metadata and OA METHOD SECTION IDs
only. Never copy full paper bodies into GitHub, infer a source-verified exact
Kd, or modify reviewer decisions. NO affinity labels or molecules processed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
OUT=ROOT/"research/source_access_v4"
API="https://www.ebi.ac.uk/europepmc/webservices/rest/"
USER_AGENT="DynaCPI-v4-publication-metadata-evidence/0.1"
MAX_BYTES=8*1024*1024
TERMS={
 "KD":re.compile(r"\bK[\s_]?d\b|dissociation constant",re.I),
 "ITC":re.compile(r"\bITC\b|isothermal titration calorimetry",re.I),
 "SPR":re.compile(r"\bSPR\b|surface plasmon resonance",re.I),
 "BLI":re.compile(r"\bBLI\b|biolayer interferometry",re.I),
 "CONSTRUCT":re.compile(r"construct|recombinant|His.tag|GST.tag|protein expression",re.I),
 "SEQUENCE_REGION":re.compile(r"residues?|protein domain|truncat",re.I),
 "OTHER_ENDPOINT":re.compile(r"\bIC\s?50\b|\bK[\s_]?i\b|inhibition",re.I)
}
def fingerprint(data):
 return hashlib.sha256(data).hexdigest()
def read_queue():
 with QUEUE.open(encoding="utf-8-sig",newline="") as f:r=list(csv.DictReader(f))
 if len(r)!=30 or len({x["task_id"] for x in r})!=30:
  raise RuntimeError("Review queue must have exactly 30 distinct tasks")
 if any(x.get("curation_decision","").upper().strip()!="UNREVIEWED" for x in r):
  raise PermissionError("Review queue already adjudicated; manual decisions are protected")
 return r
def pmids(row):
 return sorted(set(re.findall(r"(?<!\d)\d{6,9}(?!\d)",row.get("pmids",""))))
def fetch(url,limit=MAX_BYTES):
 last=None
 for retry in range(3):
  try:
   req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT})
   with urllib.request.urlopen(req,timeout=25) as conn:
    buf=conn.read(limit+1)
   if len(buf)>limit:raise ValueError("API response over size limit")
   return buf
  except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,OSError) as exc:
   last=exc
   if isinstance(exc,urllib.error.HTTPError) and exc.code in (400,403,404):break
   if retry<2:time.sleep((retry+1)*.5)
 raise RuntimeError(type(last).__name__+": "+str(getattr(last,"code","network")))
def extract_section_signals(body):
 if b"<!DOCTYPE" in body or b"<!ENTITY" in body:
  raise ValueError("Refusing source XML with DTD/entity declarations")
 root=ET.fromstring(body)
 found=[]
 for num,sec in enumerate(root.findall(".//body//sec")):
  t=sec.find("title")
  if t is None:continue
  title=" ".join("".join(t.itertext()).split())[:110]
  # Direct paragraph children only: avoid duplicate hits through nesting.
  section_text=" ".join(" ".join("".join(p.itertext()).split()) for p in sec.findall("./p"))
  hits={k:len(v.findall(section_text)) for k,v in TERMS.items()}
  hits={k:v for k,v in hits.items() if v}
  method_heading=bool(re.search("method|experiment|assay|protein|binding|measurement",title,re.I))
  if not method_heading and not hits:continue
  found.append({"section_id":sec.get("id") or "index_"+str(num),
   "section_heading":title,"method_heading_hint":method_heading,
   "keyword_hit_counts":hits})
 found.sort(key=lambda r:(-sum(r["keyword_hit_counts"].values()),
                           -r["method_heading_hint"],r["section_id"]))
 return {"full_text_sha256":fingerprint(body),
         "relevant_sections_total":len(found),
         "section_headings_and_keyword_counts":found[:24],
         "section_index_truncated":len(found)>24}
def fetch_pub(pmid,metadata_only=False):
 output={"pmid":pmid,"metadata_access":"UNAVAILABLE",
         "fulltext_access":"NOT_CHECKED",
         "approval_status":"NOT_APPROVED_SOURCE_METADATA_ONLY"}
 try:
  query=urllib.parse.urlencode({"query":"EXT_ID:"+pmid+" AND SRC:MED",
                "format":"json","resultType":"core","pageSize":1})
  raw=fetch(API+"search?"+query,2*1024*1024)
  results=json.loads(raw.decode("utf-8")).get("resultList",{}).get("result",[])
  matches=[a for a in results if str(a.get("pmid") or a.get("id"))==pmid]
  if len(matches)!=1:
   output["metadata_access"]="PMID_EXACT_MATCH_MISSING"
   return output
  a=matches[0]
  pmcid=str(a.get("pmcid") or "")
  if pmcid and not re.fullmatch(r"PMC\d+",pmcid):raise ValueError("Malformed PMCID")
  output.update(metadata_access="EXACT_PMID_MATCH",
      doi=a.get("doi"),pmcid=pmcid or None,title=a.get("title"),
      source_publication_year=(a.get("firstPublicationDate") or "")[:4],
      open_access=a.get("isOpenAccess")=="Y",
      europe_pmc_fulltext=a.get("inEPMC")=="Y")
  if not metadata_only and pmcid and output["open_access"] and output["europe_pmc_fulltext"]:
   try:
    xml=fetch(API+pmcid+"/fullTextXML")
    output.update(extract_section_signals(xml))
    output["fulltext_access"]="OPEN_ACCESS_SECTION_INDEXED"
   except (ValueError,RuntimeError,ET.ParseError) as ex:
    output["fulltext_access"]="INDEX_FAILED"
    output["index_error"]=type(ex).__name__+": "+str(ex)[:90]
  else:
   output["fulltext_access"]="NOT_PUBLIC_OPEN_ACCESS" if not metadata_only else "METADATA_ONLY"
 except (ValueError,RuntimeError,TypeError) as ex:
  output.update(metadata_access="FETCH_FAILED",
          fetch_error=type(ex).__name__+": "+str(ex)[:90])
 return output
def make_report(rows,pub_map,queue_sha):
 result=[]
 for row in rows:
  ids=pmids(row)
  linked=[pub_map[p] for p in ids]
  haspatent=bool(row.get("patent_numbers","").strip())
  hasdoi=bool(row.get("article_dois","").strip())
  if not ids and haspatent:status="PRIMARY_PATENT_NEEDED"
  elif any(p.get("fulltext_access")=="OPEN_ACCESS_SECTION_INDEXED" for p in linked):
   status="PUBLIC_OPEN_ACCESS_INDEXED"
  elif ids:status="SOURCE_ARTICLE_ACCESS_PENDING"
  elif hasdoi:status="PRIMARY_DOI_ONLY_METHODS_PENDING"
  else:status="MISSING_PRIMARY_SOURCE_IDENTIFIER"
  d={"task_id":row["task_id"],"assay_key":row["assay_key"],
     "protein_component_id":row["protein_component_id"],
     "pmids":ids,"patent_ids":row.get("patent_numbers","").split(";") if haspatent else [],
     "doi_identifiers":[z.strip() for z in re.split(r"[;,|]",row.get("article_dois","")) if z.strip()],
     "evidence_access_status":status,
     "current_review_status":"UNREVIEWED","verified_construct":False,
     "verified_kd_semantics":False,"verified_assay_compatibility":False}
  result.append(d)
 counts=Counter(r["evidence_access_status"] for r in result)
 pub_count=Counter(x["fulltext_access"] for x in pub_map.values())
 return {"status":"AUTOMATED_SOURCE_INDEX_ONLY_NOT_BIOLOGICAL_VALIDATION",
   "review_file_sha256":queue_sha,
   "generated_utc":datetime.now(timezone.utc).isoformat(),
   "task_count":len(result),"unique_publications":len(pub_map),
   "scientifically_approved_tasks":0,"human_review_queue_unchanged":True,
   "source_status_counts":dict(counts),"publication_access_counts":dict(pub_count),
   "tasks":result,"publications":[pub_map[x] for x in sorted(pub_map)],
   "limitations":[
    "Section headings and keyword counts only show where to inspect primary-source evidence.",
    "No actual experimental construct, Kd versus Ki/IC50 semantics, or assay compatibility was verified.",
    "Raw full-text content is not saved, published or reproduced.",
    "Patents and non-open-access sources require independent qualified review.",
    "This report cannot approve tasks, release a cohort or freeze an experiment split."
   ]}
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--metadata-only",action="store_true")
 parser.add_argument("--workers",type=int,default=3)
 args=parser.parse_args()
 if not 1<=args.workers<=4:raise ValueError("Workers must be 1..4")
 queue_before=QUEUE.read_bytes()
 rows=read_queue()
 ids=sorted({pmid for row in rows for pmid in pmids(row)})
 pubs={}
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  todo={pool.submit(fetch_pub,p,args.metadata_only):p for p in ids}
  for future in as_completed(todo):
   p=todo[future];r=future.result();pubs[p]=r
   print(json.dumps({"pmid":p,"metadata_access":r["metadata_access"],
    "fulltext_access":r["fulltext_access"],
    "sections":r.get("relevant_sections_total",0)}),flush=True)
 if QUEUE.read_bytes()!=queue_before:raise PermissionError("Human review queue modified during indexing")
 report=make_report(rows,pubs,fingerprint(queue_before))
 if report["task_count"]!=30 or report["scientifically_approved_tasks"]!=0:
  raise RuntimeError("Invalid output: unexpected task count or approval")
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/"source_access_index.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 doc=["# Primary-source access index for 30 unapproved BindingDB tasks","",
  "**Not scientific adjudication. ZERO approved tasks.**",
  "",
  "This file contains only metadata and, where openly available, section-level indicators. Source article bodies were not saved.",
  "",
  "| Evidence access category | Tasks |","| --- | ---: |"]
 for name,count in sorted(report["source_status_counts"].items()):doc.append("| "+name+" | "+str(count)+" |")
 doc+=["","Unique PubMed publications: "+str(len(pubs)),
  "Public open-access section indexes: "+str(report["publication_access_counts"].get("OPEN_ACCESS_SECTION_INDEXED",0)),
  "",
  "To approve an assay, a qualified reviewer must independently inspect the original paper/patent and supplementary data, verify the physical protein construct, exact Kd qualifiers, and comparable assay conditions, and record evidence without altering metadata.",
  "Original review CSV SHA-256: "+report["review_file_sha256"]]
 (OUT/"SOURCE_ACCESS_SUMMARY.md").write_text("\n".join(doc)+"\n",encoding="utf-8")
 print(json.dumps({"status":report["status"],"task_count":report["task_count"],
  "unique_publications":report["unique_publications"],
  "source_status_counts":report["source_status_counts"],
  "publication_access_counts":report["publication_access_counts"],
  "queue_unchanged":True,"source_approvals":0},indent=2),flush=True)
if __name__=="__main__":main()
