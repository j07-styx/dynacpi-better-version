"""Independently fetch two open-access original methods and verify local assay metadata.
No per-ligand values are exported; no scientific approvals or training.
"""
import csv,hashlib,json,re,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
OUT=ROOT/"research/source_access_v4/open_access_method_evidence.json"
SPECS=[
 {"assay":"50002067_1","pmid":"29648813","pmcid":"PMC5957473","accession":"O14786","n":26,"region":[273,427],"method_sections":["sec4.2.1","sec4.2.3"],"patterns":[r"273\s*[–-]\s*427",r"equilibrium binding affinity",r"25\s*°C"],"method":"steady_state_SPR_equilibrium_Kd"},
 {"assay":"50017739_13","pmid":"35333039","pmcid":"PMC9014859","accession":"P17342","n":22,"region":[27,541],"method_sections":["sec2.4","sec4.4"],"patterns":[r"27\s*[–-]\s*541",r"Langmuir",r"CM5"],"method":"SPR_affinity_model_1to1"}
]
def run():
 before=QUEUE.read_bytes()
 rows=list(csv.DictReader(QUEUE.open(encoding="utf-8-sig",newline="")))
 assert len(rows)==30 and all(x["curation_decision"]=="UNREVIEWED" for x in rows)
 by={x["assay_key"]:x for x in rows}
 results=[]
 for spec in SPECS:
  row=by[spec["assay"]]
  assert row["pmids"]==spec["pmid"] and row["swissprot"]==spec["accession"]
  assert int(row["record_count"])==spec["n"]
  url="https://www.ebi.ac.uk/europepmc/webservices/rest/"+spec["pmcid"]+"/fullTextXML"
  req=urllib.request.Request(url,headers={"User-Agent":"DynaCPI-methods-audit/1.0"})
  with urllib.request.urlopen(req,timeout=35) as response: raw=response.read()
  tree=ET.fromstring(raw)
  ids={x.attrib.get("pub-id-type"):x.text for x in tree.iter("article-id")}
  assert ids["pmid"]==spec["pmid"] and ids["pmcid"]==spec["pmcid"]
  sections={x.attrib.get("id"):x for x in tree.iter("sec")}
  method=" ".join(" ".join(sections[s].itertext()) for s in spec["method_sections"])
  checks={p:bool(re.search(p,method,re.I)) for p in spec["patterns"]}
  assert all(checks.values()),(spec["assay"],checks)
  description=row["assay_description"].lower()
  assert "spr" in description or "surface plasmon resonance" in description
  if spec["assay"]=="50002067_1":
   assert "nrp1-b1" in description and "his6" in description
  else:
   assert "npr-c" in description and "27 to 541" in description
  assert "binding affinity" in description
  results.append({"assay_key":spec["assay"],"pmid":spec["pmid"],"pmcid":spec["pmcid"],"uniprot_accession":spec["accession"],"source_record_count":spec["n"],"published_construct_region":spec["region"],"assay_method":spec["method"],"source_url":url,"source_xml_sha256":hashlib.sha256(raw).hexdigest(),"method_section_ids":spec["method_sections"],"primary_method_pattern_checks":checks,"source_level_method_corroborated":True,"record_level_compound_and_numeric_Kd_verified":False,"physical_construct_sequence_verified":False,"scientifically_approved":False})
 assert QUEUE.read_bytes()==before,"Review queue changed during source audit"
 report={"status":"PASS_PRIMARY_METHODS_ONLY_NOT_RECORD_APPROVAL","timestamp_utc":datetime.now(timezone.utc).isoformat(),"review_queue_sha256":hashlib.sha256(before).hexdigest(),"corroborated_assays":2,"source_records_in_assays":sum(x["source_record_count"] for x in results),"approved_records":0,"main_training_permitted":False,"sources":results}
 OUT.parent.mkdir(parents=True,exist_ok=True)
 OUT.write_text(json.dumps(report,indent=2),encoding="utf-8")
 return report
if __name__=="__main__":
 r=run()
 print(json.dumps({k:r[k] for k in ("status","corroborated_assays","source_records_in_assays","approved_records","main_training_permitted")},indent=2))
