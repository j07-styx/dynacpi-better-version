"""Inventory primary-paper compound affinity tables and supplemental attachments.
Source discovery only; does not equate paper compounds with BindingDB records.
"""
import json,re,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=json.loads((root/"research/source_access_v4/open_access_method_evidence.json").read_text(encoding="utf-8"))
out=[]
for s in source["sources"]:
 req=urllib.request.Request(s["source_url"],headers={"User-Agent":"DynaCPI-source-table-inventory/1.0"})
 with urllib.request.urlopen(req,timeout=40) as response: xml=response.read()
 tree=ET.fromstring(xml)
 tables=[]
 for tab in tree.iter("table-wrap"):
  label=" ".join(tab.find("label").itertext()) if tab.find("label") is not None else ""
  caption=" ".join(tab.find("caption").itertext()) if tab.find("caption") is not None else ""
  body=" ".join(tab.itertext())
  aff=re.findall(r"\b(?:K\s*d|K\s*i|IC\s*50|nM|µM|μM|SPR)\b",body,re.I)
  tables.append({"id":tab.attrib.get("id",""),"label":label[:120],"caption":caption[:550],"affinity_mentions":len(aff),"table_rows":len(list(tab.iter("tr"))),"table_cells":len(list(tab.iter("td"))),"has_explicit_kd":bool(re.search(r"\bK\s*d\b",body,re.I))})
 supplements=[]
 for el in tree.iter():
  tag=el.tag.split("}")[-1]
  if tag in ("supplementary-material","supplementary-materials","ext-link"):
   if tag=="ext-link" and not any("support" in str(v).lower() or "supp" in str(v).lower() for v in el.attrib.values()):continue
   supplements.append({"tag":tag,"id":el.attrib.get("id",""),"href":next((v for k,v in el.attrib.items() if k.endswith("href")),""),"description":" ".join(el.itertext())[:220]})
 out.append({"assay_key":s["assay_key"],"pmid":s["pmid"],"pmcid":s["pmcid"],"tables":tables,"supplement_links":supplements,"table_count":len(tables),"supplement_link_count":len(supplements),"record_level_mapping_verified":False,"source_numeric_kd_reconciled":False})
report={"status":"SOURCE_TABLE_DISCOVERY_ONLY_NOT_RECORD_APPROVAL","publications":out,"approved_records":0}
dest=root/"research/source_access_v4/PRIMARY_TABLE_AND_SUPPLEMENT_INVENTORY.json"
dest.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({"status":report["status"],"publications":[{"pmid":p["pmid"],"tables":p["table_count"],"supplements":p["supplement_link_count"],"tables_with_kd":[t["id"] for t in p["tables"] if t["has_explicit_kd"]]} for p in out]},indent=2))
