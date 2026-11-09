"""Extract text-visible original publication affinity tables and image locators.
Does not equate publication compound numbers with BindingDB source rows.
"""
import json,re,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
root=Path(__file__).resolve().parents[1]
inventory=json.loads((root/"research/source_access_v4/PRIMARY_TABLE_AND_SUPPLEMENT_INVENTORY.json").read_text(encoding="utf-8"))
results=[]
for paper in inventory["publications"]:
 url="https://www.ebi.ac.uk/europepmc/webservices/rest/"+paper["pmcid"]+"/fullTextXML"
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"DynaCPI-affinity-table-inspection/1.0"}),timeout=30) as resp: xml=resp.read()
 tree=ET.fromstring(xml)
 tabs=[]
 for tab in tree.iter("table-wrap"):
  label=" ".join(tab.find("label").itertext()) if tab.find("label") is not None else ""
  caption=" ".join(tab.find("caption").itertext()) if tab.find("caption") is not None else ""
  graphics=[{"tag":el.tag.split("}")[-1],"href":next((v for k,v in el.attrib.items() if k.endswith("href")),"")} for el in tab.iter() if el.tag.split("}")[-1] in ("graphic","inline-graphic")]
  text=" ".join(tab.itertext())
  if any(w in (label+" "+caption).lower() for w in ("binding","affinity","equilibrium")):
   rows=[]
   for tr in tab.iter("tr"):
    cells=[" ".join(c.itertext()).strip() for c in tr if c.tag.split("}")[-1] in ("td","th")]
    if cells:rows.append(cells)
   tabs.append({"id":tab.attrib.get("id"),"label":label,"caption":caption,"graphics":graphics,"visible_rows":rows[:100],"visible_text_excerpt":text[:2200],"numeric_table_reconciliation_complete":False})
 results.append({"pmid":paper["pmid"],"pmcid":paper["pmcid"],"affinity_related_tables":tabs})
dest=root/"research/source_access_v4/PRIMARY_AFFINITY_TABLE_DETAIL.json"
dest.write_text(json.dumps({"status":"PUBLICATION_TABLE_EXTRACTION_NO_BINDINGDB_MATCH","publications":results,"approved_records":0},indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({"status":"PUBLICATION_TABLE_EXTRACTION_NO_BINDINGDB_MATCH","publications":[{"pmid":x["pmid"],"affinity_tables":[{"id":t["id"],"graphics":len(t["graphics"]),"visible_rows":len(t["visible_rows"])} for t in x["affinity_related_tables"]]} for x in results]},indent=2))
