"""Download original paper table figures into a private cache for human visual reconciliation.
Reject HTML/error responses; never infer numeric Kd by OCR.
"""
import json,urllib.request,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
data=json.loads((root/"research/source_access_v4/PRIMARY_AFFINITY_TABLE_DETAIL.json").read_text(encoding="utf-8"))
dest=root/"data/audit/primary_table_figures_LOCAL_PRIVATE";dest.mkdir(exist_ok=True)
out=[]
for paper in data["publications"]:
 for table in paper["affinity_related_tables"]:
  for graphic in table["graphics"]:
   name=graphic["href"]
   if not name.lower().endswith((".jpg",".jpeg",".png")):continue
   url="https://pmc.ncbi.nlm.nih.gov/articles/"+paper["pmcid"]+"/bin/"+name
   item={"pmcid":paper["pmcid"],"table_id":table["id"],"graphic":name,"url":url}
   try:
    req=urllib.request.Request(url,headers={"User-Agent":"DynaCPI-academic-figure-audit/1.0"})
    with urllib.request.urlopen(req,timeout=18) as resp:raw=resp.read(8_000_001)
    if len(raw)>8_000_000:raise ValueError("figure exceeds size limit")
    if not(raw.startswith(bytes.fromhex("ffd8ff")) or raw.startswith(bytes.fromhex("89504e470d0a1a0a"))):raise ValueError("not a JPEG/PNG response")
    path=dest/(paper["pmcid"]+"_"+name);path.write_bytes(raw)
    item.update(status="IMAGE_CACHED_PRIVATE",bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),path=str(path))
   except Exception as e:item.update(status="UNAVAILABLE",error_type=type(e).__name__,error=str(e)[:130])
   out.append(item)
report={"status":"PRIMARY_FIGURE_ACCESS_NO_RECORD_APPROVAL","figures":out,"downloaded":sum(x.get("status")=="IMAGE_CACHED_PRIVATE" for x in out),"record_level_verified":0}
(root/"research/source_access_v4/PRIMARY_FIGURE_ACCESS.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({"status":report["status"],"downloaded":report["downloaded"],"total":len(out),"figures":[{"pmcid":x["pmcid"],"table_id":x["table_id"],"status":x["status"],"error":x.get("error","")} for x in out]},indent=2))
