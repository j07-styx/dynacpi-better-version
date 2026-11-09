"""Inspect public supplementary CSV metadata for two original papers.
Download to local private research cache; report only columns and shape.
No numeric affinity extraction or source approvals.
"""
import csv,io,json,urllib.request,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
inventory=json.loads((root/"research/source_access_v4/PRIMARY_TABLE_AND_SUPPLEMENT_INVENTORY.json").read_text(encoding="utf-8"))
cache=root/"data/audit/primary_supplements_LOCAL_PRIVATE"
cache.mkdir(exist_ok=True)
report=[]
for paper in inventory["publications"]:
 urls=list(dict.fromkeys(x["href"] for x in paper["supplement_links"] if x["href"].lower().endswith(".csv")))
 for url in urls:
  record={"pmid":paper["pmid"],"url":url,"status":"NOT_FETCHED"}
  try:
   req=urllib.request.Request(url.replace("http://","https://"),headers={"User-Agent":"DynaCPI-primary-source-audit/1.0"})
   with urllib.request.urlopen(req,timeout=25) as response:
    raw=response.read(4_000_001)
    mime=response.headers.get("Content-Type","")
   if len(raw)>4_000_000:raise ValueError("File exceeds 4 MB audit cap")
   if raw.lstrip()[:10].lower().startswith((b"<!doctype",b"<html")):raise ValueError("HTML returned instead of CSV")
   decoded=raw.decode("utf-8-sig")
   parsed=list(csv.reader(io.StringIO(decoded)))
   if len(parsed)<2:raise ValueError("No tabular rows")
   path=cache/(paper["pmid"]+"_"+url.split("/")[-1])
   path.write_bytes(raw)
   record.update(status="CSV_AVAILABLE_LOCAL_PRIVATE",bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),rows=len(parsed)-1,columns=len(parsed[0]),headers=parsed[0][:30],private_path=str(path))
  except Exception as e:
   record.update(status="FETCH_UNAVAILABLE",error_type=type(e).__name__,error=str(e)[:180])
  report.append(record)
out={"status":"SUPPLEMENTARY_CSV_ACCESS_INVENTORY_NOT_SOURCE_APPROVAL","files":report,"record_level_verified":0,"approved_records":0}
dest=root/"research/source_access_v4/SUPPLEMENTARY_CSV_ACCESS.json"
dest.write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
