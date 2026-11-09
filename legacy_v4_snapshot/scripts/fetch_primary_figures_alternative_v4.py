"""Try canonical public PMC static image delivery for primary-paper table figures.
Store only valid image bytes; no OCR or record approvals.
"""
import json,urllib.request,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
report=json.loads((root/"research/source_access_v4/PRIMARY_FIGURE_ACCESS.json").read_text(encoding="utf-8"))
cache=root/"data/audit/primary_table_figures_LOCAL_PRIVATE";cache.mkdir(exist_ok=True)
results=[]
for item in report["figures"]:
 pmc=item["pmcid"];name=item["graphic"]
 # PMC static image service uses the article-specific PMC ID as the path segment.
 urls=["https://cdn.ncbi.nlm.nih.gov/pmc/blobs/"+pmc+"/"+name,
       "https://pmc.ncbi.nlm.nih.gov/articles/instance/"+pmc+"/bin/"+name,
       "https://europepmc.org/articles/"+pmc+"/bin/"+name]
 record={"pmcid":pmc,"table_id":item["table_id"],"filename":name,"status":"UNAVAILABLE"}
 for url in urls:
  try:
   req=urllib.request.Request(url,headers={"User-Agent":"DynaCPI-research-figure-review/1.0"})
   with urllib.request.urlopen(req,timeout=12) as resp:raw=resp.read(8_000_001)
   if len(raw)>8_000_000:raise ValueError("too large")
   if not(raw.startswith(bytes.fromhex("ffd8ff")) or raw.startswith(bytes.fromhex("89504e470d0a1a0a"))):raise ValueError("not image")
   path=cache/(pmc+"_"+name);path.write_bytes(raw)
   record.update(status="IMAGE_CACHED_PRIVATE",url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),path=str(path))
   break
  except Exception as e:record["last_error"]=type(e).__name__+": "+str(e)[:85]
 results.append(record)
out={"status":"ALTERNATIVE_PRIMARY_IMAGE_ACCESS","downloaded":sum(x["status"]=="IMAGE_CACHED_PRIVATE" for x in results),"total":len(results),"figures":results,"scientific_approvals":0}
(root/"research/source_access_v4/PRIMARY_FIGURE_ALTERNATIVE_ACCESS.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({"downloaded":out["downloaded"],"total":out["total"],"by_table":[{"paper":x["pmcid"],"table":x["table_id"],"status":x["status"]} for x in results]},indent=2))
