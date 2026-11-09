"""Download official FAIR ESM files with resumable validated HTTP ranges."""
import concurrent.futures,hashlib,json,os,re,time,urllib.request
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
folder=Path(torch.hub.get_dir())/"checkpoints";folder.mkdir(parents=True,exist_ok=True)
def fetch(name,category):
 url="https://dl.fbaipublicfiles.com/fair-esm/"+category+"/"+name
 req=urllib.request.Request(url,headers={"Range":"bytes=0-0","Accept-Encoding":"identity"})
 with urllib.request.urlopen(req,timeout=40) as r:
  m=re.fullmatch(r"bytes 0-0/(\d+)",r.headers.get("Content-Range",""))
  if not m:raise RuntimeError("Missing range support")
  size=int(m.group(1));etag=r.headers.get("ETag")
 target=folder/name
 if target.is_file() and target.stat().st_size==size:return
 chunks=folder/(name+".ranges");chunks.mkdir(exist_ok=True)
 step=8*1024*1024;starts=list(range(0,size,step))
 def get(start):
  end=min(start+step,size)-1;part=chunks/str(start)
  if part.exists() and part.stat().st_size==end-start+1:return
  for attempt in range(4):
   try:
    headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"}
    if etag:headers["If-Match"]=etag
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=55) as r:
     if r.status!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{size}":raise IOError("Range mismatch")
     b=r.read(end-start+2)
    if len(b)!=end-start+1:raise IOError("Short range")
    tmp=part.with_suffix(".tmp");tmp.write_bytes(b);os.replace(tmp,part);return
   except Exception:
    if attempt==3:raise
    time.sleep(2**attempt)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(get,starts))
 tmp=folder/(name+".assembling");digest=hashlib.sha256()
 with tmp.open("wb") as f:
  for start in starts:
   b=(chunks/str(start)).read_bytes();digest.update(b);f.write(b)
 assert tmp.stat().st_size==size
 os.replace(tmp,target)
 (root/"research"/(name+".manifest.json")).write_text(json.dumps({"url":url,"size":size,"etag":etag,"sha256":digest.hexdigest()}))
 for start in starts:(chunks/str(start)).unlink()
 chunks.rmdir()
if __name__=="__main__":
 fetch("esm2_t33_650M_UR50D.pt","models")
 fetch("esm2_t33_650M_UR50D-contact-regression.pt","regression")
