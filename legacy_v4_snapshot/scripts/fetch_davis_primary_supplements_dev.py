"""Recover publisher supplement links and cache successful public downloads."""
import concurrent.futures, hashlib, json, re, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/audit/davis_primary_20261107"
OUT.mkdir(exist_ok=True)
url="https://www.nature.com/articles/nbt.1990"
req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
with urllib.request.urlopen(req,timeout=30) as f:raw=f.read()
(OUT/"publisher.html").write_bytes(raw)
links=sorted(set(re.findall(r'https?[^"<> ]+(?:xls|xlsx|pdf)',raw.decode(errors="replace"))))
links=[u.replace("&amp;","&") for u in links if "1990" in u or "supp" in u.lower()]
print("LINKS",json.dumps(links),flush=True)
def fetch(u):
    try:
        with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"Mozilla/5.0"}),timeout=30) as f: data=f.read()
        name=u.rsplit("/",1)[-1]
        if data[:4]==b"%PDF" or data[:8]==bytes.fromhex("d0cf11e0a1b11ae1") or data[:2]==b"PK":
            path=OUT/name
            if not path.exists():path.write_bytes(data)
            return {"url":u,"path":str(path),"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
        return {"url":u,"error":"unrecognized file signature"}
    except Exception as e:return {"url":u,"error":str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,links))
(OUT/"downloads.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2),flush=True)
