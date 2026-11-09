from pathlib import Path
import urllib.request,zipfile,hashlib,os,json,re,io,csv
root=Path(__file__).resolve().parents[1]
out=root/"data"/"bindingdb_202611";out.mkdir(exist_ok=True)
base="https://www.bindingdb.org/rwd/bind/downloads/"
names=["BindingDB_BindingDB_Articles_202611_tsv","BindingDB_Assays_202611_tsv","BindingDB_rsid_eaids_202611_tsv","BindingDB_All_202611_tsv"]
def fetch(name):
 expected=urllib.request.urlopen(base+name+".md5",timeout=30).read().decode().strip().split()[0].lower()
 if not re.fullmatch("[0-9a-f]{32}",expected):raise IOError("Unrecognized publisher checksum")
 path=out/(name+".zip");tmp=path.with_suffix(".tmp")
 if not path.exists():
  with urllib.request.urlopen(base+name+".zip",timeout=120) as r,tmp.open("wb") as f:
   for chunk in iter(lambda:r.read(1<<20),b""):f.write(chunk)
 check=path if path.exists() else tmp
 md=hashlib.md5();sha=hashlib.sha256()
 with check.open("rb") as f:
  for block in iter(lambda:f.read(4<<20),b""):md.update(block);sha.update(block)
 if md.hexdigest()!=expected:raise ValueError("Publisher MD5 mismatch: "+name)
 if check==tmp:os.replace(tmp,path)
 with zipfile.ZipFile(path) as z:
  contents=[(i.filename,i.file_size) for i in z.infolist()]
  member=max(z.infolist(),key=lambda i:i.file_size)
  with z.open(member) as f:head=f.readline().decode("utf-8-sig",errors="replace").strip()
 info={"url":base+name+".zip","bytes":path.stat().st_size,"md5":expected,"sha256":sha.hexdigest(),"members":contents,"header":head}
 (out/(name+".json")).write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
for name in names:fetch(name)
