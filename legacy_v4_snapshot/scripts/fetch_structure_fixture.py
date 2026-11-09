import json,urllib.request,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/"data"/"structure_fixtures"/"P15056";out.mkdir(parents=True,exist_ok=True)
api="https://alphafold.ebi.ac.uk/api/prediction/P15056"
raw=urllib.request.urlopen(api,timeout=30).read();records=json.loads(raw)
entry=next(r for r in records if r.get("uniprotAccession")=="P15056" and r.get("uniprotStart")==1)
(out/"metadata.json").write_bytes(raw)
meta={"api":api,"entry":entry.get("entryId"),"version":entry.get("latestVersion"),"status":"Canonical protein-only engineering fixture, not assigned to any measured mutant/assay"}
for key,name in [("pdbUrl","structure.pdb"),("paeDocUrl","pae.json")]:
 if entry.get(key):
  data=urllib.request.urlopen(entry[key],timeout=50).read();(out/name).write_bytes(data);meta[name]={"url":entry[key],"sha256":hashlib.sha256(data).hexdigest()}
(out/"manifest.json").write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
