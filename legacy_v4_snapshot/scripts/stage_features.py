from pathlib import Path
import csv,json,hashlib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
root=Path(r"C:\Users\J0124\dynacpi_v4")
src=Path(r"C:\Users\J0124\OneDrive\Desktop\perceiver_cpi\reproduction")
out=root/"data"/"staging";out.mkdir(parents=True,exist_ok=True)
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
compounds={}; proteins={}; rows=[]; invalid=[]
for ds in ("davis","metz","kiba"):
 for split in ("train","val","test"):
  p=src/(ds+"_novel_pair_fold0")/(split+".csv")
  if not p.exists():continue
  for idx,r in enumerate(csv.DictReader(p.open(encoding="utf-8-sig"))):
   sm=r.get("smiles"); seq=r.get("sequences",r.get("sequence","")).strip()
   m=Chem.MolFromSmiles(sm) if sm else None
   if m is None or not seq:invalid.append([ds,split,idx]);continue
   canon=Chem.MolToSmiles(m,isomericSmiles=True)
   cid=hashlib.sha256(canon.encode()).hexdigest()
   pid=hashlib.sha256(seq.encode()).hexdigest()
   compounds[cid]=canon;proteins[pid]=seq
   rows.append({"dataset":ds,"legacy_split":split,"source_row":idx+2,"compound_hash":cid,"sequence_hash":pid,"label_as_stored":r.get("label"),"status":"quarantined_pending_construct_assay_and_endpoint_provenance"})
with (out/"legacy_observations.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
ids=sorted(compounds)
fps=np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(compounds[k])) for k in ids])
np.savez_compressed(out/"morgan_radius2_2048_chiral.npz",ids=np.array(ids),bits=fps)
(out/"compounds.json").write_text(json.dumps(compounds))
(out/"sequences.json").write_text(json.dumps(proteins))
with (out/"sequences.fasta").open("w") as f:
 for pid,seq in sorted(proteins.items()):f.write(">"+pid+"\n"+seq+"\n")
manifest={"observations":len(rows),"compounds":len(compounds),"sequences":len(proteins),"invalid_rows":len(invalid),"fingerprint_shape":list(fps.shape),"status":"Staging only; no final v4 split, no exact-Kd eligibility claim, no ESM/structure preprocessing"}
(out/"manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
