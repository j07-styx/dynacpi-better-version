from pathlib import Path
import csv,json,hashlib,time
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
root=Path(r"C:\Users\J0124\dynacpi_v4")
legacy=Path(r"C:\Users\J0124\OneDrive\Desktop\perceiver_cpi\reproduction")
out=root/"data"/"audit"; out.mkdir(parents=True,exist_ok=True)
report={}
for ds in ("davis","metz","kiba"):
 folder=legacy/(ds+"_novel_pair_fold0")
 parts={}; keys={}
 for split in ("train","val","test"):
  p=folder/(split+".csv")
  if not p.exists(): continue
  rows=list(csv.DictReader(p.open(encoding="utf-8-sig")))
  if not rows: continue
  cols=list(rows[0]); smcol=next((x for x in cols if x.lower()=="smiles"),None)
  seqcol=next((x for x in cols if x.lower() in ("sequence","sequences")),None)
  labcol=next((x for x in cols if x.lower() in ("label","target","affinity")),None)
  if not all((smcol,seqcol,labcol)):
   parts[split]={"error":"unrecognized schema","columns":cols};continue
  mols={}; seqs=set(); pairs=[]; values=[]; invalid=0
  for r in rows:
   raw=r[smcol]
   if raw not in mols:
    m=Chem.MolFromSmiles(raw)
    mols[raw]=Chem.MolToSmiles(m,isomericSmiles=True) if m else None
   sm=mols[raw]; seq=r[seqcol].strip()
   if sm is None: invalid+=1;continue
   seqs.add(seq); pairs.append((sm,seq))
   try: values.append(float(r[labcol]))
   except ValueError: invalid+=1
  keys[split]={"compounds":set(mols.values())-{None},"proteins":seqs,"pairs":set(pairs)}
  parts[split]={"rows":len(rows),"unique_compounds":len(keys[split]["compounds"]),"unique_sequences":len(seqs),"unique_pairs":len(set(pairs)),"duplicate_pair_rows":len(pairs)-len(set(pairs)),"invalid_rows":invalid,"label_min":min(values) if values else None,"label_max":max(values) if values else None,"labels_equal_5":sum(v==5 for v in values),"max_sequence_length":max(map(len,seqs),default=0),"source":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
 overlaps={}
 for a,b in (("train","val"),("train","test"),("val","test")):
  if a in keys and b in keys: overlaps[a+"_"+b]={k:len(keys[a][k]&keys[b][k]) for k in keys[a]}
 report[ds]={"splits":parts,"overlaps":overlaps,"status":"Legacy inventory only; not an approved v4 split; label provenance remains unverified"}
(out/"legacy_inventory.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
