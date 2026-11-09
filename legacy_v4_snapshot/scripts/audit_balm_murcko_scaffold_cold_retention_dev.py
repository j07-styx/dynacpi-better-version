import json
from collections import Counter
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
import numpy as np
from dynacpi.balm_public_data_dev import get_train_only

raw=get_train_only()
keys={}
for sm in set(r["smiles"] for r in raw):
    mol=Chem.MolFromSmiles(sm)
    assert mol is not None
    scaffold=MurckoScaffold.GetScaffoldForMol(mol)
    key=Chem.MolToSmiles(scaffold,canonical=True,isomericSmiles=True)
    keys[sm]=key if key else "ACYCLIC_SINGLE_GROUP"
print("SMILES",len(keys),"unique scaffold groups",len(set(keys.values())),"ACYCLIC unique smiles",sum(v=="ACYCLIC_SINGLE_GROUP" for v in keys.values()),flush=True)
grp=np.asarray([r["group"] for r in raw])
for i,(it,iv) in enumerate(GroupKFold(n_splits=4).split(grp,groups=grp),1):
    blocked={keys[raw[int(j)]["smiles"]] for j in iv}
    kept=[int(j) for j in it if keys[raw[int(j)]["smiles"]] not in blocked]
    print("outer",i,"kept raw",len(kept),"blocked",len(it)-len(kept),"key count",len(blocked),"acyclic withheld", "ACYCLIC_SINGLE_GROUP" in blocked, flush=True)
    g2=np.array([raw[j]["group"] for j in kept])
    for k,(ti,vi) in enumerate(GroupKFold(n_splits=3).split(g2,groups=g2),1):
        blocked2={keys[raw[kept[int(j)]]["smiles"]] for j in vi}
        left=[j for j in ti if keys[raw[kept[int(j)]]["smiles"]] not in blocked2]
        print("  inner",k,"retained",len(left),"unique groups",len(set(g2[left])),flush=True)
