"""Molecular graph, chiral fingerprint and property cache for train/dev only."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
from rdkit import Chem,DataStructs,rdBase
from rdkit.Chem import Descriptors,Crippen,Lipinski,rdMolDescriptors,rdFingerprintGenerator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.transfer_data import load_development
def featurize(sm):
    mol=Chem.MolFromSmiles(sm)
    if mol is None or not mol.GetNumAtoms():raise ValueError("Invalid molecule")
    n=mol.GetNumAtoms();z=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
    if z.max()>118:raise ValueError("Unsupported element")
    features=torch.tensor([[a.GetDegree()/6,a.GetFormalCharge()/4,a.GetTotalNumHs()/4,
      float(a.GetIsAromatic()),float(a.IsInRing()),a.GetMass()/200,
      float(a.GetHybridization()==Chem.HybridizationType.SP),float(a.GetHybridization()==Chem.HybridizationType.SP2),
      float(a.GetHybridization()==Chem.HybridizationType.SP3),int(a.GetChiralTag())/3] for a in mol.GetAtoms()],dtype=torch.float32)
    adj=torch.zeros(4,n,n);types={Chem.BondType.SINGLE:0,Chem.BondType.DOUBLE:1,Chem.BondType.TRIPLE:2,Chem.BondType.AROMATIC:3}
    for bond in mol.GetBonds():
        if bond.GetBondType() not in types:raise ValueError("Unsupported bond")
        k=types[bond.GetBondType()];i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();adj[k,i,j]=adj[k,j,i]=1
    prop=torch.tensor([Descriptors.MolWt(mol),Crippen.MolLogP(mol),rdMolDescriptors.CalcTPSA(mol),
      Lipinski.NumHDonors(mol),Lipinski.NumHAcceptors(mol),Lipinski.NumRotatableBonds(mol),
      rdMolDescriptors.CalcFractionCSP3(mol),Lipinski.RingCount(mol),rdMolDescriptors.CalcNumAromaticRings(mol),
      mol.GetNumHeavyAtoms(),Chem.GetFormalCharge(mol),len(Chem.GetMolFrags(mol))],dtype=torch.float32)
    a=np.zeros(1024,dtype=np.uint8)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024,includeChirality=True)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),a)
    assert torch.isfinite(prop).all()
    return {"z":z,"atom_features":features,"adj":adj,"properties":prop,"fp":torch.from_numpy(a)}
def main():
    rows,_,split_hash=load_development();out=ROOT/"data/transfer_compounds_dev.pt"
    if out.exists():raise FileExistsError(out)
    smiles={r["compound"]:r["smiles"] for r in rows};graphs={}
    for i,(cid,sm) in enumerate(sorted(smiles.items()),1):
        assert hashlib.sha256(sm.encode()).hexdigest()==cid
        graphs[cid]=featurize(sm)
        if i%1000==0:print("molecules",i,"/",len(smiles),flush=True)
    train_ids=sorted({r["compound"] for r in rows if r["split"]=="train"})
    props=torch.stack([graphs[c]["properties"] for c in train_ids])
    meta={"split_sha256":split_hash,"rdkit_version":rdBase.rdkitVersion,"n_compounds":len(graphs),
      "property_scaling":"unique training molecules mean/std","max_atoms":max(len(g["z"]) for g in graphs.values())}
    torch.save({"graphs":graphs,"property_mean":props.mean(0),"property_std":props.std(0,unbiased=False).clamp_min(.1),"meta":meta},out)
    print(json.dumps(meta),flush=True)
if __name__=="__main__":main()
