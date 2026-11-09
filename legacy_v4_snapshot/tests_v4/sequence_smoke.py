import sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from dynacpi.sequence_model import SequenceFoundation
torch.manual_seed(7); model=SequenceFoundation().cuda().eval()
smiles=["CCO","CC(C)(C)C"]
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
fp=torch.tensor(np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles]),device="cuda",dtype=torch.float32)
seq=torch.randn(2,37,1280,device="cuda")
mask=torch.arange(37,device="cuda")[None,:]<torch.tensor([19,37],device="cuda")[:,None]
with torch.no_grad():
 batch=model(smiles,fp,seq,mask)
 single=model(smiles[:1],fp[:1],seq[:1,:19],mask[:1,:19])
 comp=model.compound(smiles,fp)
 only=model.compound(smiles[:1],fp[:1])
 torch.testing.assert_close(batch[:1],single,atol=1e-5,rtol=1e-5)
 torch.testing.assert_close(comp[:1],only,atol=1e-5,rtol=1e-5)
model.train()
loss=(model(smiles,fp,seq,mask)-torch.tensor([.3,-.2],device="cuda")).square().mean();loss.backward()
grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
assert grads and all(np.isfinite(list(grads.values())))
assert model.sequence[0].weight.grad.norm().item()>0
result={"scope":"Real RDKit molecules, synthetic residue embeddings and regression targets; interface/gradient verification only","device":torch.cuda.get_device_name(),"parameters":sum(p.numel() for p in model.parameters()),"prediction_batch_difference":float((batch[:1]-single).abs().max()),"compound_batch_difference":float((comp[:1]-only).abs().max()),"finite_gradient_parameters":len(grads),"sequence_projection_grad_norm":grads["sequence.0.weight"],"synthetic_loss":float(loss.detach())}
print(json.dumps(result));(root/"research"/"sequence_smoke.json").write_text(json.dumps(result,indent=2))
