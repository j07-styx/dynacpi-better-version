"""Batch/padding/atom-order invariance, gradient flow and checkpoint parity."""
import sys,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from dynacpi.transfer_cpi import TransferCPI
from dynacpi.transfer_batch import TransferCollator
from prepare_transfer_compounds_dev import featurize
torch.set_num_threads(2);torch.manual_seed(13)
graphs={"a":featurize("CCO"),"b":featurize("c1ccccc1C(=O)O")}
proteins={"p":torch.randn(7,1280),"q":torch.randn(11,1280)}
collate=TransferCollator(graphs,proteins,torch.zeros(12),torch.ones(12))
rows=[{"compound":"a","protein":"p","y":1.},{"compound":"b","protein":"q","y":2.}]
b=collate(rows);model=TransferCPI().eval()
with torch.no_grad():
    pred,_=model(b);single,_=model(collate(rows[:1]))
    assert torch.allclose(pred[:1],single,atol=2e-6)
    reverse,_=model(collate(rows[::-1]));assert torch.allclose(reverse.flip(0),pred,atol=2e-6)
    padded={k:v.clone() for k,v in b.items()}
    padded["protein"][~b["protein_mask"]]=1234
    padded["atom_features"][~b["atom_mask"]]=1234
    pp,_=model(padded);assert torch.allclose(pp,pred,atol=2e-6)
    permutation=torch.randperm(b["z"].shape[1]);permuted=dict(b)
    for key in ("z","atom_features","atom_mask"):permuted[key]=b[key][:,permutation]
    permuted["adj"]=b["adj"][:,:,permutation][:,:,:,permutation]
    pp,_=model(permuted);assert torch.allclose(pp,pred,atol=2e-6)
    other=dict(b);other["protein"]=torch.randn_like(b["protein"])
    a,_=model(b,True);c,_=model(other,True);assert torch.equal(a,c)
model.zero_grad();out,_=model(b);out.square().mean().backward()
for name in ("atom.weight","protein.1.weight","drug_to_protein.in_proj_weight","interaction_head.3.weight"):
    grad=dict(model.named_parameters())[name].grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/"state.pt";torch.save(model.state_dict(),path)
    loaded=TransferCPI().eval();loaded.load_state_dict(torch.load(path,weights_only=True))
    with torch.no_grad():assert torch.equal(model(b)[0],loaded(b)[0])
print("PASS batch/order/padding/atom invariance, chemical ablation, all-branch gradients and checkpoint parity")
print("parameters",sum(p.numel() for p in model.parameters()))
