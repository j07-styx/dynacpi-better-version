"""Focused gated-geometry functional and gradient tests; CPU-only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from dynacpi.confidence_gated_geometry import ConfidenceGatedGeometry
class DummyGeometry(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(2.))
    def enhance(self,tokens,graph):
        out=tokens.clone()
        idx=graph["indices"]
        return out.index_add(0,idx,torch.ones((len(idx),256))*self.scale)
torch.manual_seed(10)
base=DummyGeometry()
model=ConfidenceGatedGeometry(base)
tokens=torch.randn(7,256,requires_grad=True)
graph={"indices":torch.tensor([1,3,5])}
missing,missing_gate=model(tokens,None,True)
assert torch.equal(missing,tokens) and torch.count_nonzero(missing_gate)==0
out,gate=model(tokens,graph,True)
assert out.shape==tokens.shape and gate.shape==(7,1)
assert torch.allclose(model.enhance(tokens,graph),out)
assert torch.equal(model.enhance(tokens,None),tokens)
assert torch.equal(out[[0,2,4,6]],tokens[[0,2,4,6]])
assert torch.all((gate[[1,3,5]]>0)&(gate[[1,3,5]]<1))
assert torch.count_nonzero(gate[[0,2,4,6]])==0
loss=out.sum();loss.backward()
assert model.gate[-1].weight.grad is not None
assert model.gate[-1].weight.grad.abs().sum()>0
assert base.scale.grad is not None and base.scale.grad.abs().sum()>0
for invalid in ([1,1],[7],[-1],[]):
 try:model(tokens,{"indices":torch.tensor(invalid,dtype=torch.long)})
 except ValueError:pass
 else:raise AssertionError("Bad graph indices accepted")
print("PASS gated geometry: identity fallback, residue mask, gradients, index validation")
