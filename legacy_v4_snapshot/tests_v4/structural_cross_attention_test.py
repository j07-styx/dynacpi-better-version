"""Unit tests for opt-in compound-conditioned structural cross-attention."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from dynacpi.structural_cross_attention import StructuralCrossAttention

class ToyGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(0.7))
    def enhance(self,tokens,graph):
        out=tokens.clone()
        return out.index_add(0,graph['indices'],self.weight*graph['offsets'])

torch.manual_seed(9)
model=StructuralCrossAttention(ToyGeometry(),width=8,heads=2)
tokens=torch.randn(7,8)
compounds=torch.randn(3,8)
graph={'indices':torch.tensor([1,3,5]),'offsets':torch.randn(3,8)}
assert torch.equal(model(tokens,None,compounds),compounds)
out=model(tokens,graph,compounds)
assert out.shape==compounds.shape and torch.isfinite(out).all()
assert not torch.allclose(out,compounds,atol=1e-8)
(out-compounds).square().sum().backward()
assert model.geometry.weight.grad is not None and model.geometry.weight.grad.abs()>0
assert model.project.weight.grad is not None and model.project.weight.grad.abs().sum()>0
for indices in ([1,1],[7],[-1],[]):
    try:model(tokens,{**graph,'indices':torch.tensor(indices)},compounds)
    except ValueError:pass
    else:raise AssertionError('Accepted invalid indices')
print('PASS structural cross attention: no-graph identity, shape, structural signal, gradients and index guards')
