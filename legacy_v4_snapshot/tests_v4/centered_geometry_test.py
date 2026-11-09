"""Test zero-mean residual, no-geometry identity, gradients and invalid indices."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from dynacpi.centered_geometry import CenteredGeometry
class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w=nn.Parameter(torch.tensor(0.3))
    def enhance(self,t,g):
        z=t.clone()
        idx=g['indices']
        return z.index_add(0,idx,self.w*t.index_select(0,idx)+1.0)
torch.manual_seed(8)
t=torch.randn(7,256,requires_grad=True)
g={'indices':torch.tensor([1,3,5])}
m=CenteredGeometry(Dummy())
y=m.enhance(t,g)
assert y.shape==t.shape
assert torch.equal(y[[0,2,4,6]],t[[0,2,4,6]])
assert torch.allclose((y-t)[g['indices']].mean(0),torch.zeros(256),atol=1e-6)
assert torch.equal(m.enhance(t,None),t)
y.square().sum().backward()
assert m.geometry.w.grad is not None and m.geometry.w.grad.abs()>0
for idx in ([1,1],[7],[-1],[]):
    try:m.enhance(t,{'indices':torch.tensor(idx)},)
    except ValueError:pass
    else:raise AssertionError(idx)
class InvalidGeometry(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode=mode
    def enhance(self,t,g):
        if self.mode=='shape':
            return t[:-1]
        return torch.full_like(t,float('nan'))
for mode in ('shape','nonfinite'):
    try:CenteredGeometry(InvalidGeometry(mode)).enhance(t,g)
    except ValueError:pass
    else:raise AssertionError(f'Accepted invalid geometry output: {mode}')
print('PASS centered geometry: mean-zero structural correction, exact uncovered identity, gradients, invalid indices and invalid enhancements')
