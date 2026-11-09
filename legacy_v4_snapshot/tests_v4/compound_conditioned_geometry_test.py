"""CPU checks for compound-specific gates, gradients, identity and validation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from dynacpi.compound_conditioned_geometry import CompoundConditionedGeometry
class Dummy(nn.Module):
 def __init__(self):
  super().__init__()
  self.weight=nn.Parameter(torch.tensor(0.7))
 def enhance(self,tokens,graph):
  x=tokens.clone()
  return x.index_add(0,graph["indices"],self.weight*torch.ones(len(graph["indices"]),256))
torch.manual_seed(7)
m=CompoundConditionedGeometry(Dummy())
t=torch.randn(8,256,requires_grad=True)
c=torch.randn(3,256,requires_grad=True)
g={"indices":torch.tensor([1,3,6])}
a,w=m(t,g,c,True)
assert a.shape==(3,8,256) and w.shape==(3,8,1)
assert torch.equal(a[:,[0,2,4,5,7]],t[None,:, :].expand(3,-1,-1)[:,[0,2,4,5,7]])
assert torch.count_nonzero(w[:,[0,2,4,5,7]])==0
assert torch.equal(m(t,None,c),t[None].expand(3,-1,-1))
assert torch.count_nonzero(m(t,None,c,True)[1])==0
with torch.no_grad():
 m.gate[-1].weight.fill_(0.005)
assert not torch.allclose(m(t,g,c)[0],m(t,g,c)[1])
a=m(t,g,c)
a.sum().backward()
assert m.geometry.weight.grad is not None and m.geometry.weight.grad.abs().sum()>0
assert m.gate[-1].bias.grad is not None
for idx in ([1,1],[8],[-1],[]):
 try:m(t,{"indices":torch.tensor(idx,dtype=torch.long)},c)
 except ValueError:pass
 else:raise AssertionError(f"accepted {idx}")
# Width is a public constructor parameter, not a hardcoded 256-dimension contract.
small=CompoundConditionedGeometry(Dummy(),width=32,hidden=16)
small_t=torch.randn(5,32)
small_c=torch.randn(2,32)
small_out,small_gate=small(small_t,None,small_c,True)
assert small_out.shape==(2,5,32) and small_gate.shape==(2,5,1)
assert torch.equal(small_out,small_t.unsqueeze(0).expand(2,-1,-1))
for bad_t,bad_c in ((torch.empty(0,32),small_c),(small_t,torch.randn(2,256)),(small_t,small_c.double())):
 try:small(bad_t,None,bad_c)
 except ValueError:pass
 else:raise AssertionError("accepted invalid token/compound shape or dtype")
for width,hidden in ((0,16),(32,0)):
 try:CompoundConditionedGeometry(Dummy(),width=width,hidden=hidden)
 except ValueError:pass
 else:raise AssertionError("accepted invalid module dimensions")
class InvalidEnhancement(nn.Module):
 def __init__(self, mode):
  super().__init__()
  self.mode=mode
 def enhance(self,tokens,graph):
  if self.mode=='shape':return tokens[:-1]
  return torch.full_like(tokens,float('nan'))
for mode in ('shape','nonfinite'):
 try:CompoundConditionedGeometry(InvalidEnhancement(mode))(t,g,c)
 except ValueError:pass
 else:raise AssertionError(f'Accepted invalid GVP enhancement: {mode}')
print("PASS compound conditioned geometry: gates, identity, gradients, shape/dtype validation, invalid and nonfinite enhancements")
