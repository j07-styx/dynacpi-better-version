import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dynacpi.bounded_structural_fusion import BoundedStructuralFusion
torch.manual_seed(7)
m = BoundedStructuralFusion().eval()
tokens, c = torch.randn(8,256), torch.randn(3,256)
g = {"indices":torch.arange(8), "node_s":torch.rand(8,14), "node_v":torch.randn(8,3,3)}
assert torch.equal(m(tokens,None,c), torch.zeros(3))
assert torch.equal(m(tokens,g,c), torch.zeros(3))
with torch.no_grad(): m.output[-1].weight.normal_(std=.1)
y, weights = m(tokens,g,c,return_attention=True)
assert y.abs().max() <= m.cap
assert torch.allclose(weights.sum(-1), torch.ones(3))
assert torch.allclose(y, torch.cat([m(tokens,g,c[i:i+1]) for i in range(3)]), atol=1e-6)
perm = torch.randperm(8)
gp = {k:v[perm] for k,v in g.items()}
assert torch.allclose(y,m(tokens,gp,c),atol=1e-6)
rotation = torch.linalg.qr(torch.randn(3,3))[0]
gr = dict(g,node_v=g["node_v"]@rotation)
assert torch.allclose(y,m(tokens,gr,c),atol=1e-6)
y.sum().backward()
assert m.geometry[0].weight.grad.abs().sum() > 0
assert m.query.weight.grad.abs().sum() > 0
assert torch.equal(m(tokens,g,c,mode="disabled"),torch.zeros(3))
print("PASS boundedness, fallback, batch independence, permutation, rotation, gradient flow")
