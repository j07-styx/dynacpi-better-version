"""Regression test: both nonzero projections permit gradient flow to GVP and attention."""
import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.centered_geometry import CenteredGeometry
from dynacpi.structural_cross_attention import StructuralCrossAttention
root=Path(__file__).resolve().parents[1]
torch.manual_seed(2103)
model=StructuralCrossAttention(CenteredGeometry(GeometryBranch()),scale=0.1)
with torch.no_grad():
    torch.nn.init.normal_(model.geometry.geometry.final[-1].weight,std=0.01)
    torch.nn.init.normal_(model.project.weight,std=0.05)
path=next((root/'data/engineering_pilot/structure_graphs').glob('*/graph.pt'))
graph=torch.load(path,map_location='cpu',weights_only=True)
tokens=torch.randn(int(graph['indices'].max())+1,256)
compounds=torch.randn(3,256)
out=model(tokens,graph,compounds)
assert (out-compounds).abs().max()>0
out.square().sum().backward()
for key in ('project.weight','attention.out_proj.weight','geometry.geometry.final.1.weight'):
    param=dict(model.named_parameters())[key]
    assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0,key
first=[p.grad for name,p in model.named_parameters() if name.startswith('geometry.geometry.nodes') and p.grad is not None]
assert first and any(g.norm()>0 for g in first)
print('PASS nonzero geometry+attention initialization restores deep gradient path')
