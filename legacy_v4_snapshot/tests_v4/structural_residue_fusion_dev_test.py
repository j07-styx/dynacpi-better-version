"""Check residue fusion shape, missing-graph identity, and geometry gradient flow."""
import sys,torch
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.centered_geometry import CenteredGeometry
from dynacpi.structural_residue_fusion_dev import StructuralResidueFusion
model=StructuralResidueFusion(CenteredGeometry(GeometryBranch()))
torch.manual_seed(2103)
with torch.no_grad():torch.nn.init.normal_(model.geometry.geometry.final[-1].weight,std=.01)
path=next((root/'data/engineering_pilot/structure_graphs').glob('*/graph.pt'))
graph=torch.load(path,map_location='cpu',weights_only=True)
tokens=torch.randn(int(graph['indices'].max())+1,256)
assert torch.equal(model(tokens,None),tokens)
out=model(tokens,graph)
assert out.shape==tokens.shape and torch.isfinite(out).all()
assert not torch.equal(out,tokens)
out.square().mean().backward()
for name in ('project.weight','geometry.geometry.final.1.weight'):
    grad=dict(model.named_parameters())[name].grad
    assert grad is not None and grad.norm()>0 and torch.isfinite(grad).all(),name
print('PASS residue fusion identity, finite outputs, and geometry gradients')
