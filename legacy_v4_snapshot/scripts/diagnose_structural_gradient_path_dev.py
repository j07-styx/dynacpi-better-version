"""Test whether structural cross-attention receives usable geometry gradients."""
import sys,json,torch
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.centered_geometry import CenteredGeometry
from dynacpi.structural_cross_attention import StructuralCrossAttention

def inspect():
    torch.manual_seed(2103)
    model=StructuralCrossAttention(CenteredGeometry(GeometryBranch()),scale=0.1).cpu()
    graph_path=next((root/'data/engineering_pilot/structure_graphs').glob('*/graph.pt'))
    graph=torch.load(graph_path,map_location='cpu',weights_only=True)
    n=int(graph['indices'].max())+1
    tokens=torch.randn(n,256)
    compounds=torch.randn(3,256)
    if len(sys.argv)>1 and sys.argv[1]=="nonzero":
        torch.nn.init.normal_(model.geometry.geometry.final[-1].weight,std=0.01)
        torch.nn.init.normal_(model.project.weight,std=0.05)
    initial=model(tokens,graph,compounds)
    initial_delta=float((initial-compounds).abs().max())
    initial.square().sum().backward()
    norms={name:float(p.grad.norm()) if p.grad is not None else None for name,p in model.named_parameters()}
    counts={'total_parameters':len(norms),'zero_grad_parameters':sum(v==0 for v in norms.values() if v is not None),
            'none_grad_parameters':sum(v is None for v in norms.values())}
    print(json.dumps({'initial_delta_max':initial_delta,'counts':counts,
      'projection_grad_norm':norms['project.weight'],
      'geometry_final_weight_grad_norm':norms['geometry.geometry.final.1.weight'],
      'attention_out_proj_grad_norm':norms['attention.out_proj.weight'],
      'first_gvp_grad_norm':next((v for k,v in norms.items() if k.startswith('geometry.geometry.nodes') and v is not None),None)},indent=2))
    return initial_delta,norms
if __name__=='__main__':inspect()

