import json,sys
from pathlib import Path
import numpy as np,torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));sys.path.insert(0,str(root/"scripts"))
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.backbone_graph import make_graph
from gpu_lease import gpu_lease
def main():
 base=root/"data"/"structure_fixtures"/"P15056"
 record=json.loads((base/"metadata.json").read_text())[0]
 seq=record["uniprotSequence"];L=96
 xyz=np.full((L,4,3),np.nan);conf=np.full(L,np.nan)
 chain=PDBParser(QUIET=True).get_structure("BRAF",base/"structure.pdb")[0]["A"]
 for residue in chain:
  j=residue.id[1]-1
  if 0<=j<L and residue.id[0]==" ":
   assert seq1(residue.resname)==seq[j]
   if all(k in residue for k in ("N","CA","C","O")):
    xyz[j]=[residue[k].coord for k in ("N","CA","C","O")];conf[j]=residue["CA"].bfactor
 pae=np.array(json.loads((base/"pae.json").read_text())[0]["predicted_aligned_error"])[:L,:L]
 def graph(x):return make_graph(x,np.arange(L),["A"]*L,conf,pae)
 g=graph(xyz);assert g is not None
 torch.manual_seed(1105);model=GeometryBranch().cuda().eval()
 t=torch.randn(L,256,device="cuda");weight=torch.randn(256,device="cuda")
 before=model.enhance(t,g)
 assert torch.equal(before,t) and torch.equal(model.enhance(t,None),t)
 (before@weight).mean().backward()
 projection_grad=float(model.final[-1].weight.grad.norm());assert projection_grad>0
 with torch.no_grad():model.final[-1].weight.normal_(std=.005)
 model.zero_grad(set_to_none=True)
 enhanced=model.enhance(t,g);(enhanced@weight).mean().backward()
 gvp_grad=float(sum(p.grad.norm().item() for p in model.nodes.parameters() if p.grad is not None))
 assert np.isfinite(gvp_grad) and gvp_grad>0
 Q,_=np.linalg.qr(np.random.default_rng(7).normal(size=(3,3)))
 if np.linalg.det(Q)<0:Q[:,0]*=-1
 rotated=graph(xyz@Q+np.array([5.,-3.,1.]))
 with torch.no_grad():b=model.enhance(t,rotated);a=model.enhance(t,g)
 diff=float((a-b).abs().max());torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
 broken=xyz.copy();broken[10]=np.nan;gb=graph(broken);assert 10 not in gb["indices"].tolist()
 info={"scope":"First 96 residues of canonical AF-P15056-F1 v6, real geometry and PAE; synthetic sequence tokens; no affinity measurements","nodes":len(g["indices"]),"directed_edges":g["edges"].shape[1],"projection_grad":projection_grad,"gvp_input_grad":gvp_grad,"max_rot_trans_token_difference":diff,"exact_init_and_missing_fallback":True,"missing_node_excluded":True,"device":torch.cuda.get_device_name()}
 (root/"research"/"geometry_smoke.json").write_text(json.dumps(info,indent=2));print(json.dumps(info))
if __name__=="__main__":
 with gpu_lease(root):main()
