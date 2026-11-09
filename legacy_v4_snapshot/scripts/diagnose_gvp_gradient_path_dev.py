"""One-batch gradient probe of the existing GVP attention path; not model training."""
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.centered_geometry import CenteredGeometry
from dynacpi.structural_cross_attention import StructuralCrossAttention
from gpu_lease import gpu_lease
def main():
    output=ROOT/"research/gvp_gradient_probe_20261107.json"
    if output.exists():raise FileExistsError(output)
    torch.manual_seed(1201)
    sd=torch.load(ROOT/"research/pilot_sequence/checkpoint.pt",map_location="cpu",weights_only=False)
    base=SequenceFoundation().cuda().eval();base.load_state_dict(sd["state"])
    for p in base.parameters():p.requires_grad_(False)
    records=json.loads((ROOT/"data/engineering_pilot/limited_alpha_3d_graph_manifest.json").read_text())["records"]
    sid=sorted(r["sequence_sha256"] for r in records if r["pilot_split"]=="train")[0]
    rows=[r for r in json.loads((ROOT/"data/engineering_pilot/rows.json").read_text()) if r["sequence"]==sid][:8]
    compounds=json.loads((ROOT/"data/staging/compounds.json").read_text())
    pack=np.load(ROOT/"data/staging/morgan_radius2_2048_chiral.npz");fp=dict(zip(pack["ids"],pack["bits"]))
    emb=torch.load(ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(sid+".pt"),weights_only=True)
    graph=torch.load(ROOT/"data/engineering_pilot/structure_graphs"/sid/"graph.pt",weights_only=True)
    with torch.no_grad():
        tokens=base.sequence(emb["embeddings"].float().cuda()).detach()
        c=base.compound([compounds[r["compound"]] for r in rows],torch.tensor(np.stack([fp[r["compound"]] for r in rows]),device="cuda",dtype=torch.float32))
    model=StructuralCrossAttention(CenteredGeometry(GeometryBranch())).cuda().eval()
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5)
    target=torch.tensor([(r["y"]-sd["label_mean"])/sd["label_std"] for r in rows],device="cuda")
    q=tokens.unsqueeze(0).expand(len(rows),-1,-1);mask=torch.ones(q.shape[:2],device="cuda",dtype=torch.bool)
    with torch.no_grad():baseline=base.predict_tokens(c,q,mask)
    results=[]
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        pred=base.predict_tokens(model(tokens,graph,c),q,mask)
        (pred-target).square().mean().backward()
        groups={}
        for name,p in model.named_parameters():
            group="gvp" if ".layers." in name or ".nodes." in name or ".edges." in name else "projection" if ".final." in name else "attention" if name.startswith("attention") else "other"
            groups.setdefault(group,[]).append(0.0 if p.grad is None else float(p.grad.norm()))
        results.append({"optimizer_steps_before":step,"mean_absolute_prediction_change":float((pred-baseline).abs().mean()),
                        "gradient_l2_by_group":{k:float(np.linalg.norm(v)) for k,v in groups.items()}})
        opt.step()
    assert all(p.grad is None for p in base.parameters())
    output.write_text(json.dumps({"scope":"Single training batch, deterministic eval-mode diagnostic; no checkpoints replaced",
        "measurements":results,"note":"Zero-initialized GVP final projection initially blocks gradients to earlier GVP layers; later nonzero gradients distinguish temporary initialization from a permanently dead path."},indent=2))
    print(output.read_text())
if __name__=="__main__":
    with gpu_lease(ROOT):main()
