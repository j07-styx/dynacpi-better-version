"""Compare frozen sequence, original GVP and gated GVP on the same DAVIS dev cohort."""
import json,sys,math
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));sys.path.insert(0,str(root/"scripts"))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.confidence_gated_geometry import ConfidenceGatedGeometry
from gpu_lease import gpu_lease
def run():
 p=root/"data/engineering_pilot"
 ck=lambda x:torch.load(root/x,map_location="cpu",weights_only=False)
 seq=ck("research/pilot_sequence/checkpoint.pt")
 old=ck("research/pilot_geometry_9train_4val/best_checkpoint.pt")
 new=ck("research/pilot_gated_geometry_dev/seed_1108/best_checkpoint.pt")
 base=SequenceFoundation().cuda().eval();base.load_state_dict(seq["state"])
 g=GeometryBranch().cuda().eval();g.load_state_dict(old["state"])
 gated=ConfidenceGatedGeometry(GeometryBranch()).cuda().eval();gated.load_state_dict(new["state"])
 manifest=json.loads((p/"limited_alpha_3d_graph_manifest.json").read_text())
 ids={r["sequence_sha256"] for r in manifest["records"] if r["status"]=="GRAPH_MAPPED_ENGINEERING_ONLY" and r["pilot_split"]=="val"}
 rows=[r for r in json.loads((p/"rows.json").read_text()) if r["split"]=="val" and r["sequence"] in ids]
 smiles=json.loads((root/"data/staging/compounds.json").read_text())
 f=np.load(root/"data/staging/morgan_radius2_2048_chiral.npz");fp={str(k):v for k,v in zip(f["ids"],f["bits"])}
 errors={k:defaultdict(list) for k in ("sequence","ungated","gated")};weights=[]
 with torch.inference_mode():
  for tid in sorted(ids):
   emb=torch.load(root/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(tid+".pt"),weights_only=True,map_location="cpu")["embeddings"].float().cuda()
   graph=torch.load(p/"structure_graphs"/tid/"graph.pt",weights_only=True,map_location="cpu")
   t=base.sequence(emb);a=g.enhance(t,graph);b,w=gated(t,graph,True)
   weights.extend(w[graph["indices"].long().cuda()].flatten().cpu().tolist())
   subset=[r for r in rows if r["sequence"]==tid]
   for i in range(0,len(subset),8):
    rr=subset[i:i+8];ss=[smiles[r["compound"]] for r in rr]
    bits=torch.tensor(np.stack([fp[r["compound"]] for r in rr]),dtype=torch.float32,device="cuda")
    c=base.compound(ss,bits)
    for name,tokens in (("sequence",t),("ungated",a),("gated",b)):
     mask=torch.ones((len(rr),len(tokens)),dtype=torch.bool,device="cuda")
     y=base.predict_tokens(c,tokens[None].expand(len(rr),-1,-1),mask).cpu().numpy()*seq["label_std"]+seq["label_mean"]
     for r,pred in zip(rr,y):errors[name][tid].append((float(pred)-float(r["y"]))**2)
 assert len(ids)==4 and len(rows)==216
 metric=lambda z:math.sqrt(sum(z)/len(z))
 results={k:{"macro_rmse":sum(metric(v) for v in groups.values())/4,"pooled_rmse":metric([e for group in groups.values() for e in group])} for k,groups in errors.items()}
 report={"status":"DAVIS_DEV_GATED_COMPARISON","validation_proteins":4,"validation_pairs":216,"models":results,"mean_gate":float(np.mean(weights)),"scientific_claim":False}
 out=root/"research/pilot_gated_geometry_dev/seed_1108/comparison.json";out.write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":
 with gpu_lease(root):run()
