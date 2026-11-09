"""Reconstruct every saved development checkpoint and replay validation predictions."""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.bounded_structural_fusion import BoundedStructuralFusion
from gpu_lease import gpu_lease
def main():
    study=ROOT/"research/bounded_fusion_study_20261107"
    out=study/"checkpoint_replay.json"
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(4)
    sd=torch.load(ROOT/"research/pilot_sequence/checkpoint.pt",map_location="cpu",weights_only=False)
    base=SequenceFoundation().cuda().eval();base.load_state_dict(sd["state"])
    compounds=json.loads((ROOT/"data/staging/compounds.json").read_text())
    fpz=np.load(ROOT/"data/staging/morgan_radius2_2048_chiral.npz")
    fp=dict(zip(fpz["ids"],fpz["bits"]))
    rows=json.loads((study/"baseline.json").read_text())["predictions"]
    cache={}
    with torch.no_grad():
        for sid in sorted({r["sequence"] for r in rows}):
            emb=torch.load(ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(sid+".pt"),weights_only=True)
            tokens=base.sequence(emb["embeddings"].float().cuda())
            g={k:v.cuda() for k,v in torch.load(ROOT/"data/engineering_pilot/structure_graphs"/sid/"graph.pt",weights_only=True).items()}
            rr=[r for r in rows if r["sequence"]==sid]; parts=[]
            for off in range(0,len(rr),8):
                part=rr[off:off+8]
                bits=torch.tensor(np.stack([fp[r["compound"]] for r in part]),device="cuda",dtype=torch.float32)
                c=base.compound([compounds[r["compound"]] for r in part],bits)
                expanded=tokens.unsqueeze(0).expand(len(part),-1,-1)
                residue=base.readout(c.unsqueeze(1),expanded,expanded,need_weights=False)[0][:,0]
                features=base.channel(c,base.norm(expanded.mean(1)+residue),base.norm(expanded.mean(1)+residue))
                parts.append((part,c,features))
            cache[sid]=(tokens,g,parts)
        results=[]
        for run in sorted(study.glob("*_seed_*")):
            saved=torch.load(run/"best.pt",map_location="cuda",weights_only=True)
            variant,seed=saved["variant"],saved["seed"]
            model=BoundedStructuralFusion(confidence=variant!="no_confidence").cuda().eval()
            model.load_state_dict(saved["model"])
            head=copy.deepcopy(base.head).eval();head.load_state_dict(saved["head"])
            expected=json.loads((run/"result.json").read_text())
            lookup={(r["sequence"],r["compound"]):r["pred"] for r in expected["predictions"]}
            errors=[]
            for sid,(tokens,g,parts) in cache.items():
                if variant=="shuffled":
                    gen=torch.Generator().manual_seed(seed+int(sid[:8],16))
                    perm=torch.randperm(len(g["indices"]),generator=gen).cuda()
                    g=dict(g,node_s=g["node_s"][perm],node_v=g["node_v"][perm])
                mode="sequence" if variant=="sequence" else "disabled" if variant=="head_only" else "real"
                for part,c,features in parts:
                    prediction=(head(features).squeeze(-1)+model(tokens,g,c,mode=mode))*sd["label_std"]+sd["label_mean"]
                    errors.extend(abs(float(v)-lookup[(r["sequence"],r["compound"])]) for r,v in zip(part,prediction))
            assert max(errors)<2e-6,(run,max(errors))
            results.append({"run":run.name,"max_absolute_error":max(errors),"pairs":len(errors),"epoch":saved["epoch"]})
    out.write_text(json.dumps({"status":"PASS","runs":results},indent=2))
    print("PASS replay",len(results),"checkpoints; max error",max(r["max_absolute_error"] for r in results))
if __name__=="__main__":
    with gpu_lease(ROOT): main()
