"""Development-only real residue-level GVP structure training on reconciled DAVIS numeric Kd.

Uses 106 numeric-Kd training pairs and 71 numeric-Kd validation pairs,
seed 2103 and a frozen numeric-only sequence foundation. AlphaFold graph
identity is name mapped, not assay-construct verified. The development
validation subset has been repeatedly inspected and is not an independent
test; this script cannot establish structural superiority.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.centered_geometry import CenteredGeometry
from dynacpi.structural_residue_fusion_dev import StructuralResidueFusion
from gpu_lease import gpu_lease

def hashfile(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(2*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def run():
    seed=int(os.environ.get("DYNACPI_ENGINEERING_SEED","2103"))
    if seed not in (2103,): raise ValueError("Only reconciled residue fusion DEV seed 2103 allowed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    rootpilot=root/"data/engineering_pilot_primary_reconciled_20261107"
    graph_root=root/"data/engineering_pilot/structure_graphs"
    graphmeta=json.loads((root/"data/engineering_pilot/limited_alpha_3d_graph_manifest.json").read_text(encoding="utf-8"))
    approved={r["sequence_sha256"]:r for r in graphmeta["records"] if r["status"]=="GRAPH_MAPPED_ENGINEERING_ONLY"}
    train_ids={p for p,r in approved.items() if r["pilot_split"]=="train"}
    val_ids={p for p,r in approved.items() if r["pilot_split"]=="val"}
    if len(train_ids)!=9 or len(val_ids)!=4 or train_ids&val_ids:
        raise ValueError("Expected fixed 9/4 engineering 3D cohort")
    pilotrows=json.loads((rootpilot/"rows.json").read_text(encoding="utf-8"))
    groups={"train":defaultdict(list),"val":defaultdict(list)}
    for row in pilotrows:
        p=row["sequence"]
        if p in (train_ids if row["split"]=="train" else val_ids):
            groups[row["split"]][p].append(row)
    for p in groups["train"]: groups["train"][p]=[r for r in groups["train"][p] if r["primary_label_kind"]=="numeric_Kd"]
    if set(groups["train"])!=train_ids or set(groups["val"])!=val_ids:
        raise ValueError("3D mapping/pilot split inconsistency")
    compounds=json.loads((root/"data/staging/compounds.json").read_text(encoding="utf-8"))
    sequences=json.loads((root/"data/staging/sequences.json").read_text(encoding="utf-8"))
    fpz=np.load(root/"data/staging/morgan_radius2_2048_chiral.npz")
    fp={str(x):y for x,y in zip(fpz["ids"],fpz["bits"])}
    seq_ckpt=root/"research/reconciled_numeric_sequence_seed_2103_dev/checkpoint.pt"
    seq_data=torch.load(seq_ckpt,map_location="cpu",weights_only=False)
    assert seq_data["kind"]=="sequence" and seq_data["row_sha256"]==hashfile(rootpilot/"rows.json")
    mean=float(seq_data["label_mean"]);std=float(seq_data["label_std"])
    base=SequenceFoundation().cuda().eval()
    base.load_state_dict(seq_data["state"],strict=True)
    for p in base.parameters():p.requires_grad_(False)
    geom=StructuralResidueFusion(CenteredGeometry(GeometryBranch()),scale=0.1).cuda().train()
    nn.init.normal_(geom.geometry.geometry.final[-1].weight,std=0.01)
    nn.init.normal_(geom.project.weight,std=0.05)
    optimizer=torch.optim.AdamW(geom.parameters(),lr=3e-4,weight_decay=3e-3)
    dev="cuda"

    input_cache={}
    for p in sorted(train_ids|val_ids):
        seq=sequences[p]
        obj=torch.load(root/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(p+".pt"),
                       map_location="cpu",weights_only=True)
        assert obj["sequence_hash"]==p and obj["embeddings"].shape==(len(seq),1280)
        graph_file=graph_root/p/"graph.pt"
        assert hashfile(graph_file)==approved[p]["graph_sha256"]
        graph=torch.load(graph_file,map_location="cpu",weights_only=True)
        with torch.inference_mode():
            tokens=base.sequence(obj["embeddings"].float().cuda()).detach()
        assert tokens.shape==(len(seq),256)
        input_cache[p]={"tokens":tokens,"graph":graph}
    def pieces(rows,width=8):
        for i in range(0,len(rows),width):
            yield rows[i:i+width]

    def predict_pair(rows,p,train):
        sm=[compounds[r["compound"]] for r in rows]
        bits=torch.tensor(np.stack([fp[r["compound"]] for r in rows]),
                          device=dev,dtype=torch.float32)
        with torch.no_grad():
            c=base.compound(sm,bits).detach()
        tokens=input_cache[p]["tokens"]
        batch=len(rows)
        mask=torch.ones((batch,len(tokens)),device=dev,dtype=torch.bool)
        c_struct=c
        structured_tokens=geom(tokens,input_cache[p]["graph"])
        if train:
            output=base.predict_tokens(c_struct,structured_tokens.unsqueeze(0).expand(batch,-1,-1),mask)
            with torch.no_grad():
                raw=base.predict_tokens(c,tokens.unsqueeze(0).expand(batch,-1,-1),mask)
            return output,raw.detach()
        with torch.no_grad():
            raw=base.predict_tokens(c,tokens.unsqueeze(0).expand(batch,-1,-1),mask)
            grown=base.predict_tokens(c_struct,structured_tokens.unsqueeze(0).expand(batch,-1,-1),mask)
        return raw.detach().cpu().numpy()*std+mean,grown.detach().cpu().numpy()*std+mean

    directory=root/"research/reconciled_numeric_residue_fusion_dev"/("seed_"+str(seed))
    directory.mkdir(parents=True,exist_ok=False)
    history=[]
    best=float("inf")
    stale=0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,5):
        start=time.monotonic()
        geom.train()
        train_batches=[]
        for p in sorted(train_ids):
            for rr in pieces(groups["train"][p]):
                train_batches.append((p,rr))
        random.Random(seed+epoch).shuffle(train_batches)
        total_loss=0.0
        for p,rr in train_batches:
            optimizer.zero_grad(set_to_none=True)
            pred,baseline=predict_pair(rr,p,train=True)
            targets=torch.tensor([(float(r["y"])-mean)/std for r in rr],
                                 device=dev,dtype=torch.float32)
            fit_loss=(pred-targets).square().mean()
            anchor_loss=(pred-baseline).square().mean()
            loss=fit_loss+0.25*anchor_loss
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite geometry training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(geom.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            total_loss+=float(loss.detach())*len(rr)
        if any(p.grad is not None for p in base.parameters()):
            raise RuntimeError("Frozen base unexpectedly received gradients")
        geom.eval()
        errors={"seq":[],"geom":[]}
        bytarget={"seq":defaultdict(list),"geom":defaultdict(list)}
        for p in sorted(val_ids):
            for rr in pieces([r for r in groups["val"][p] if r["primary_label_kind"]=="numeric_Kd"]):
                baseline,enhanced=predict_pair(rr,p,train=False)
                for r,s,g in zip(rr,baseline,enhanced):
                    ys=float(r["y"])
                    for key,value in (("seq",s),("geom",g)):
                        err=float(value-ys)**2
                        errors[key].append(err)
                        bytarget[key][p].append(err)
        def calc(key):
            pooled=math.sqrt(float(np.mean(errors[key])))
            macro=float(np.mean([math.sqrt(float(np.mean(v))) for v in bytarget[key].values()]))
            return pooled,macro
        s_pooled,s_macro=calc("seq")
        g_pooled,g_macro=calc("geom")
        item={"epoch":epoch,"validation_sequence_pooled_rmse":s_pooled,
              "validation_sequence_target_macro_rmse":s_macro,
              "validation_geometry_pooled_rmse":g_pooled,
              "validation_geometry_target_macro_rmse":g_macro,
              "train_standardized_mse":total_loss/sum(len(g) for g in groups["train"].values()),
              "elapsed_seconds":round(time.monotonic()-start,2),
              "peak_cuda_mib":round(torch.cuda.max_memory_allocated()/2**20,2)}
        print(json.dumps(item),flush=True)
        history.append(item)
        (directory/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        if g_macro<best-1e-5:
            best=g_macro
            stale=0
            tmp=directory/"best_checkpoint.tmp"
            torch.save({
              "state":geom.state_dict(),"architecture":"structural_residue_fusion_dev","epoch":epoch,
              "geometry_train_component_count":len(train_ids),
              "geometry_val_component_count":len(val_ids),
              "sequence_checkpoint_sha256":hashfile(seq_ckpt),
              "manifest_sha256":hashfile(root/"data/engineering_pilot/limited_alpha_3d_graph_manifest.json"),
              "row_sha256":hashfile(rootpilot/"rows.json"),
              "label_mean":mean,"label_std":std,
              "status":"RECONCILED_NUMERIC_RESIDUE_FUSION_DEV_NOT_INDEPENDENT_TEST"
            },tmp)
            os.replace(tmp,directory/"best_checkpoint.pt")
        else:
            stale+=1
        if stale>=999:
            print("EARLY_STOP: four epochs without macro RMSE improvement",flush=True)
            break
    report={"status":"COMPLETE_RECONCILED_NUMERIC_RESIDUE_DEV_NOT_INDEPENDENT_TEST",
        "train_protein_count":len(train_ids),"validation_protein_count":len(val_ids),
        "train_pair_count":sum(map(len,groups["train"].values())),
        "val_pair_count":sum(sum(r["primary_label_kind"]=="numeric_Kd" for r in g) for g in groups["val"].values()),
        "engineering_seed":seed, "best_epoch":min(history,key=lambda e:e["validation_geometry_target_macro_rmse"])["epoch"],
        "sequence_checkpoint_sha256":hashfile(seq_ckpt),
        "geometry_best_checkpoint_sha256":hashfile(directory/"best_checkpoint.pt"),
        "history":history,
        "limitations":[
          "Only publisher-verified numeric-Kd cells are used in fitting and validation; blank screening-negative cells are excluded. The cohort is not family-disjoint.",
          "AlphaFold geometry is name-mapped and is not verified against experimental assay constructs.",
          "The geometry-eligible cohort is a selected subset; comparisons outside that common cohort are invalid.",
          "Single-seed repeatedly inspected development validation; insufficient to infer structural benefit or publish an independent test result."
        ]}
    (directory/"manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"status":report["status"],"best_epoch":report["best_epoch"],
                      "train_pairs":report["train_pair_count"],"val_pairs":report["val_pair_count"]}),flush=True)

if __name__=="__main__":
    with gpu_lease(root):
        run()




