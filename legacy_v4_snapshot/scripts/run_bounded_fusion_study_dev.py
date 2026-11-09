"""Predeclared matched development study. Never reads independent test labels."""
import argparse, copy, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.bounded_structural_fusion import BoundedStructuralFusion
from gpu_lease import gpu_lease

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(rows, pred):
    err = (np.asarray(pred)-np.array([r["y"] for r in rows]))**2
    per = {p:float(np.sqrt(np.mean([e for r,e in zip(rows,err) if r["sequence"] == p])))
           for p in sorted({r["sequence"] for r in rows})}
    return {"macro_rmse":float(np.mean(list(per.values()))), "pooled_rmse":float(np.sqrt(err.mean())), "per_target":per}

def main(out):
    out.mkdir(parents=True, exist_ok=False)
    cfg = {"seeds":[1201,1202,1203], "variants":["real","shuffled","sequence","no_confidence","head_only","real_head"],
           "epochs":8, "lr":0.001, "weight_decay":0.01, "cap_standardized":.25,
           "checkpoint_selection":"minimum validation macro RMSE including epoch zero", "status":"DEVELOPMENT_ONLY"}
    (out/"protocol.json").write_text(json.dumps(cfg,indent=2))
    torch.set_num_threads(4)
    pilot = ROOT/"data/engineering_pilot"
    ckpt = ROOT/"research/pilot_sequence/checkpoint.pt"
    sd = torch.load(ckpt,map_location="cpu",weights_only=False)
    assert sd["row_sha256"] == sha(pilot/"rows.json")
    base = SequenceFoundation().cuda().eval()
    base.load_state_dict(sd["state"],strict=True)
    for p in base.parameters(): p.requires_grad_(False)
    mean,std = sd["label_mean"],sd["label_std"]
    records = json.loads((pilot/"limited_alpha_3d_graph_manifest.json").read_text())["records"]
    records = {r["sequence_sha256"]:r for r in records if r["status"]=="GRAPH_MAPPED_ENGINEERING_ONLY"}
    rows = [r for r in json.loads((pilot/"rows.json").read_text()) if r["sequence"] in records]
    assert all(r["split"] == records[r["sequence"]]["pilot_split"] for r in rows)
    compounds = json.loads((ROOT/"data/staging/compounds.json").read_text())
    fpz = np.load(ROOT/"data/staging/morgan_radius2_2048_chiral.npz")
    fp = dict(zip(fpz["ids"],fpz["bits"]))
    cache, batches = {}, {"train":[],"val":[]}
    for sid in sorted(records):
        path = pilot/"structure_graphs"/sid/"graph.pt"
        assert sha(path) == records[sid]["graph_sha256"]
        graph = {k:v.cuda() for k,v in torch.load(path,weights_only=True).items()}
        emb = torch.load(ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(sid+".pt"),weights_only=True)
        assert emb["sequence_hash"] == sid
        with torch.no_grad(): tokens = base.sequence(emb["embeddings"].float().cuda()).detach()
        cache[sid] = (tokens, graph)
        rr = [r for r in rows if r["sequence"]==sid]
        for off in range(0,len(rr),8):
            part = rr[off:off+8]
            bits = torch.tensor(np.stack([fp[r["compound"]] for r in part]),device="cuda",dtype=torch.float32)
            with torch.no_grad():
                c = base.compound([compounds[r["compound"]] for r in part],bits).detach()
                expanded = tokens.unsqueeze(0).expand(len(part),-1,-1)
                mask = torch.ones(expanded.shape[:2],device="cuda",dtype=torch.bool)
                residue = base.readout(c.unsqueeze(1),expanded,expanded,key_padding_mask=~mask,need_weights=False)[0][:,0]
                protein = base.norm(expanded.mean(1)+residue)
                features = base.channel(c,protein,protein).detach()
                pred = base.head(features).squeeze(-1).detach()
                direct = base.predict_tokens(c,expanded,mask)
                assert torch.allclose(pred,direct,atol=1e-6)
            batches[part[0]["split"]].append({"sid":sid,"rows":part,"c":c,"features":features,"baseline":pred,
                "y":torch.tensor([(r["y"]-mean)/std for r in part],device="cuda")})
    valid_rows = [r for b in batches["val"] for r in b["rows"]]
    baseline = [x for b in batches["val"] for x in (b["baseline"]*std+mean).tolist()]
    baseline_metrics = metrics(valid_rows,baseline)
    (out/"baseline.json").write_text(json.dumps({"metrics":baseline_metrics,"checkpoint_sha256":sha(ckpt),
        "rows_sha256":sha(pilot/"rows.json"),"predictions":[dict(r,pred=p) for r,p in zip(valid_rows,baseline)]},indent=2))
    print("BASELINE",json.dumps(baseline_metrics),flush=True)
    all_results = []
    for seed in cfg["seeds"]:
        for variant in cfg["variants"]:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model = BoundedStructuralFusion(confidence=variant!="no_confidence").cuda()
            head = copy.deepcopy(base.head).eval()
            adapt_head = variant in ("head_only","real_head")
            for p in head.parameters(): p.requires_grad_(False)
            if adapt_head:
                for p in head[-1].parameters(): p.requires_grad_(True)
            params = list(model.parameters()) + [p for p in head.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params,lr=cfg["lr"],weight_decay=cfg["weight_decay"])
            graphs = {}
            for sid,(_,g) in cache.items():
                graphs[sid] = g
                if variant=="shuffled":
                    gen = torch.Generator().manual_seed(seed+int(sid[:8],16))
                    perm = torch.randperm(len(g["indices"]),generator=gen).cuda()
                    graphs[sid] = dict(g,node_s=g["node_s"][perm],node_v=g["node_v"][perm])
            mode = "sequence" if variant=="sequence" else "disabled" if variant=="head_only" else "real"
            def predict(b):
                tokens,_ = cache[b["sid"]]
                residual = model(tokens,graphs[b["sid"]],b["c"],mode=mode)
                return head(b["features"]).squeeze(-1)+residual if adapt_head else b["baseline"]+residual
            def evaluate(split):
                model.eval(); pred=[]; rr=[]
                with torch.no_grad():
                    for b in batches[split]:
                        pred.extend((predict(b)*std+mean).tolist()); rr.extend(b["rows"])
                return metrics(rr,pred),pred
            history=[]; best=float("inf"); best_epoch=0; best_pred=None; gradients=[]
            run = out/(variant+"_seed_"+str(seed)); run.mkdir()
            for epoch in range(cfg["epochs"]+1):
                if epoch:
                    model.train()
                    order=list(range(len(batches["train"]))); random.Random(seed+epoch).shuffle(order)
                    for ix in order:
                        b=batches["train"][ix]
                        optimizer.zero_grad(set_to_none=True)
                        pred=predict(b)
                        loss=(pred-b["y"]).square().mean()+.25*(pred-b["baseline"]).square().mean()
                        loss.backward()
                        if ix==order[0]:
                            gradients.append({"epoch":epoch,"norms":{n:None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters()}})
                        nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True); optimizer.step()
                train,_=evaluate("train"); val,pred=evaluate("val")
                history.append({"epoch":epoch,"train":train,"val":val})
                if val["macro_rmse"] < best:
                    best=val["macro_rmse"];best_epoch=epoch;best_pred=pred
                    torch.save({"model":model.state_dict(),"head":head.state_dict(),"variant":variant,"seed":seed,"epoch":epoch},run/"best.pt")
            result={"seed":seed,"variant":variant,"best_epoch":best_epoch,"best_macro_rmse":best,
                "delta_vs_baseline":best-baseline_metrics["macro_rmse"],"history":history,"gradients":gradients,
                "predictions":[dict(r,pred=p) for r,p in zip(valid_rows,best_pred)]}
            (run/"result.json").write_text(json.dumps(result,indent=2))
            all_results.append({k:v for k,v in result.items() if k not in ("history","gradients","predictions")})
            (out/"results.json").write_text(json.dumps(all_results,indent=2))
            print(json.dumps(all_results[-1]),flush=True)
    assert all(p.grad is None for p in base.parameters())
    summary={}
    for v in cfg["variants"]:
        vv=[r["best_macro_rmse"] for r in all_results if r["variant"]==v]
        dd=np.array(vv)-baseline_metrics["macro_rmse"]
        summary[v]={"mean":float(np.mean(vv)),"sd":float(np.std(vv,ddof=1)),
                    "mean_paired_delta":float(dd.mean()),"relative_reduction":float(-dd.mean()/baseline_metrics["macro_rmse"])}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("COMPLETE",json.dumps(summary),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    with gpu_lease(ROOT): main(a.output)
