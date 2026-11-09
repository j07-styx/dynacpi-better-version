"""Re-evaluate saved GVP engineering pilot on the identical held-out pilot rows.

This is a checkpoint-integrity and failure-analysis test ONLY. It does not
measure generalization to genuinely unseen protein families or a source-
verified BindingDB assay cohort. Training/validation chemistry remains DAVIS.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import torch

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/"scripts"))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from gpu_lease import gpu_lease

def h(path):
    dig=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2*1024*1024),b""):dig.update(b)
    return dig.hexdigest()

def run():
    cohort=root/"data/engineering_pilot"
    seqckpt=root/"research/pilot_sequence/checkpoint.pt"
    geo_dir=root/"research/pilot_geometry_9train_4val"
    geopth=geo_dir/"best_checkpoint.pt"
    record=torch.load(geopth,map_location="cpu",weights_only=False)
    original=torch.load(seqckpt,map_location="cpu",weights_only=False)
    manifest=cohort/"limited_alpha_3d_graph_manifest.json"
    assert record["sequence_checkpoint_sha256"]==h(seqckpt)
    assert record["manifest_sha256"]==h(manifest)
    assert record["row_sha256"]==h(cohort/"rows.json")
    assert record["epoch"]==1
    lm=float(original["label_mean"]);ls=float(original["label_std"])
    assert record["label_mean"]==lm and record["label_std"]==ls
    model=SequenceFoundation().cuda().eval()
    model.load_state_dict(original["state"],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    geometry=GeometryBranch().cuda().eval()
    geometry.load_state_dict(record["state"],strict=True)
    for p in geometry.parameters():p.requires_grad_(False)
    fingerprints=np.load(root/"data/staging/morgan_radius2_2048_chiral.npz")
    fps={str(k):v for k,v in zip(fingerprints["ids"],fingerprints["bits"])}
    compounds=json.loads((root/"data/staging/compounds.json").read_text(encoding="utf-8"))
    splitrows=json.loads((cohort/"rows.json").read_text(encoding="utf-8"))
    meta=json.loads(manifest.read_text(encoding="utf-8"))
    selected={r["sequence_sha256"]:r for r in meta["records"] if r["status"]=="GRAPH_MAPPED_ENGINEERING_ONLY" and r["pilot_split"]=="val"}
    assert len(selected)==4
    rows=defaultdict(list)
    for r in splitrows:
        if r["sequence"] in selected:
            assert r["split"]=="val"
            rows[r["sequence"]].append(r)
    assert set(rows)==set(selected)
    errors={"seq":[],"geom":[]}
    bytarget=[]
    count_gdiff=0
    with torch.inference_mode():
        for seqid in sorted(selected):
            emb=torch.load(root/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(seqid+".pt"),
                map_location="cpu",weights_only=True)["embeddings"]
            graphpath=cohort/"structure_graphs"/seqid/"graph.pt"
            assert h(graphpath)==selected[seqid]["graph_sha256"]
            graph=torch.load(graphpath,map_location="cpu",weights_only=True)
            toks=model.sequence(emb.float().cuda())
            enh=geometry.enhance(toks,graph)
            rawerr=[];geoerr=[]
            for start in range(0,len(rows[seqid]),8):
                batch=rows[seqid][start:start+8]
                smile=[compounds[r["compound"]] for r in batch]
                bits=torch.tensor(np.stack([fps[r["compound"]] for r in batch]),device="cuda",dtype=torch.float32)
                c=model.compound(smile,bits)
                valid=torch.ones((len(batch),len(toks)),device="cuda",dtype=torch.bool)
                ps=model.predict_tokens(c,toks.unsqueeze(0).expand(len(batch),-1,-1),valid).cpu().numpy()*ls+lm
                pg=model.predict_tokens(c,enh.unsqueeze(0).expand(len(batch),-1,-1),valid).cpu().numpy()*ls+lm
                count_gdiff+=int((np.abs(ps-pg)>1e-7).sum())
                actual=np.array([r["y"] for r in batch],dtype=np.float32)
                a=(ps-actual)**2;b=(pg-actual)**2
                errors["seq"].extend(a.astype(float).tolist())
                errors["geom"].extend(b.astype(float).tolist())
                rawerr.extend(a.astype(float).tolist());geoerr.extend(b.astype(float).tolist())
            sq=math.sqrt(float(np.mean(rawerr)))
            gq=math.sqrt(float(np.mean(geoerr)))
            bytarget.append({"sequence_sha256":seqid,"name":next((r["source_name"] for r in meta["records"] if r["sequence_sha256"]==seqid),None),
                             "query_pairs":len(rows[seqid]),"sequence_rmse":sq,
                             "geometry_rmse":gq,"geometry_minus_sequence_rmse":gq-sq})
    pooled={x:math.sqrt(float(np.mean(v))) for x,v in errors.items()}
    macro={k:float(np.mean([d[k+"_rmse"] for d in bytarget])) for k in ("sequence","geometry")}
    hist=json.loads((geo_dir/"history.json").read_text(encoding="utf-8"))
    chosen=hist[record["epoch"]-1]
    for metric,expected in [
        (pooled["seq"],chosen["validation_sequence_pooled_rmse"]),
        (pooled["geom"],chosen["validation_geometry_pooled_rmse"]),
        (macro["sequence"],chosen["validation_sequence_target_macro_rmse"]),
        (macro["geometry"],chosen["validation_geometry_target_macro_rmse"])]:
        assert abs(metric-expected)<1e-5,(metric,expected)
    assert count_gdiff>0
    result={
      "status":"PASS_ENGINEERING_CHECKPOINT_REEVALUATION",
      "geometry_best_epoch":record["epoch"],"val_proteins":len(selected),
      "val_pairs":sum(len(v) for v in rows.values()),
      "checkpoint_sha256":h(geopth),
      "recomputed_sequence_target_macro_rmse":macro["sequence"],
      "recomputed_geometry_target_macro_rmse":macro["geometry"],
      "structure_improved_target_count":sum(d["geometry_minus_sequence_rmse"]<0 for d in bytarget),
      "structure_harmed_target_count":sum(d["geometry_minus_sequence_rmse"]>0 for d in bytarget),
      "prediction_rows_with_change_from_geometry":count_gdiff,
      "max_metric_discrepancy_tolerance":1e-5,
      "per_target":bytarget,
      "warning":"Engineering-only original DAVIS subset; does not establish novelty, independent holdout or reliable biological benefit."}
    (geo_dir/"checkpoint_parity.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="per_target"},indent=2),flush=True)
    for d in bytarget:print(json.dumps(d),flush=True)

if __name__=="__main__":
    with gpu_lease(root):run()
