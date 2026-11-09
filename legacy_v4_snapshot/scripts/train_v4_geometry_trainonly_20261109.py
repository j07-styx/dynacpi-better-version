"""DynaCPI v4 3D-branch training with train-only geometry selection controls.

DAVIS engineering pilot ONLY, not source-approved/independent CPI science.
Existing frozen sequence foundation (trained previously on pilot TRAIN proteins)
and 13 sequence-exact AlphaFold-mapped graphs are reused unchanged.
For each seed train REAL and MISALIGNED graph-to-residue controls separately.
Checkpoint selection uses 2 of the 9 historical TRAIN proteins, and the
four historically reused DAVIS validation proteins are scored only AFTER
checkpoint selection (still not an independent test).
"""
from __future__ import annotations
import hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.scientific_cohort_gate import current_status
from gpu_lease import gpu_lease

SOURCE=ROOT/"data/engineering_pilot"
STAGE=ROOT/"data/staging"
CACHE=ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"
SEQ=ROOT/"research/pilot_sequence/checkpoint.pt"
OUTPUT=ROOT/"research/dynacpi_v4_trainonly_geometry_20261109"
SEEDS=(1901,1902,1903)
MODES=("real","misaligned")
MAX_EPOCHS=6
BATCH=16
LR=3e-4
WEIGHT_DECAY=1e-4
GEOMETRY_TRUST=0.02
MIN_INNER_GAIN=1e-4
PERMUTATION_SALT="dynacpi_geometry_misalignment_fixed_v1"

def sha_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def inner_partition(training_ids):
    """Immutable train-only partition, not selected using affinity labels."""
    ordered=sorted(training_ids,key=lambda p:hashlib.sha256(("fixed_inner_dev_v1|"+p).encode()).hexdigest())
    if len(ordered)!=9 or len(set(ordered))!=9:
        raise ValueError("Expected exactly nine distinct engineering TRAIN proteins.")
    return tuple(sorted(ordered[2:])), tuple(sorted(ordered[:2]))

def permute_residue_map(graph,protein_id):
    """Hold graph features/edges fixed but break graph-residue alignment."""
    indices=graph["indices"]
    if not isinstance(indices,torch.Tensor) or indices.ndim!=1 or len(indices)<3:
        raise ValueError("Expected mapped contiguous 3D backbone graph.")
    seed=int(hashlib.sha256((PERMUTATION_SALT+"|"+protein_id).encode()).hexdigest()[:15],16)
    gen=torch.Generator(device="cpu").manual_seed(seed)
    order=torch.randperm(len(indices),generator=gen)
    if torch.equal(order,torch.arange(len(indices))):
        order=torch.roll(order,1)
    shuffled=dict(graph)
    shuffled["indices"]=indices[order].clone()
    if set(shuffled["indices"].tolist())!=set(indices.tolist()):
        raise ValueError("Misaligned mapping changed residue coverage.")
    if torch.equal(shuffled["indices"],indices):
        raise ValueError("Misalignment control did not alter alignment.")
    return shuffled

def rmse(pred,y):
    a=np.asarray(pred,dtype=np.float64)
    b=np.asarray(y,dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(a-b))))

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)

def setup():
    if current_status(ROOT)["status"]!="BLOCKED_UNAPPROVED_NOT_TRAINING_READY":
        raise RuntimeError("Science gate has changed; review before running this fixed engineering-only protocol.")
    if not torch.cuda.is_available():
        raise RuntimeError("Existing molecular MPN needs CUDA for this research training run.")
    torch.set_num_threads(min(torch.get_num_threads(),4))
    manifest_path=SOURCE/"limited_alpha_3d_graph_manifest.json"
    rows_path=SOURCE/"rows.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    records={r["sequence_sha256"]:r for r in manifest["records"]
             if r["status"]=="GRAPH_MAPPED_ENGINEERING_ONLY"}
    train_ids={p for p,r in records.items() if r["pilot_split"]=="train"}
    validation_ids={p for p,r in records.items() if r["pilot_split"]=="val"}
    if len(train_ids)!=9 or len(validation_ids)!=4 or train_ids&validation_ids:
        raise RuntimeError("3D mapped cohort was changed unexpectedly.")
    fit_ids,inner_ids=inner_partition(train_ids)
    seqck=torch.load(SEQ,map_location="cpu",weights_only=False)
    if seqck["kind"]!="sequence" or seqck["row_sha256"]!=sha_file(rows_path):
        raise RuntimeError("Frozen sequence checkpoint does not match the DAVIS rows.")
    base=SequenceFoundation().cuda().eval()
    base.load_state_dict(seqck["state"],strict=True)
    for p in base.parameters():p.requires_grad_(False)
    mean=float(seqck["label_mean"]);std=float(seqck["label_std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std<=0:
        raise RuntimeError("Invalid train-only original label statistics")
    rows=json.loads(rows_path.read_text(encoding="utf-8"))
    smiles=json.loads((STAGE/"compounds.json").read_text(encoding="utf-8"))
    sequences=json.loads((STAGE/"sequences.json").read_text(encoding="utf-8"))
    fpz=np.load(STAGE/"morgan_radius2_2048_chiral.npz")
    fps=dict(zip(fpz["ids"].astype(str),fpz["bits"]))
    packed={}
    for pid in sorted(records):
        relevant=[r for r in rows if r["sequence"]==pid]
        expected="train" if pid in train_ids else "val"
        if len(relevant)!=54 or any(r["split"]!=expected for r in relevant):
            raise RuntimeError("Original DAVIS engineering split/observation counts changed.")
        obj=torch.load(CACHE/(pid+".pt"),map_location="cpu",weights_only=True)
        if obj["sequence_hash"]!=pid or obj["embeddings"].shape!=(len(sequences[pid]),1280):
            raise RuntimeError("ESM protein/residue identity mismatch.")
        graphpath=SOURCE/"structure_graphs"/pid/"graph.pt"
        if sha_file(graphpath)!=records[pid]["graph_sha256"]:
            raise RuntimeError("3D source graph checksum mismatch.")
        graph=torch.load(graphpath,map_location="cpu",weights_only=True)
        with torch.inference_mode():
            tokens=base.sequence(obj["embeddings"].float().cuda()).detach()
        compounds=[]
        base_scores=[]
        for start in range(0,54,BATCH):
            rr=relevant[start:start+BATCH]
            sm=[smiles[r["compound"]] for r in rr]
            bits=torch.as_tensor(np.stack([fps[r["compound"]] for r in rr]),device="cuda",dtype=torch.float32)
            with torch.inference_mode():
                x=base.compound(sm,bits).detach()
                valid=torch.ones((len(rr),tokens.shape[0]),device="cuda",dtype=torch.bool)
                yp=base.predict_tokens(x,tokens[None,:,:].expand(len(rr),-1,-1),valid)
            compounds.append(x)
            base_scores.extend((yp.float().cpu().numpy()*std+mean).tolist())
        xx=torch.cat(compounds,dim=0)
        if xx.shape!=(54,256):raise RuntimeError("Unexpected compound vector cache shape.")
        yy=np.array([float(r["y"]) for r in relevant],dtype=np.float64)
        if not np.isfinite(yy).all():raise RuntimeError("Invalid historical pilot target.")
        packed[pid]={"name":records[pid]["source_name"],"tokens":tokens,
                     "graph":graph,"misaligned_graph":permute_residue_map(graph,pid),
                     "x":xx,"y":yy,"base_predictions":base_scores}
    hashes={"sequence_checkpoint_sha256":sha_file(SEQ),
            "graph_manifest_sha256":sha_file(manifest_path),
            "davis_rows_sha256":sha_file(rows_path)}
    return base,packed,fit_ids,inner_ids,tuple(sorted(validation_ids)),mean,std,hashes

def score_model(base,geom,data,ids,mode,mean,std):
    geom.eval()
    info={}
    with torch.inference_mode():
        for pid in ids:
            t=data[pid]
            tokens=t["tokens"]
            graph=t["graph"] if mode=="real" else t["misaligned_graph"]
            refined=geom.enhance(tokens,graph)
            outputs=[]
            for start in range(0,len(t["x"]),BATCH):
                x=t["x"][start:start+BATCH]
                valid=torch.ones((len(x),len(tokens)),device="cuda",dtype=torch.bool)
                yp=base.predict_tokens(x,refined[None,:,:].expand(len(x),-1,-1),valid)
                outputs.extend((yp.float().cpu().numpy()*std+mean).tolist())
            info[pid]={"protein":t["name"],"pairs":len(outputs),
                       "baseline_rmse":rmse(t["base_predictions"],t["y"]),
                       "geometry_rmse":rmse(outputs,t["y"]),
                       "mean_abs_prediction_change":float(np.mean(np.abs(np.asarray(outputs)-t["base_predictions"])))}
    macro=lambda key:float(np.mean([r[key] for r in info.values()]))
    return {"protein_count":len(ids),"macro_sequence_rmse":macro("baseline_rmse"),
            "macro_geometry_rmse":macro("geometry_rmse"),
            "macro_geometry_minus_sequence":macro("geometry_rmse")-macro("baseline_rmse"),
            "per_protein":info}

def run_arm(base,data,fit_ids,inner_ids,val_ids,mean,std,hashes,seed,mode):
    dirname=OUTPUT/(mode+"_seed"+str(seed))
    if dirname.exists() and (dirname/"COMPLETE.json").exists():
        raise RuntimeError("Refusing to overwrite completed arm: "+str(dirname))
    dirname.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    geom=GeometryBranch().cuda()
    optimizer=torch.optim.AdamW(geom.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    history=[]
    best_score=float("inf")
    best_epoch=-1
    filename=dirname/"best_checkpoint.pt"
    for epoch in range(0,MAX_EPOCHS+1):
        if epoch:
            geom.train()
            batch_plan=[]
            for pid in fit_ids:
                perm=np.random.default_rng(seed*100+epoch+int(pid[:4],16)).permutation(54).tolist()
                for pos in range(0,54,BATCH):
                    batch_plan.append((pid,perm[pos:pos+BATCH]))
            random.Random(seed+epoch).shuffle(batch_plan)
            trainloss=0.0
            for pid,idx in batch_plan:
                row=data[pid]
                x=row["x"][idx]
                tokens=row["tokens"]
                graph=row["graph"] if mode=="real" else row["misaligned_graph"]
                valid=torch.ones((len(idx),len(tokens)),device="cuda",dtype=torch.bool)
                optimizer.zero_grad(set_to_none=True)
                out=geom.enhance(tokens,graph)
                pred=base.predict_tokens(x,out[None,:,:].expand(len(idx),-1,-1),valid)
                y=torch.as_tensor((row["y"][idx]-mean)/std,device="cuda",dtype=torch.float32)
                prior=torch.as_tensor((np.asarray(row["base_predictions"])[idx]-mean)/std,device="cuda",dtype=torch.float32)
                loss=(pred-y).square().mean()+GEOMETRY_TRUST*(pred-prior).square().mean()
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite GVP training loss.")
                loss.backward()
                nn.utils.clip_grad_norm_(geom.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                trainloss+=float(loss.detach())*len(idx)
            trainloss/=len(fit_ids)*54
            if any(p.grad is not None for p in base.parameters()):
                raise RuntimeError("Frozen sequence foundation received gradients.")
        else:
            trainloss=None
        inner=score_model(base,geom,data,inner_ids,mode,mean,std)
        metric=inner["macro_geometry_rmse"]
        item={"epoch":epoch,"train_loss":trainloss,"inner_macro_seq_rmse":inner["macro_sequence_rmse"],
              "inner_macro_geom_rmse":metric,
              "inner_delta":inner["macro_geometry_minus_sequence"]}
        history.append(item)
        if metric < best_score-MIN_INNER_GAIN or best_epoch<0:
            best_score=metric
            best_epoch=epoch
            ckpt={"state":geom.state_dict(),"epoch":epoch,"seed":seed,"mode":mode,
                  "sequence_checkpoint_sha256":hashes["sequence_checkpoint_sha256"],
                  "graph_manifest_sha256":hashes["graph_manifest_sha256"],
                  "davis_rows_sha256":hashes["davis_rows_sha256"],
                  "label_mean":mean,"label_std":std,
                  "status":"ENGINEERING_DAVIS_TRAINONLY_GEOMETRY_SELECTION_NOT_BLIND_TEST"}
            tmp=filename.with_suffix(".pt.tmp")
            torch.save(ckpt,tmp)
            os.replace(tmp,filename)
        write_json(dirname/"history.json",history)
        print(json.dumps({"arm":mode,"seed":seed,**item,"best_epoch":best_epoch}),flush=True)
    check=torch.load(filename,map_location="cpu",weights_only=False)
    if check["seed"]!=seed or check["mode"]!=mode:raise RuntimeError("Checkpoint arm identity mismatch.")
    geom.load_state_dict(check["state"],strict=True)
    after=score_model(base,geom,data,val_ids,mode,mean,std)
    selected=score_model(base,geom,data,inner_ids,mode,mean,std)
    result={"status":"COMPLETE_DEVELOPMENT_ONLY_NOT_SCIENTIFIC_VALIDATION",
            "arm":mode,"seed":seed,"epochs_run":MAX_EPOCHS,"selected_epoch":best_epoch,
            "fit_protein_names":[data[p]["name"] for p in fit_ids],
            "inner_protein_names":[data[p]["name"] for p in inner_ids],
            "dev_protein_names":[data[p]["name"] for p in val_ids],
            "fit_pair_count":len(fit_ids)*54,"inner_pair_count":len(inner_ids)*54,
            "dev_pair_count":len(val_ids)*54,"selection":"minimum inner TRAIN-only protein macro RMSE incl epoch 0",
            "hyperparameters":{"lr":LR,"weight_decay":WEIGHT_DECAY,"trust_penalty":GEOMETRY_TRUST,
                               "max_epochs":MAX_EPOCHS,"batch":BATCH,"minimum_gain":MIN_INNER_GAIN},
            "input_hashes":hashes,"checkpoint_sha256":sha_file(filename),
            "inner_selection_result":selected,
            "historically_inspected_dev_diagnostic":after,
            "history":history,
            "limits":["Foundation pretrained on historical pilot TRAIN labels incl both geometry inner proteins",
                      "DAVIS development targets have been repeatedly viewed in earlier work",
                      "No verified physical assay constructs and no independent protein/source-family split",
                      "Geometry-shuffled control scrambles residue mapping, not raw 3D structure",
                      "Unreviewed BindingDB tasks, prospective test labels and publication source families are never loaded"]}
    write_json(dirname/"COMPLETE.json",result)
    print(json.dumps({"arm":mode,"seed":seed,"status":"COMPLETE","selected_epoch":best_epoch,
                      "dev_macro":after["macro_geometry_rmse"],"baseline":after["macro_sequence_rmse"]}),flush=True)
    del geom,optimizer
    torch.cuda.empty_cache()
    return result

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    base,data,fit,inner,val,mean,std,hashes=setup()
    write_json(OUTPUT/"PROTOCOL.json",{
        "status":"DAVIS_ENGINEERING_ONLY","seeds":SEEDS,"modes":MODES,
        "fit_proteins":[data[p]["name"] for p in fit],"inner_proteins":[data[p]["name"] for p in inner],
        "previously_seen_dev_proteins":[data[p]["name"] for p in val],
        "input_hashes":hashes,"epochs":MAX_EPOCHS,"lr":LR,
        "trust_penalty":GEOMETRY_TRUST,"checkpoint_policy":"best train-only inner macro RMSE, epoch zero eligible",
        "excluded":"no BindingDB task, no reserved prospective TEST label"})
    results=[]
    for seed in SEEDS:
        for mode in MODES:
            results.append(run_arm(base,data,fit,inner,val,mean,std,hashes,seed,mode))
    summaries=[{"seed":r["seed"],"arm":r["arm"],"epoch":r["selected_epoch"],
                "inner_rmse":r["inner_selection_result"]["macro_geometry_rmse"],
                "dev_rmse":r["historically_inspected_dev_diagnostic"]["macro_geometry_rmse"],
                "dev_delta":r["historically_inspected_dev_diagnostic"]["macro_geometry_minus_sequence"]} for r in results]
    write_json(OUTPUT/"SUMMARY.json",{"status":"COMPLETE_DEVELOPMENT_ONLY","arms":summaries,
          "baseline_dev":results[0]["historically_inspected_dev_diagnostic"]["macro_sequence_rmse"],
          "elapsed_seconds":time.monotonic()-begin,"data_hashes":hashes,
          "interpretation":"Training engineering result only; this development cohort has been repeatedly inspected and cannot prove generalization."})
    print(json.dumps({"status":"ALL_ARMS_COMPLETE","arm_count":len(summaries),"results":summaries}),flush=True)

if __name__=="__main__":
    with gpu_lease(ROOT):
        main()
