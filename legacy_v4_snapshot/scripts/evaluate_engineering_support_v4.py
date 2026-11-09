"""Evaluate A/B/C/D adapter mechanics on real legacy DAVIS development labels.

This is NOT DynaCPI scientific evaluation:
  - just 4 development-selected kinases, not independent protein family groups
  - numeric legacy DAVIS floor includes unverified assay endpoints
  - geometry checkpoint was selected using these validation proteins
  - w0=0.5, lambda_b=5, lambda_theta=0.1 are fixed engineering settings
  - no test labels, model parameters or adapter settings are optimized here.
"""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys
import numpy as np
import torch
from rdkit import Chem

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/"scripts"))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.episode_data import PredictionEpisode,QueryCompound,SupportObservation
from dynacpi.frozen_pair_predictor import FrozenPairPredictor
from dynacpi.frozen_adapter import adapt_episode,AdapterSettings
from gpu_lease import gpu_lease

def idhash(s):return hashlib.sha256(s.encode()).hexdigest()
def rmse(pred,true):return math.sqrt(statistics.mean((a-b)**2 for a,b in zip(pred,true)))
def run():
    source=root/"data/engineering_pilot"
    figures=root/"research/pilot_geometry_9train_4val"
    base_ckpt=torch.load(root/"research/pilot_sequence/checkpoint.pt",map_location="cpu",weights_only=False)
    geo_ckpt=torch.load(figures/"best_checkpoint.pt",map_location="cpu",weights_only=False)
    assert base_ckpt["kind"]=="sequence"
    assert geo_ckpt["epoch"]==1
    base=SequenceFoundation().cuda().eval()
    base.load_state_dict(base_ckpt["state"],strict=True)
    geo=GeometryBranch().cuda().eval()
    geo.load_state_dict(geo_ckpt["state"],strict=True)
    predictor=FrozenPairPredictor(base,geo,"cuda",
                label_mean=float(base_ckpt["label_mean"]),
                label_std=float(base_ckpt["label_std"]))
    compounds=json.loads((root/"data/staging/compounds.json").read_text(encoding="utf-8"))
    data=json.loads((source/"rows.json").read_text(encoding="utf-8"))
    manifest=json.loads((source/"limited_alpha_3d_graph_manifest.json").read_text(encoding="utf-8"))
    selected={r["sequence_sha256"]:r for r in manifest["records"] if r["status"]=="GRAPH_MAPPED_ENGINEERING_ONLY" and r["pilot_split"]=="val"}
    assert len(selected)==4
    rows=defaultdict(list)
    for r in data:
        if r["sequence"] in selected:
            assert r["split"]=="val"
            mol=Chem.MolFromSmiles(compounds[r["compound"]])
            if mol is None:raise ValueError("Invalid legacy DAVIS compound")
            smi=Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)
            cid=idhash(smi)
            rows[r["sequence"]].append((cid,smi,float(r["y"])))
    summary=[]
    for p in sorted(selected):
        observations=rows[p]
        if len({c for c,_,y in observations})!=len(observations):
            raise ValueError("Duplicate canonical compound within legacy pilot task")
        seqobj=torch.load(root/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(p+".pt"),
                          map_location="cpu",weights_only=True)
        graph=torch.load(source/"structure_graphs"/p/"graph.pt",map_location="cpu",weights_only=True)
        means=predictor.predict([smi for _,smi,_ in observations],seqobj["embeddings"],graph)
        qids=[cid for cid,_,_ in sorted(observations,
                   key=lambda x:idhash("pilot_query|"+p+"|"+x[0]))[:10]]
        support_pool=[(c,s,y) for c,s,y in observations if c not in set(qids)]
        query_list=[(c,s,y) for c,s,y in observations if c in set(qids)]
        queries=tuple(QueryCompound(cid,smi) for cid,smi,y in sorted(query_list))
        truth={cid:y for cid,smi,y in query_list}
        assert len(queries)==10 and len(support_pool)>=5
        episodes=[]
        all_abs_delta=[]
        all_W=[]
        all_V=[]
        for draw in range(10):
            chosen=sorted(support_pool,key=lambda x:idhash("pilot_draw|"+str(draw)+"|"+p+"|"+x[0]))[:5]
            supports=tuple(SupportObservation(c,s,y,p) for c,s,y in chosen)
            ep=PredictionEpisode(task_id="ENG-"+p,sequence_sha256=p,
                protein_component_id=p,assay_key=p,
                support=supports,queries=queries)
            chosen_ids=set(q.compound_id for q in queries)|set(s.compound_id for s in supports)
            submeans={cid:means[cid] for cid in chosen_ids}
            predictions=adapt_episode(ep,submeans,AdapterSettings())
            yy=[truth[v.compound_id] for v in predictions]
            trial={k:rmse([getattr(x,k) for x in predictions],yy) for k in ("A","B","C","D")}
            episodes.append(trial)
            all_abs_delta.extend(abs(x.differential) for x in predictions)
            all_W.extend(x.W for x in predictions)
            all_V.extend(x.V for x in predictions)
        m={k:statistics.mean(v[k] for v in episodes) for k in ("A","B","C","D")}
        summary.append({
          "name":selected[p]["source_name"],"sequence_sha256":p,
          "query_count":10,"support_draws":10,"support_size":5,
          "mean_episode_rmse":m,
          "D_minus_B_development_rmse":m["D"]-m["B"],
          "mean_abs_differential_pKd":statistics.mean(all_abs_delta),
          "mean_W":statistics.mean(all_W),"mean_V":statistics.mean(all_V),
          "fraction_zero_differential":sum(d<1e-8 for d in all_abs_delta)/len(all_abs_delta)
        })
    macro={k:statistics.mean(t["mean_episode_rmse"][k] for t in summary)
           for k in ("A","B","C","D")}
    report={
      "status":"DEVELOPMENT_PILOT_ONLY_NOT_A_SCIENTIFIC_TEST",
      "study":"Legacy DAVIS four-kinase geometry-selected validation targets with fixed support labels (K5)",
      "cohort_targets":4,"total_unique_queries":40,"repeated_support_draws_per_target":10,
      "fixed_settings":{"w0":0.5,"lambda_b":5,"lambda_theta":0.1,"caps":[2,1],"Tanimoto_exponent":4},
      "target_macro_mean_episode_rmse":macro,
      "D_minus_B_development_macro_rmse":macro["D"]-macro["B"],
      "per_target":summary,
      "limitations":[
        "The geometry checkpoint was selected on these targets: reused development data; statistics are not an unbiased test.",
        "Numeric legacy DAVIS reporting-boundary values were treated as exact and were not source audited.",
        "Four closely related targets and just one trained seed cannot substantiate a cold-target claim.",
        "No hyperparameter selection, model update or final scientific benchmark is performed in this script."
      ]}
    destination=figures/"support_adapter_development_diagnostics.json"
    destination.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("per_target","limitations")},indent=2),flush=True)
    for x in summary:
        print(json.dumps({"target":x["name"],"A":x["mean_episode_rmse"]["A"],
            "B":x["mean_episode_rmse"]["B"],"C":x["mean_episode_rmse"]["C"],
            "D":x["mean_episode_rmse"]["D"],
            "D_minus_B":x["D_minus_B_development_rmse"],
            "mean_W":x["mean_W"],"mean_V":x["mean_V"],
            "mean_abs_differential":x["mean_abs_differential_pKd"]}),flush=True)

if __name__=="__main__":
    with gpu_lease(root):run()
