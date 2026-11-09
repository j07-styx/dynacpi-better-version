# DERIVED 2026-11-06: same DEV K-sweep with alternate geometry checkpoint seed.
# Does not alter original K5 or K0 outcomes for original checkpoint.
"""DynaCPI v4 frozen-checkpoint SUPPORT-K DEVELOPMENT BENCHMARK.

Run ONLY on the original legacy DAVIS engineering validation quartet. It
reuses targets used for geometry checkpoint selection. It is NOT a
scientifically independent / cold-family holdout result.

K = 0,1,2,5,10,20, 10 support draws per target, 10 immutable query SMILES.
Frozen ESM sequence / GVP checkpoint weights, fixed engineering settings.
No model training, hyperparameter tuning, candidate BindingDB affinity labels
or access to source-unapproved scientific cohort.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict
import csv
import os
import hashlib
import json
import math
import statistics
import sys
import time
import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.episode_data import PredictionEpisode, QueryCompound, SupportObservation
from dynacpi.frozen_pair_predictor import FrozenPairPredictor
from dynacpi.frozen_adapter import adapt_episode, AdapterSettings
from dynacpi.scientific_cohort_gate import current_status
from gpu_lease import gpu_lease

KS = (0,1,2,5,10,20)
DRAWS = 10
QUERIES = 10
SEED = 1105
STUDY = "ENGINEERING_DEVELOPMENT_NOT_SCIENTIFIC_TEST"
SOURCE = ROOT / "data/engineering_pilot"
CHECKPOINTS = ROOT / "research/pilot_geometry_seed_sweep_dev" / ("seed_" + os.environ.get("DYNACPI_GEOMETRY_DEV_SEED",""))
OUTPUT = ROOT / "research/support_k_sweep_seed_dev" / ("seed_" + os.environ.get("DYNACPI_GEOMETRY_DEV_SEED",""))

def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(2*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def idhash(s):return hashlib.sha256(s.encode("utf-8")).hexdigest()
def rmse(x,y):return math.sqrt(statistics.mean((float(a)-float(b))**2 for a,b in zip(x,y)))
def mae(x,y):return statistics.mean(abs(float(a)-float(b)) for a,b in zip(x,y))
def pct(values,percent):
    a=sorted(float(x) for x in values)
    if not a:return None
    v=percent*(len(a)-1);lo=int(v);hi=min(lo+1,len(a)-1)
    return a[lo]*(hi-v)+a[hi]*(v-lo)

def run():
    if current_status(ROOT).get("status") != "BLOCKED_UNAPPROVED_NOT_TRAINING_READY":
        raise RuntimeError("Unexpected scientific cohort state: do not reuse engineering benchmark script as science.")
    if not torch.cuda.is_available():raise RuntimeError("This engineering benchmark requires CUDA RTX.")
    torch.manual_seed(SEED)
    src_seq=ROOT/"research/pilot_sequence/checkpoint.pt"
    src_geom=CHECKPOINTS/"best_checkpoint.pt"
    seq_ckpt=torch.load(src_seq,map_location="cpu",weights_only=False)
    geo_ckpt=torch.load(src_geom,map_location="cpu",weights_only=False)
    assert seq_ckpt["kind"]=="sequence" and seq_ckpt["epoch"]==3
    assert os.environ.get("DYNACPI_GEOMETRY_DEV_SEED") in ("1106","1107") and geo_ckpt["sequence_checkpoint_sha256"]==digest_file(src_seq)
    assert seq_ckpt["row_sha256"]==digest_file(SOURCE/"rows.json")
    assert geo_ckpt["row_sha256"]==digest_file(SOURCE/"rows.json")
    assert geo_ckpt["manifest_sha256"]==digest_file(SOURCE/"limited_alpha_3d_graph_manifest.json")
    seq=SequenceFoundation().cuda().eval()
    seq.load_state_dict(seq_ckpt["state"],strict=True)
    geom=GeometryBranch().cuda().eval()
    geom.load_state_dict(geo_ckpt["state"],strict=True)
    predictor=FrozenPairPredictor(seq,geom,"cuda",
        label_mean=float(seq_ckpt["label_mean"]),label_std=float(seq_ckpt["label_std"]))
    comp=json.loads((ROOT/"data/staging/compounds.json").read_text(encoding="utf-8"))
    rows=json.loads((SOURCE/"rows.json").read_text(encoding="utf-8"))
    meta=json.loads((SOURCE/"limited_alpha_3d_graph_manifest.json").read_text(encoding="utf-8"))
    targets={r["sequence_sha256"]:r for r in meta["records"] if r["status"]=="GRAPH_MAPPED_ENGINEERING_ONLY" and r["pilot_split"]=="val"}
    assert len(targets)==4,"Expected four selected development validation proteins"
    by_target=defaultdict(list)
    for r in rows:
        if r["sequence"] in targets:
            assert r["split"]=="val","Cross-split target contamination"
            mol=Chem.MolFromSmiles(comp[r["compound"]])
            if mol is None:raise ValueError("Bad source SMILES")
            can=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
            by_target[r["sequence"]].append((idhash(can),can,float(r["y"])))
    assert set(by_target)==set(targets)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    out_json=OUTPUT/"results.json"
    out_csv=OUTPUT/"aggregate.csv"
    results=[]
    start=time.monotonic()

    for target in sorted(targets):
        obs=by_target[target]
        assert len(obs)==54 and len({cid for cid,_,_ in obs})==len(obs)
        esm=torch.load(ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(target+".pt"),
            map_location="cpu",weights_only=True)
        assert esm["sequence_hash"]==target
        graphp=SOURCE/"structure_graphs"/target/"graph.pt"
        assert digest_file(graphp)==targets[target]["graph_sha256"]
        graph=torch.load(graphp,map_location="cpu",weights_only=True)
        means=predictor.predict([smi for _,smi,_ in obs],esm["embeddings"],graph)
        query_set=set(cid for cid,_,_ in sorted(obs,key=lambda x:idhash("pilot_query|"+target+"|"+x[0]))[:QUERIES])
        queries=tuple(QueryCompound(cid,smi) for cid,smi,_ in sorted(obs) if cid in query_set)
        truth={cid:y for cid,_,y in obs if cid in query_set}
        support_pool=[(cid,smi,y) for cid,smi,y in obs if cid not in query_set]
        assert len(queries)==QUERIES and len(support_pool)==44 and all(not q.compound_id in {s[0] for s in support_pool} for q in queries)
        seq_minus_geom=[abs(means[cid].geometry_pKd-means[cid].sequence_pKd) for cid,_,_ in obs]
        for k in KS:
            eval_trials=[]
            kernel_W=[];variance_V=[];delta_abs=[];offset_abs=[]
            for draw in range(DRAWS):
                # Same query identities for all K, nested support prefixes per
                # draw, with compound-identity-only selection.
                pool=sorted(support_pool,key=lambda x:idhash("pilot_draw|"+str(draw)+"|"+target+"|"+x[0]))
                chosen=pool[:k]
                supp=tuple(SupportObservation(cid,smi,y,target) for cid,smi,y in chosen)
                episode=PredictionEpisode(task_id="ENGINEERING-"+target,
                    sequence_sha256=target,protein_component_id=target,assay_key=target,
                    support=supp,queries=queries)
                ids={x.compound_id for x in (*supp,*queries)}
                predictions=adapt_episode(episode,{cid:means[cid] for cid in ids},AdapterSettings())
                yy=[truth[r.compound_id] for r in predictions]
                m={name:rmse([getattr(r,name) for r in predictions],yy) for name in "ABCD"}
                m.update({name+"_mae":mae([getattr(r,name) for r in predictions],yy) for name in "ABCD"})
                eval_trials.append(m)
                kernel_W += [q.W for q in predictions]
                variance_V += [q.V for q in predictions]
                delta_abs += [abs(q.differential) for q in predictions]
                offset_abs += [abs(q.offset) for q in predictions]
            aggregate={name:statistics.mean(trial[name] for trial in eval_trials) for name in ("A","B","C","D","A_mae","B_mae","C_mae","D_mae")}
            item={
              "target":targets[target]["source_name"],"sequence_sha256":target,
              "K":k,"draws":DRAWS,"query_count":QUERIES,
              "averaged_episode_rmse":{v:aggregate[v] for v in "ABCD"},
              "averaged_episode_mae":{v:aggregate[v+"_mae"] for v in "ABCD"},
              "D_minus_B_rmse":aggregate["D"]-aggregate["B"],
              "D_minus_B_mae":aggregate["D_mae"]-aggregate["B_mae"],
              "mean_W":statistics.mean(kernel_W),
              "p50_W":pct(kernel_W,.5),"p90_W":pct(kernel_W,.9),
              "mean_V":statistics.mean(variance_V),
              "p50_V":pct(variance_V,.5),
              "mean_abs_differential":statistics.mean(delta_abs),
              "fraction_differential_below_1e_8":sum(x<1e-8 for x in delta_abs)/len(delta_abs),
              "mean_abs_offset":statistics.mean(offset_abs),
              "mean_s_g_disagreement":statistics.mean(seq_minus_geom),
              "max_s_g_disagreement":max(seq_minus_geom),
            }
            if k==0:
                assert all(item["averaged_episode_rmse"][name]==aggregate["A"] for name in "ABCD")
                assert item["mean_W"]==item["mean_V"]==0
            results.append(item)
            print(json.dumps({"target":item["target"],"K":k,
                  "B_rmse":aggregate["B"],"D_rmse":aggregate["D"],
                  "D_minus_B":item["D_minus_B_rmse"],
                  "mean_W":item["mean_W"],"mean_V":item["mean_V"]}),flush=True)
        del means,graph,esm

    macro=[]
    for k in KS:
        t=[r for r in results if r["K"]==k]
        assert len(t)==4
        row={"K":k,"n_targets":len(t),"n_draws":DRAWS,"n_unique_queries":QUERIES*len(t)}
        for name in "ABCD":
            row[name+"_macro_RMSE"]=statistics.mean(x["averaged_episode_rmse"][name] for x in t)
            row[name+"_macro_MAE"]=statistics.mean(x["averaged_episode_mae"][name] for x in t)
        row["D_minus_B_macro_RMSE"]=row["D_macro_RMSE"]-row["B_macro_RMSE"]
        for stat in ("mean_W","mean_V","mean_abs_differential","mean_abs_offset","fraction_differential_below_1e_8"):
            row[stat]=statistics.mean(x[stat] for x in t)
        macro.append(row)
    report={
       "status":"COMPLETE_ENGINEERING_DEVELOPMENT_NOT_BLIND_SCIENTIFIC_RESULT",
       "timestamp_utc":datetime.now(timezone.utc).isoformat(),
       "study_scope":"Exactly four structure-mapped, geometry-checkpoint-selected DAVIS validation targets; experimental Kd provenance and protein-family isolation NOT established.",
       "predefined_K":list(KS),
       "support_draws_per_target":DRAWS,"queries_per_target":QUERIES,
       "frozen_weights":{"sequence_epoch":3,"geometry_epoch":int(geo_ckpt["epoch"]),"geometry_development_seed":int(os.environ["DYNACPI_GEOMETRY_DEV_SEED"]),
         "sequence_checkpoint_sha256":digest_file(src_seq),
         "geometry_checkpoint_sha256":digest_file(src_geom),
         "pilot_rows_sha256":digest_file(SOURCE/"rows.json")},
       "adapter_settings":asdict(AdapterSettings()),
       "elapsed_seconds":round(time.monotonic()-start,3),
       "macro_by_K":macro,"per_target_by_K":results,
       "limitations":[
           "Geometry checkpoint was selected using these same four development proteins; results are not unbiased or independent.",
           "DAVIS reporting-floor affinity labels were handled as numeric without exact-source Kd/censoring review.",
           "All four targets are closely related kinase proteins; this is not a general cold-family benchmark.",
           "No architecture or regularizer was tuned with these results; effects measured at existing fixed engineering defaults.",
           "No scientific BindingDB cohort or frozen source-approved split has been created; source gate remains blocked.",
           "Repeated deterministic draws improve engineering diagnostics but do not increase independent protein target count."
       ]}
    out_json.write_text(json.dumps(report,indent=2),encoding="utf-8")
    with out_csv.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(macro[0]))
        writer.writeheader();writer.writerows(macro)
    with (OUTPUT/"SUMMARY.md").open("w",encoding="utf-8") as f:
        f.write("# DynaCPI v4 frozen adapter support-K DEV diagnostics\n\n")
        f.write("**NOT a source-approved, family-held-out scientific benchmark.** Four geometry-checkpoint-selected DAVIS development proteins only.\n\n")
        f.write("| K | A RMSE | B RMSE | C RMSE | D RMSE | D–B | Mean W | Mean V |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in macro:
            f.write("| "+str(row["K"])+" | "+" | ".join(f"{row[v]:.6g}" for v in ("A_macro_RMSE","B_macro_RMSE","C_macro_RMSE","D_macro_RMSE","D_minus_B_macro_RMSE","mean_W","mean_V"))+" |\n")
        f.write("\nCheckpoint SHA-256: "+report["frozen_weights"]["sequence_checkpoint_sha256"]+" (sequence); "+report["frozen_weights"]["geometry_checkpoint_sha256"]+" (geometry).\n")
        f.write("\nInterpretation must separate measured differential effect from effective support-kernel evidence. This experiment does not justify a new DynaCPI architecture or an improvement claim.\n")
    print(json.dumps({"status":report["status"],
                      "report":str(out_json),"duration_s":report["elapsed_seconds"],
                      "macro":macro},indent=2),flush=True)

if __name__=="__main__":
    with gpu_lease(ROOT):run()
