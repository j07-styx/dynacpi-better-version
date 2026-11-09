"""DynaCPI v4: geometry contribution decomposition, legacy DAVIS DEV only.

On the same frozen 4 selected protein targets, 10 fixed query compounds and
nested 10 support draws per target, evaluate 3 trained geometry seeds and
K=0,1,2,5,10,20. Direct contrasts:
  S0: sequence-only base; S_B: sequence-only calibrated
  G_A: geometry-mixture base; G_B: geometry-mixture calibrated
  G_D: geometry-mixture plus differential adapter
This distinguishes main effect G_B-S_B from differential effect G_D-G_B.
No training, parameter tuning, BindingDB labels, or independent science.
"""
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib, json, math, statistics, sys, time
import numpy as np
import torch
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.frozen_pair_predictor import FrozenPairPredictor
from dynacpi.frozen_adapter import FrozenMeans, adapt_episode, AdapterSettings
from dynacpi.episode_data import PredictionEpisode, QueryCompound, SupportObservation
from dynacpi.scientific_cohort_gate import current_status
from gpu_lease import gpu_lease

SEEDS=(1105,1106,1107)
KS=(0,1,2,5,10,20)
DRAWS=10
NQ=10
BASE=ROOT/"research/pilot_sequence/checkpoint.pt"
SOURCE=ROOT/"data/engineering_pilot"
OUTPUT=ROOT/"research/geometry_factorial_dev"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def idhash(s):return hashlib.sha256(str(s).encode()).hexdigest()
def rmse(x,y):
    return math.sqrt(statistics.mean((float(a)-float(b))**2 for a,b in zip(x,y)))
def mae(x,y):
    return statistics.mean(abs(float(a)-float(b)) for a,b in zip(x,y))
def stdpct(x):return float(sum(v>0 for v in x)/len(x)) if x else None
def setup():
    if current_status(ROOT)["status"]!="BLOCKED_UNAPPROVED_NOT_TRAINING_READY":
        raise PermissionError("This is fixed DAVIS DEV only. Scientific gate changed.")
    if not torch.cuda.is_available():raise RuntimeError("CUDA needed")
    seqck=torch.load(BASE,map_location="cpu",weights_only=False)
    assert seqck["kind"]=="sequence" and int(seqck["epoch"])==3
    assert seqck["row_sha256"]==sha(SOURCE/"rows.json")
    base=SequenceFoundation().cuda().eval()
    base.load_state_dict(seqck["state"],strict=True)
    for p in base.parameters():p.requires_grad_(False)
    rows=json.loads((SOURCE/"rows.json").read_text(encoding="utf-8"))
    smiles=json.loads((ROOT/"data/staging/compounds.json").read_text(encoding="utf-8"))
    meta=json.loads((SOURCE/"limited_alpha_3d_graph_manifest.json").read_text(encoding="utf-8"))
    targets={r["sequence_sha256"]:r for r in meta["records"]
             if r["status"]=="GRAPH_MAPPED_ENGINEERING_ONLY" and r["pilot_split"]=="val"}
    assert len(targets)==4
    grouped=defaultdict(list)
    for row in rows:
        tid=row["sequence"]
        if tid not in targets:continue
        assert row["split"]=="val"
        mol=Chem.MolFromSmiles(smiles[row["compound"]])
        if mol is None:raise ValueError("Invalid DAVIS pilot SMILES")
        canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
        grouped[tid].append((idhash(canonical),canonical,float(row["y"])))
    assert set(grouped)==set(targets)
    for tid,obs in grouped.items():
        assert len(obs)==54 and len({i for i,_,_ in obs})==54
    checkpointmap={
      1105:ROOT/"research/pilot_geometry_9train_4val/best_checkpoint.pt",
      1106:ROOT/"research/pilot_geometry_seed_sweep_dev/seed_1106/best_checkpoint.pt",
      1107:ROOT/"research/pilot_geometry_seed_sweep_dev/seed_1107/best_checkpoint.pt",
    }
    return seqck,base,targets,grouped,checkpointmap

def run():
    seqck,base,targets,grouped,checkpointmap=setup()
    allrows=[]
    begin=time.monotonic()
    devhash=sha(SOURCE/"rows.json")
    for seed in SEEDS:
        cpath=checkpointmap[seed]
        ckpt=torch.load(cpath,map_location="cpu",weights_only=False)
        assert ckpt["sequence_checkpoint_sha256"]==sha(BASE)
        assert ckpt["row_sha256"]==devhash
        assert ckpt["manifest_sha256"]==sha(SOURCE/"limited_alpha_3d_graph_manifest.json")
        assert ckpt["status"]=="ENGINEERING_LEGACY_DAVIS_NOT_SCIENCE_BENCHMARK" if seed!=1105 else True
        geometry=GeometryBranch().cuda().eval()
        geometry.load_state_dict(ckpt["state"],strict=True)
        predictor=FrozenPairPredictor(base,geometry,"cuda",
           label_mean=float(seqck["label_mean"]),label_std=float(seqck["label_std"]))
        for tid in sorted(targets):
            obs=grouped[tid]
            esm=torch.load(ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(tid+".pt"),map_location="cpu",weights_only=True)
            graphp=SOURCE/"structure_graphs"/tid/"graph.pt"
            assert sha(graphp)==targets[tid]["graph_sha256"]
            graph=torch.load(graphp,map_location="cpu",weights_only=True)
            means=predictor.predict([smi for _,smi,_ in obs],esm["embeddings"],graph)
            seqmeans={cid:FrozenMeans(sequence_pKd=m.sequence_pKd,geometry_pKd=m.sequence_pKd,geometry_available=False)
                       for cid,m in means.items()}
            query_set={cid for cid,_,_ in sorted(obs,key=lambda x:idhash("pilot_query|"+tid+"|"+x[0]))[:NQ]}
            queries=tuple(QueryCompound(cid,smi) for cid,smi,_ in sorted(obs) if cid in query_set)
            truth={cid:y for cid,_,y in obs if cid in query_set}
            supppool=[(cid,smi,y) for cid,smi,y in obs if cid not in query_set]
            assert len(queries)==NQ and len(supppool)==44
            for K in KS:
                for draw in range(DRAWS):
                    ordered=sorted(supppool,key=lambda x:idhash("pilot_draw|"+str(draw)+"|"+tid+"|"+x[0]))
                    supports=tuple(SupportObservation(cid,smi,y,tid) for cid,smi,y in ordered[:K])
                    episode=PredictionEpisode(task_id="DEV_FACTORIAL_"+tid,
                       sequence_sha256=tid,protein_component_id=tid,
                       assay_key=tid,support=supports,queries=queries)
                    selected={x.compound_id for x in (*supports,*queries)}
                    g=adapt_episode(episode,{i:means[i] for i in selected},AdapterSettings())
                    s=adapt_episode(episode,{i:seqmeans[i] for i in selected},AdapterSettings())
                    ys=[truth[x.compound_id] for x in g]
                    assert [x.compound_id for x in g]==[x.compound_id for x in s]
                    outputs={"S0":[x.A for x in s],"S_B":[x.B for x in s],
                             "G_A":[x.A for x in g],"G_B":[x.B for x in g],"G_D":[x.D for x in g]}
                    stat={label:rmse(v,ys) for label,v in outputs.items()}
                    stat.update({label+"_MAE":mae(v,ys) for label,v in outputs.items()})
                    allrows.append({"seed":seed,"target_id":tid,"target_name":targets[tid]["source_name"],
                         "K":K,"draw":draw,**stat,
                         "mean_W":statistics.mean(x.W for x in g),
                         "mean_V":statistics.mean(x.V for x in g),
                         "mean_abs_delta":statistics.mean(abs(x.differential) for x in g),
                         "mean_s_g_absolute_difference":statistics.mean(abs(means[c].geometry_pKd-means[c].sequence_pKd) for c in truth)})
            print(json.dumps({"stage":"target_done","seed":seed,"target":targets[tid]["source_name"],"K_values":list(KS)}),flush=True)
        del geometry,predictor
        torch.cuda.empty_cache()
    return allrows,checkpointmap,time.monotonic()-begin

def summarise(rows,checkpointmap,elapsed):
    # Average repeated draws WITHIN each target before target macro;
    # never treat the 10 correlated draws as independent targets.
    per_target=[]
    macros=[]
    metrics=("S0","S_B","G_A","G_B","G_D")
    for seed in SEEDS:
        for K in KS:
            sub=[x for x in rows if x["seed"]==seed and x["K"]==K]
            assert len(sub)==40
            groups=defaultdict(list)
            for x in sub:groups[x["target_id"]].append(x)
            assert len(groups)==4 and all(len(g)==10 for g in groups.values())
            tar=[]
            for tid in sorted(groups):
                g=groups[tid]
                item={"seed":seed,"K":K,"target_name":g[0]["target_name"],
                      "target_sha256":tid}
                for metric in metrics:
                    item[metric+"_RMSE"]=statistics.mean(x[metric] for x in g)
                    item[metric+"_MAE"]=statistics.mean(x[metric+"_MAE"] for x in g)
                item["G_B_minus_S_B_RMSE"]=item["G_B_RMSE"]-item["S_B_RMSE"]
                item["G_D_minus_G_B_RMSE"]=item["G_D_RMSE"]-item["G_B_RMSE"]
                item["G_D_minus_S_B_RMSE"]=item["G_D_RMSE"]-item["S_B_RMSE"]
                item["mean_W"]=statistics.mean(x["mean_W"] for x in g)
                item["mean_V"]=statistics.mean(x["mean_V"] for x in g)
                item["mean_s_g_absolute_difference"]=statistics.mean(x["mean_s_g_absolute_difference"] for x in g)
                tar.append(item)
                per_target.append(item)
            macro={"seed":seed,"K":K,"independent_target_count":4,"support_draws_per_target":10,"query_ids_per_target":10}
            for metric in metrics:
                macro[metric+"_macro_RMSE"]=statistics.mean(x[metric+"_RMSE"] for x in tar)
                macro[metric+"_macro_MAE"]=statistics.mean(x[metric+"_MAE"] for x in tar)
            for metric in ("G_B_minus_S_B_RMSE","G_D_minus_G_B_RMSE","G_D_minus_S_B_RMSE","mean_W","mean_V","mean_s_g_absolute_difference"):
                macro[metric+"_macro"]=statistics.mean(x[metric] for x in tar)
            macro["targets_geometry_calibrated_better"]=sum(x["G_B_minus_S_B_RMSE"]<0 for x in tar)
            macro["targets_full_better_than_sequence_calibrated"]=sum(x["G_D_minus_S_B_RMSE"]<0 for x in tar)
            macros.append(macro)
    # Known frozen output equivalence: G_A, G_B, G_D should match the previous
    # K sweep for the SAME seed, target and fixed support/query identity.
    baseline={
      1105:ROOT/"research/support_k_sweep_dev/results.json",
      1106:ROOT/"research/support_k_sweep_seed_dev/seed_1106/results.json",
      1107:ROOT/"research/support_k_sweep_seed_dev/seed_1107/results.json"}
    max_discrepancy=0.
    for seed,path in baseline.items():
        older=json.loads(path.read_text(encoding="utf-8"))
        assert older["status"]=="COMPLETE_ENGINEERING_DEVELOPMENT_NOT_BLIND_SCIENTIFIC_RESULT"
        assert older["predefined_K"]==list(KS)
        assert older["support_draws_per_target"]==10 and older["queries_per_target"]==10
        prev={x["K"]:x for x in older["macro_by_K"]}
        for row in (x for x in macros if x["seed"]==seed):
            for name,oldname in (("G_A","A"),("G_B","B"),("G_D","D")):
                diff=abs(row[name+"_macro_RMSE"]-prev[row["K"]][oldname+"_macro_RMSE"])
                max_discrepancy=max(max_discrepancy,diff)
                if diff>1e-6:
                    raise RuntimeError(f"Recomputed frozen benchmark parity failure: {seed}, K={row['K']}, {name}, {diff}")
    status="COMPLETED_THREE_SEED_GEOMETRY_FACTORIAL_DEV_NOT_SCIENTIFIC_VALIDATION"
    report={
       "status":status,"created_utc":datetime.now(timezone.utc).isoformat(),
       "purpose":"Separate geometry main effect (G_B vs S_B) from differential geometry adaptation (G_D vs G_B) using fixed legacy DAVIS DEV.",
       "variant_labels":{"S0":"sequence-only uncalibrated","S_B":"sequence-only + support calibration",
                         "G_A":"geometry/sequence mixture uncalibrated","G_B":"geometry/sequence mixture + calibration","G_D":"geometry/sequence mixture + calibration + differential support adaptation"},
       "seeds":list(SEEDS),"support_sizes":list(KS),
       "independent_dev_protein_count":4,"repeated_support_draws_per_protein":10,
       "fixed_query_compounds_per_protein":10,
       "total_target_draw_K_trials":len(rows),
       "calibration_and_geometry_adapter_settings":asdict(AdapterSettings()),
       "frozen_sequence_checkpoint_sha256":sha(BASE),
       "frozen_geometry_checkpoint_sha256_by_seed":{str(k):sha(v) for k,v in checkpointmap.items()},
       "development_rows_sha256":sha(SOURCE/"rows.json"),
       "previous_K_sweep_agreement_max_abs_RMSE":max_discrepancy,
       "duration_seconds":round(elapsed,2),
       "macro_by_seed_K":macros,
       "per_target_by_seed_K":per_target,
       "warnings":[
         "This was evaluated on the exact same four DAVIS development targets used to choose each geometry checkpoint; not independent generalization.",
         "Legacy DAVIS reporting-floor values were used numerically without experimental exact-Kd adjudication.",
         "Four closely related kinase proteins and repeated support draws are not a powered protein-family benchmark.",
         "Any positive result is exploratory and may reflect validation reuse / favorable geometry initialization.",
         "This analysis does not select, tune, update or recommend production hyperparameters.",
         "No source-approved BindingDB training or test result; scientific gate remains blocked."
       ]}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"factorial.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines=["# Geometry contribution vs calibration: three-seed DAVIS engineering DEV", "",
           "**NOT independent source-approved validation.** Checkpoint-selected development targets and repeated draws only.","",
           "S_B = sequence-only + calibration; G_B = sequence+geometry mixture + calibration; G_D = G_B + differential support correction.","",
           "| Seed | K | S_B RMSE | G_B RMSE | G_D RMSE | G_B - S_B | G_D - G_B |",
           "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in macros:
        lines.append("| "+" | ".join([
           str(item["seed"]),str(item["K"]),
           *(f"{item[x+'_macro_RMSE']:.6f}" for x in ("S_B","G_B","G_D")),
           f"{item['G_B_minus_S_B_RMSE_macro']:+.8f}",
           f"{item['G_D_minus_G_B_RMSE_macro']:+.4g}",
           ])+" |")
    lines += ["",
        "Negative G_B - S_B indicates the geometry mixture beats sequence-only calibration on that same checkpoint-selected DEV cohort, NOT in any independent study.",
        "G_D - G_B isolates the added differential support adapter after the geometry contribution is already in the base prediction.",
        "The original 3-seed benchmark outputs were reproduced exactly to within 1e-6 RMSE.",
        "",
        "Repeated draws are nested within four targets. Three initialization seeds are not independent proteins.",
        "No scientific cohort approval, final BindingDB test, or code/weight updates occurred."]
    (OUTPUT/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":status,"macro_rows":len(macros),
      "n_trial_rows":len(rows),"checkpoint_parity_max_difference":max_discrepancy,
      "selected_K5":[{k:r[k] for k in ("seed","S_B_macro_RMSE","G_B_macro_RMSE","G_D_macro_RMSE","G_B_minus_S_B_RMSE_macro","G_D_minus_G_B_RMSE_macro")} for r in macros if r["K"]==5],
      "duration_seconds":elapsed},indent=2),flush=True)

if __name__=="__main__":
    with gpu_lease(ROOT):
        data,cpaths,elapsed=run()
        summarise(data,cpaths,elapsed)
