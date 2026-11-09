"""Analyze completed development runs; bootstrap targets, not independent families."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_bounded_fusion_study_dev import metrics
study=ROOT/"research/bounded_fusion_study_20261107"
out=study/"analysis.json"
if out.exists(): raise FileExistsError(out)
baseline=json.loads((study/"baseline.json").read_text())
protocol=json.loads((study/"protocol.json").read_text())
rows=baseline["predictions"]
keys=[(r["sequence"],r["compound"],r["y"]) for r in rows]
targets=sorted({r["sequence"] for r in rows})
b=np.array([baseline["metrics"]["per_target"][p] for p in targets])
allruns={}
for variant in protocol["variants"]:
    runs=[]
    for seed in protocol["seeds"]:
        r=json.loads((study/(variant+"_seed_"+str(seed))/"result.json").read_text())
        assert [(x["sequence"],x["compound"],x["y"]) for x in r["predictions"]]==keys
        m=metrics(rows,[x["pred"] for x in r["predictions"]])
        assert abs(m["macro_rmse"]-r["best_macro_rmse"])<1e-12
        assert r["best_epoch"]==min(r["history"],key=lambda h:h["val"]["macro_rmse"])["epoch"]
        runs.append(r)
    allruns[variant]=runs
rng=np.random.default_rng(1200)
draws=rng.integers(0,len(targets),size=(10000,len(targets)))
result={"scope":"Development-target bootstrap; four targets; selected checkpoints. NOT a family-cluster confidence interval or independent significance test.",
        "variants":{}}
for v,runs in allruns.items():
    per=np.array([[next(h for h in r["history"] if h["epoch"]==r["best_epoch"])["val"]["per_target"][p]
                  for p in targets] for r in runs])
    delta=per.mean(0)-b
    boot=delta[draws].mean(1)
    result["variants"][v]={"mean_target_delta":dict(zip(targets,delta.tolist())),
        "descriptive_95_interval":np.quantile(boot,[.025,.975]).tolist(),
        "targets_harmed":int((delta>0).sum()),
        "divergence":[{"seed":r["seed"],"best_epoch":r["best_epoch"],
            "train_rmse_epoch0":r["history"][0]["train"]["macro_rmse"],
            "train_rmse_final":r["history"][-1]["train"]["macro_rmse"],
            "val_rmse_final":r["history"][-1]["val"]["macro_rmse"]} for r in runs]}
real=np.array([r["best_macro_rmse"] for r in allruns["real"]])
sequence=np.array([r["best_macro_rmse"] for r in allruns["sequence"]])
d=real-sequence
result["real_minus_sequence"]={"paired_seed_differences":d.tolist(),"mean":float(d.mean()),
    "sd":float(d.std(ddof=1)),"standardized_paired_effect":float(d.mean()/d.std(ddof=1))}
# Simple training-only baselines on exactly the geometry cohort.
pilot=json.loads((ROOT/"data/engineering_pilot/rows.json").read_text())
manifest=json.loads((ROOT/"data/engineering_pilot/limited_alpha_3d_graph_manifest.json").read_text())
trainids={r["sequence_sha256"] for r in manifest["records"] if r["pilot_split"]=="train"}
train=[r for r in pilot if r["sequence"] in trainids]
means={c:np.mean([r["y"] for r in train if r["compound"]==c]) for c in {r["compound"] for r in train}}
result["training_only_baselines"]={
    "global_mean":metrics(rows,[np.mean([r["y"] for r in train])]*len(rows)),
    "compound_mean":metrics(rows,[means[r["compound"]] for r in rows])}
# Archived matched-cohort outputs: no new paper-level reproduction claim.
arch=json.loads((ROOT/"research/pilot_perceiver/val_predictions.json").read_text())
lookup={(r["sequence"],r["compound"]):r for r in arch}
assert all(lookup[(r["sequence"],r["compound"])]["y"]==r["y"] for r in rows)
result["archived_perceiver_common_cohort"]=metrics(rows,[lookup[(r["sequence"],r["compound"])]["pred"] for r in rows])
result["archived_perceiver_caveat"]="Three-epoch local engineering implementation, not paper benchmark; preprocessing differs from extension; predictions not rerun here."
result["success"]=False
result["decision"]="Do not promote geometry: below 2% development threshold, sequence control better on all three matched seeds, no independent holdout."
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
