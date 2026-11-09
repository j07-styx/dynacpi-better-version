"""Aggregate three PREDECLARED same-4-target engineering seed trials.

Best epoch is selected on the SAME validation proteins; this is a biased
development number and NOT an unbiased estimate of unseen-family utility.
"""
from pathlib import Path
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1]
ORIG=ROOT/"research/pilot_geometry_9train_4val"
AUX=ROOT/"research/pilot_geometry_seed_sweep_dev"
AUX.mkdir(parents=True,exist_ok=True)
data=[]
for seed,folder in [(1105,ORIG),(1106,AUX/"seed_1106"),(1107,AUX/"seed_1107")]:
    history=json.loads((folder/"history.json").read_text(encoding="utf-8"))
    assert len(history)==2 and [r["epoch"] for r in history]==[1,2]
    val=history[0]["validation_sequence_target_macro_rmse"]
    assert abs(val-0.9256850780053539)<1e-6
    best=min(history,key=lambda r:r["validation_geometry_target_macro_rmse"])
    ckpt=folder/"best_checkpoint.pt"
    assert ckpt.exists()
    status=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    assert status["best_epoch"]==best["epoch"]
    data.append({"seed":seed,"best_epoch":best["epoch"],"sequence_rmse":val,
                 "geometry_rmse_best":best["validation_geometry_target_macro_rmse"],
                 "geometry_minus_sequence_rmse":best["validation_geometry_target_macro_rmse"]-val,
                 "all_epochs":[{"epoch":r["epoch"],"geom_rmse":r["validation_geometry_target_macro_rmse"],
                      "sequence_rmse":r["validation_sequence_target_macro_rmse"]} for r in history],
                 "checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest()})
differences=[r["geometry_minus_sequence_rmse"] for r in data]
report={"status":"COMPLETE_SMALL_DEVELOPMENT_SEED_VARIABILITY_ONLY",
 "n_seeds":3,"n_validation_proteins":4,"n_validation_pairs":216,
 "same_approved_science_cohort":False,
 "seeds":data,
 "mean_best_epoch_geometry_minus_sequence_rmse":statistics.mean(differences),
 "sample_sd_best_epoch_geometry_minus_sequence_rmse":statistics.stdev(differences),
 "n_better_than_sequence":sum(d<0 for d in differences),
 "n_worse_than_sequence":sum(d>0 for d in differences),
 "limitations":[
   "All three seeds and epoch selection used the same four DAVIS development targets; not a blinded independent test.",
   "Two supervised epochs per seed are too few to establish stable converged performance.",
   "Comparing best epochs on the same validation cohort introduces optimism.",
   "Sample of three seeds is insufficient for robust uncertainty on future protein families.",
   "DAVIS reporting-floor labels and physical experimental constructs were not source approved."
 ]}
(AUX/"seed_variability.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
lines=["# DynaCPI v4 — GVP seed sensitivity (development only)","","**Not an independent held-out test.** Same 9-train/4-validation structure-mapped DAVIS pilot targets, two epochs, fixed parameters.","",
       "| Seed | Best epoch | Sequence RMSE | Geometry RMSE | Geometry − sequence |",
       "| ---: | ---: | ---: | ---: | ---: |"]
for r in data:
    lines.append(f"| {r['seed']} | {r['best_epoch']} | {r['sequence_rmse']:.6f} | {r['geometry_rmse_best']:.6f} | {r['geometry_minus_sequence_rmse']:+.6f} |")
lines+=["",f"Mean best-epoch difference across three seeds: {report['mean_best_epoch_geometry_minus_sequence_rmse']:+.6f} RMSE.",
       f"Sample SD: {report['sample_sd_best_epoch_geometry_minus_sequence_rmse']:.6f}.",
       "Two seeds are lower-error than the frozen sequence predictor, one higher-error on *these exact development targets*. This is initialization sensitivity, not evidence of generalizable gain.",
       "",
       "Next: check whether fixed A/B/C/D differential adaptation becomes identifiable with these alternate geometry checkpoints, without changing support/query identities or regularizers. Leave any source-approved test untouched."]
(AUX/"SEED_VARIABILITY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ("limitations","seeds")},indent=2))
for r in data:print(json.dumps({k:r[k] for k in ("seed","best_epoch","sequence_rmse","geometry_rmse_best","geometry_minus_sequence_rmse")}))
