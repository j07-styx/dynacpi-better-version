"""Compare fixed-K adapter outputs for three DEV geometry seeds; no tuning."""
from pathlib import Path
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"research"
outdir=R/"pilot_geometry_seed_sweep_dev"
rows=[]
for seed,path in [
    (1105,R/"support_k_sweep_dev/results.json"),
    (1106,R/"support_k_sweep_seed_dev/seed_1106/results.json"),
    (1107,R/"support_k_sweep_seed_dev/seed_1107/results.json")]:
    d=json.loads(path.read_text(encoding="utf-8"))
    assert d["status"]=="COMPLETE_ENGINEERING_DEVELOPMENT_NOT_BLIND_SCIENTIFIC_RESULT"
    assert d["predefined_K"]==[0,1,2,5,10,20] and d["support_draws_per_target"]==10
    assert d["queries_per_target"]==10
    assert len(d["per_target_by_K"])==24
    assert len({x["sequence_sha256"] for x in d["per_target_by_K"]})==4
    ckpt=(R/"pilot_geometry_9train_4val/best_checkpoint.pt" if seed==1105
          else R/"pilot_geometry_seed_sweep_dev"/("seed_"+str(seed))/"best_checkpoint.pt")
    assert d["frozen_weights"]["geometry_checkpoint_sha256"]==hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert d["adapter_settings"]["lambda_theta"]==0.1
    for item in d["macro_by_K"]:
        if item["K"] in (0,5,20):
            rows.append({"seed":seed,"K":item["K"],
                         "A_RMSE":item["A_macro_RMSE"],"B_RMSE":item["B_macro_RMSE"],
                         "C_RMSE":item["C_macro_RMSE"],"D_RMSE":item["D_macro_RMSE"],
                         "D_minus_B_RMSE":item["D_minus_B_macro_RMSE"],
                         "mean_W":item["mean_W"],"mean_V":item["mean_V"],
                         "mean_abs_differential":item["mean_abs_differential"]})
assert len(rows)==9
assert max(abs(x["D_minus_B_RMSE"]) for x in rows)<1e-6
out={"status":"COMPLETE_ENGINEERING_DEV_ONLY", "seeds":[1105,1106,1107],
     "n_independent_targets":4,"n_fixed_query_compounds":40,"support_draws":10,
     "all_k_values":[0,1,2,5,10,20],
     "table_selected_K":rows,
     "D_minus_B_max_abs_selected":max(abs(x["D_minus_B_RMSE"]) for x in rows),
     "interpretation":"Across three trained geometry initialization seeds, geometric sequence-only means differ and geometry DEV macro RMSE is initialization-sensitive, but centered differential support adaptation D remains indistinguishable from calibration-only B under unchanged regularization. NOT an unbiased protein-family test.",
     "limitations":["Development proteins reused for geometry checkpoint selection.",
                    "Only 3 seeds, 2 GVP epochs each, 4 related targets.",
                    "DAVIS measurement provenance unapproved and reporting-floor labels numerical.",
                    "No hyperparameter tuning or architecture redesign performed."]}
(outdir/"adapter_seed_comparison.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
text=[
    "# DynaCPI v4 — Three geometry seeds, fixed support adaptation",
    "",
    "**Development only, not independent family generalization.** Geometry checkpoint selected on the same four DAVIS development proteins.",
    "",
    "| Geometry seed | K | B macro RMSE | D macro RMSE | D−B | Mean V |",
    "| ---: | ---: | ---: | ---: | ---: | ---: |",
]
for x in rows:
    text.append(f"| {x['seed']} | {x['K']} | {x['B_RMSE']:.8f} | {x['D_RMSE']:.8f} | {x['D_minus_B_RMSE']:.3g} | {x['mean_V']:.3g} |")
text += ["","The geometry branch is sensitive to training seed, but geometry-informed differential support correction is almost zero in all three predeclared seed trials. Results are fixed-regularizer engineering observations; no scientifically approved Kd test has been performed."]
(outdir/"ADAPTER_SEED_COMPARISON.md").write_text("\n".join(text)+"\n",encoding="utf-8")
print(json.dumps({"status":out["status"],"n_table_rows":len(rows),
                  "max_abs_D_minus_B":out["D_minus_B_max_abs_selected"]},indent=2))
