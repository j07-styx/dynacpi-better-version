"""Post-hoc scale diagnostic of FIXED support-K DEV benchmark, no tuning.

W/(W+lambda_b) and V/(V+lambda_theta) are descriptive ratios, not
estimated performance effects or exact optimizer coefficients.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"research/support_k_sweep_dev/results.json"
DEST=ROOT/"research/support_k_sweep_dev/identifiability.json"
DOC=ROOT/"research/support_k_sweep_dev/IDENTIFIABILITY.md"
raw=SRC.read_bytes()
data=json.loads(raw)
if data["status"]!="COMPLETE_ENGINEERING_DEVELOPMENT_NOT_BLIND_SCIENTIFIC_RESULT":
    raise RuntimeError("Refusing to analyze unverified or non-development benchmark")
settings=data["adapter_settings"]
lb=float(settings["lambda_b"]);lt=float(settings["lambda_theta"])
if lb<=0 or lt<=0:raise ValueError("Invalid regularizers")
table=[]
for item in data["macro_by_K"]:
    w=float(item["mean_W"]);v=float(item["mean_V"])
    if w<0 or v<0:raise ValueError("Invalid kernel statistic")
    table.append({
      "K":int(item["K"]),
      "mean_W":w,"mean_V":v,
      "calibration_weight_scale_W_over_W_plus_lambda_b":w/(w+lb),
      "disagreement_variance_scale_V_over_V_plus_lambda_theta":v/(v+lt),
      "mean_abs_differential":float(item["mean_abs_differential"]),
      "D_minus_B_macro_RMSE":float(item["D_minus_B_macro_RMSE"])
    })
assert [x["K"] for x in table]==[0,1,2,5,10,20]
assert max(abs(x["D_minus_B_macro_RMSE"]) for x in table)<1e-7,"Unexpected large effect: inspect benchmark"
report={
 "status":"DIAGNOSTIC_NO_HYPERPARAMETER_SELECTION",
 "time_utc":datetime.now(timezone.utc).isoformat(),
 "source_sha256":hashlib.sha256(raw).hexdigest(),
 "K_values":[x["K"] for x in table],
 "lambda_b":lb,"lambda_theta":lt,
 "table":table,
 "main_interpretation":"For every selected K, D and B are numerically indistinguishable at six-decimal RMSE. Support-weight mass W is small relative to the fixed calibration regularizer; centered sequence-geometry disagreement variance V is negligible relative to the fixed differential regularizer. The diagnostic does NOT imply changing regularizers would improve accuracy.",
 "critical_limits":[
   "All labels are from the same four geometry-checkpoint-selected DAVIS development proteins, not an unbiased or source-approved test.",
   "Ratios of averaged W and V are descriptive regularization-scale indicators, not exact averages of optimizer coefficients.",
   "Tuning regularizers or architecture on these four development targets then claiming gains on them would be biased.",
   "The broader preliminary BindingDB cohort has distinct chemistry; do not extrapolate this pilot result."
 ]}
DEST.write_text(json.dumps(report,indent=2),encoding="utf-8")
lines=["# DynaCPI v4 — Differential correction inactivity (DEV ONLY)","",
"**No scientific performance claim.** Derived from the four-target DAVIS development experiment using fixed checkpoints, no hyperparameter change.","",
"| K | Mean W | W/(W+lambda_b) | Mean V | V/(V+lambda_theta) | D-B RMSE |",
"| ---: | ---: | ---: | ---: | ---: | ---: |"]
for x in table:
    lines.append(f"| {x['K']} | {x['mean_W']:.5g} | {x['calibration_weight_scale_W_over_W_plus_lambda_b']:.5g} | {x['mean_V']:.5g} | {x['disagreement_variance_scale_V_over_V_plus_lambda_theta']:.5g} | {x['D_minus_B_macro_RMSE']:.3g} |")
lines+=["",
"With lambda_b=5 and lambda_theta=0.1, the evidence scales are far smaller than regularization across K=1..20.",
"The differential estimator also has a tiny centered support disagreement covariance in its numerator. Ratios are not direct values of theta or delta.",
"",
"**Next action:** Test sequence-versus-geometry complementarity, source-verified construct alignment and support coverage on independent approved development targets before tuning or redesigning. Preserve the negative result.",
"",
"Source result SHA256: "+report["source_sha256"]]
DOC.write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"status":report["status"],"report":str(DEST),"K20":table[-1],
                  "max_abs_D_minus_B_RMSE":max(abs(x["D_minus_B_macro_RMSE"]) for x in table)},indent=2))
