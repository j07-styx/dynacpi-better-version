"""Structural and numerical integrity check for development K=0..20 result.

NO new affinity model training, no scientific test set and no clinical claim.
"""
from pathlib import Path
import hashlib,json,math
root=Path(__file__).resolve().parents[1]
file=root/"research/support_k_sweep_dev/results.json"
if not file.is_file():raise RuntimeError("Run the original DAVIS K-sweep first")
d=json.loads(file.read_text(encoding="utf-8"))
assert d["status"]=="COMPLETE_ENGINEERING_DEVELOPMENT_NOT_BLIND_SCIENTIFIC_RESULT"
assert d["predefined_K"]==[0,1,2,5,10,20]
assert d["support_draws_per_target"]==10
assert d["queries_per_target"]==10
for name,p in [
("sequence_checkpoint_sha256",root/"research/pilot_sequence/checkpoint.pt"),
("geometry_checkpoint_sha256",root/"research/pilot_geometry_9train_4val/best_checkpoint.pt"),
("pilot_rows_sha256",root/"data/engineering_pilot/rows.json")]:
    assert d["frozen_weights"][name]==hashlib.sha256(p.read_bytes()).hexdigest(),name
macro=d["macro_by_K"]
assert [r["K"] for r in macro]==[0,1,2,5,10,20]
assert len(d["per_target_by_K"])==24
names=sorted({r["target"] for r in d["per_target_by_K"]})
assert names==["ADCK4","FRK","MEK5","OSR1"]
assert all([r["n_targets"]==4 and r["n_draws"]==10 and r["n_unique_queries"]==40 for r in macro])
assert all(all(math.isfinite(float(x[k+"_macro_RMSE"])) for k in "ABCD") for x in macro)
assert all(abs(macro[0][n+"_macro_RMSE"]-macro[0]["A_macro_RMSE"])<1e-12 for n in "BCD")
assert all(x["mean_W"]>0 for x in macro[1:])
assert all(x["mean_W"]>=y["mean_W"] for x,y in zip(macro[1:],macro[:-1]))
assert all(abs(x["D_minus_B_macro_RMSE"])<1e-7 for x in macro)
old=json.loads((root/"research/pilot_geometry_9train_4val/support_adapter_development_diagnostics.json").read_text(encoding="utf-8"))
for variant in "ABCD":
    assert abs(macro[3][variant+"_macro_RMSE"]-old["target_macro_mean_episode_rmse"][variant])<1e-9
assert d["adapter_settings"]["lambda_b"]==5.0
assert d["adapter_settings"]["lambda_theta"]==0.1
assert d["adapter_settings"]["kernel_exponent"]==4
report={
 "status":"PASS_ENGINEERING_DEVELOPMENT_ONLY",
 "K_values":[x["K"] for x in macro],
 "targets":names,
 "K5_matches_prior_report":True,
 "checkpoint_hashes_match":True,
 "D_minus_B_max_abs_development_RMSE":max(abs(x["D_minus_B_macro_RMSE"]) for x in macro),
 "warning":"No scientifically independent held-out Kd evaluation. No trained model gain is asserted."
}
(root/"research/support_k_sweep_dev/integrity_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
