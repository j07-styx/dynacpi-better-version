"""Read-only integrity test for predeclared frozen 3-seed DEV factorial.

No source-unapproved BindingDB labels, no test-label tuning,
no new model training, and no unsupported scientific superiority claims.
"""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/"research/geometry_factorial_dev/factorial.json"
d=json.loads(P.read_text(encoding="utf-8"))
assert d["status"]=="COMPLETED_THREE_SEED_GEOMETRY_FACTORIAL_DEV_NOT_SCIENTIFIC_VALIDATION"
assert d["seeds"]==[1105,1106,1107]
assert d["support_sizes"]==[0,1,2,5,10,20]
assert d["total_target_draw_K_trials"]==720
assert d["independent_dev_protein_count"]==4
assert d["repeated_support_draws_per_protein"]==10
assert d["fixed_query_compounds_per_protein"]==10
assert d["previous_K_sweep_agreement_max_abs_RMSE"]<1e-6
assert len(d["macro_by_seed_K"])==18 and len(d["per_target_by_seed_K"])==72
assert "independent" in " ".join(d["warnings"]).lower()
seeds={s:set() for s in d["seeds"]}
for row in d["macro_by_seed_K"]:
    seeds[row["seed"]].add(row["K"])
    for name in ("S0","S_B","G_A","G_B","G_D"):
        assert math.isfinite(row[name+"_macro_RMSE"])
        assert math.isfinite(row[name+"_macro_MAE"])
        assert row[name+"_macro_RMSE"]>=0
    assert abs(row["G_B_minus_S_B_RMSE_macro"]-(row["G_B_macro_RMSE"]-row["S_B_macro_RMSE"]))<1e-10
    assert abs(row["G_D_minus_G_B_RMSE_macro"]-(row["G_D_macro_RMSE"]-row["G_B_macro_RMSE"]))<1e-10
    assert abs(row["G_D_minus_G_B_RMSE_macro"])<1e-6
    if row["K"]==0:
        assert abs(row["G_A_macro_RMSE"]-row["G_B_macro_RMSE"])<1e-10
        assert abs(row["S0_macro_RMSE"]-row["S_B_macro_RMSE"])<1e-10
for seed in d["seeds"]:
    assert seeds[seed]==set(d["support_sizes"])
    assert d["frozen_geometry_checkpoint_sha256_by_seed"][str(seed)]
k5={row["seed"]:row for row in d["macro_by_seed_K"] if row["K"]==5}
assert k5[1105]["G_B_minus_S_B_RMSE_macro"]>0
assert k5[1106]["G_B_minus_S_B_RMSE_macro"]<0
assert k5[1107]["G_B_minus_S_B_RMSE_macro"]<0
assert all(abs(row["G_D_minus_G_B_RMSE_macro"])<1e-6 for row in k5.values())
seq_hash=hashlib.sha256((ROOT/"research/pilot_sequence/checkpoint.pt").read_bytes()).hexdigest()
assert d["frozen_sequence_checkpoint_sha256"]==seq_hash
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"
result={
 "status":"PASS_FROZEN_THREE_SEED_DEV_GEOMETRY_FACTORIAL",
 "n_seeds":3,"n_dev_proteins":4,"n_target_draw_K_trials":720,
 "old_K_sweep_parity_max_error":d["previous_K_sweep_agreement_max_abs_RMSE"],
 "K5_geometry_main_effect_signs":{str(k):("improves" if v["G_B_minus_S_B_RMSE_macro"]<0 else "harms") for k,v in k5.items()},
 "geometry_differential_effect_negligible":True,"scientific_cohort_gate":"BLOCKED"}
(ROOT/"data/audit/geometry_factorial_dev_integrity.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
