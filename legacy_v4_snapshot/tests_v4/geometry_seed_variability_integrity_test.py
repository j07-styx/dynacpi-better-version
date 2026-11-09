"""Ensure extra-geometry-seed DEV experiments use exact same pilot cohort.

No affinity model training or final scientific assessment.
"""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
r=ROOT/"research"
report=json.loads((r/"pilot_geometry_seed_sweep_dev/seed_variability.json").read_text(encoding="utf-8"))
compare=json.loads((r/"pilot_geometry_seed_sweep_dev/adapter_seed_comparison.json").read_text(encoding="utf-8"))
assert report["status"]=="COMPLETE_SMALL_DEVELOPMENT_SEED_VARIABILITY_ONLY"
assert compare["status"]=="COMPLETE_ENGINEERING_DEV_ONLY"
assert [x["seed"] for x in report["seeds"]]==[1105,1106,1107]
assert compare["seeds"]==[1105,1106,1107]
assert report["n_validation_proteins"]==4 and report["n_validation_pairs"]==216
assert compare["n_independent_targets"]==4 and compare["n_fixed_query_compounds"]==40
assert report["n_better_than_sequence"]==2 and report["n_worse_than_sequence"]==1
by_seed={x["seed"]:x for x in report["seeds"]}
for seed,folder in [
(1105,r/"pilot_geometry_9train_4val"),
(1106,r/"pilot_geometry_seed_sweep_dev/seed_1106"),
(1107,r/"pilot_geometry_seed_sweep_dev/seed_1107")]:
    history=json.loads((folder/"history.json").read_text(encoding="utf-8"))
    assert len(history)==2
    chosen=min(history,key=lambda x:x["validation_geometry_target_macro_rmse"])
    assert by_seed[seed]["best_epoch"]==chosen["epoch"]
    assert abs(by_seed[seed]["geometry_rmse_best"]-chosen["validation_geometry_target_macro_rmse"])<1e-9
    c=folder/"best_checkpoint.pt"
    assert by_seed[seed]["checkpoint_sha256"]==hashlib.sha256(c.read_bytes()).hexdigest()
entries=compare["table_selected_K"]
assert len(entries)==9
for seed in (1105,1106,1107):
    rows=[x for x in entries if x["seed"]==seed]
    assert [x["K"] for x in rows]==[0,5,20]
    assert rows[0]["A_RMSE"]==rows[0]["B_RMSE"]==rows[0]["C_RMSE"]==rows[0]["D_RMSE"]
    assert all(abs(x["D_minus_B_RMSE"])<1e-6 for x in rows)
for x in entries:
    assert all(math.isfinite(float(x[k])) for k in
               ("A_RMSE","B_RMSE","C_RMSE","D_RMSE","D_minus_B_RMSE","mean_W","mean_V"))
assert compare["D_minus_B_max_abs_selected"] < 1e-6
assert not (ROOT/"data/audit/bindingdb_source_approved_cohort.json").exists()
assert not (ROOT/"data/audit/bindingdb_frozen_protein_group_split.json").exists()
print(json.dumps({"status":"PASS_ENGINEERING_DEV_ONLY",
                 "n_initialization_seeds":3,
                 "n_existing_proteins":4,
                 "n_adapter_seed_K_records":len(entries),
                 "maximum_abs_D_minus_B_RMSE":compare["D_minus_B_max_abs_selected"],
                 "source_approved_cohort":False,
                 "warning":"No scientific model superiority established"},indent=2))
