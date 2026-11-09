"""Generic synthetic paired-regression evaluation test: NO BindingDB labels used."""
from pathlib import Path
import json
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from dynacpi.paired_evaluation import evaluate_paired

rows=[]
for component in range(15):
    for task in range(2):
        tid=f"synthetic_task_{component}_{task}"
        for seed in range(3):
            for draw in range(4):
                for query in range(10):
                    y=4.5+component*0.02+task*0.01+query*0.001
                    rows.append({"task_id":tid,"construct_id":f"synthetic_construct_{component}","component_id":f"synthetic_component_{component}",
                        "seed":seed,"draw":draw,"query_id":f"fake_query_{query}",
                        "y":y,"baseline":y+0.4+0.002*seed+0.003*draw,
                        "candidate":y+0.2+0.002*seed+0.003*draw})
result=evaluate_paired(rows,bootstrap_draws=1500)
assert result["n_protein_components"]==15
assert result["n_tasks"]==30 and result["n_constructs"]==15 and result["n_episodes"]==360
assert result["n_prediction_rows"]==3600
assert result["paired_rmse_reduction"]>0.15
assert result["paired_rmse_reduction_CI95"][0]>0
assert evaluate_paired(rows,bootstrap_draws=1500)==result
def expect_failure(bad,needle):
    try:
        evaluate_paired(bad,bootstrap_draws=100)
    except ValueError as exc:
        if needle not in str(exc):raise
    else:raise AssertionError("Expected evaluator rejection: "+needle)
expect_failure(rows+[rows[0]],"Duplicate paired query")
changed=[dict(r) for r in rows]
changed[10]["y"]+=3
expect_failure(changed,"query label changed")
changed=[dict(r) for r in rows]
changed[0]["component_id"]="other"
expect_failure(changed,"multiple protein components")
changed=[dict(r) for r in rows]
changed[0]["construct_id"]="fake_second_construct"
expect_failure(changed,"multiple protein constructs")
changed=[dict(r) for r in rows]
changed[0]["query_id"]="different_query"
expect_failure(changed,"matching fixed queries")
changed=[dict(r) for r in rows]
changed[0]["baseline"]=float("nan")
expect_failure(changed,"Non-finite")
missing_grid=[r for r in rows if not (r["task_id"]=="synthetic_task_0_0" and r["seed"]==0 and r["draw"]==0)]
expect_failure(missing_grid,"same fixed seed/support-draw grid")
report={"status":"PASS_SYNTHETIC_STATS_ONLY",
        "checks":["paired task and cluster macro RMSE","deterministic cluster bootstrap CI",
                  "reject duplicate rows","reject changing query labels between support draws",
                  "reject task spanning protein components",
                  "reject mismatched query sets","reject nonfinite predictions"],
        "synthetic_n_components":15,
        "synthetic_n_prediction_rows":3600,
        "demonstration_absolute_reduction":round(result["paired_rmse_reduction"],6),
        "warning":"Synthetic values are not drug-target affinity measurements; no research effect was observed."}
dest=root/"data/audit/paired_evaluation_synthetic_test.json"
dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
