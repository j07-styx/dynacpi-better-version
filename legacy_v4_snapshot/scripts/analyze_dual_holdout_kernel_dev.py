"""Size-matched controls distinguish training depletion from chemical exclusion."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.tanimoto_regressor import TanimotoRegressor
OUT=ROOT/"research/dual_holdout_kernel_20261107"
def main():
    results=json.loads((OUT/"results.json").read_text())
    rows=[r for r in json.loads((ROOT/"data/engineering_pilot_primary_reconciled_20261107/rows.json").read_text()) if r["split"]=="train" and r["primary_label_kind"]=="numeric_Kd"]
    pack=np.load(ROOT/"data/staging/morgan_radius2_2048_chiral.npz");fp=dict(zip(pack["ids"],pack["bits"]))
    x=np.stack([fp[r["compound"]] for r in rows]);y=np.array([r["numeric_pKd"] for r in rows])
    rng=np.random.default_rng(71114);controls=[]
    for fold in results["folds"]:
        if fold["mode"]!="protein_and_compound" or fold["status"]!="SCORED":continue
        test=np.array([r["primary_protein"]==fold["target"] for r in rows]);available=np.flatnonzero(~test)
        scores=[]
        for draw in range(100):
            train=rng.choice(available,size=fold["train_rows"],replace=False)
            p=TanimotoRegressor().fit(x[train],y[train]).predict(x[test])
            scores.append(float(np.sqrt(np.mean((p-y[test])**2))))
        controls.append({"target":fold["target"],"n_train":fold["train_rows"],
          "dual_rmse":fold["kernel_rmse"],"size_matched_mean_rmse":float(np.mean(scores)),
          "dual_global_mean_rmse":fold["global_mean_rmse"],"draws":100})
    comparisons=[]
    for key in ("size_matched_mean_rmse","dual_global_mean_rmse"):
        delta=np.array([r["dual_rmse"]-r[key] for r in controls])
        boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
        comparisons.append({"comparison":"dual_kernel minus "+key,"delta":float(delta.mean()),
                            "target_bootstrap_95_interval":np.quantile(boot,[.025,.975]).tolist()})
    report={"controls":controls,"comparisons":comparisons,"seed":71114,
      "scope":"Post-experiment diagnostic; row-count matching does not match chemical or protein composition",
      "random_reduction_fits":len(controls)*100}
    with (OUT/"size_controls.json").open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
