"""Export reusable protein-agnostic development baseline after fold parity check."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.tanimoto_regressor import TanimotoRegressor
def main():
    source=ROOT/"data/engineering_pilot_primary_reconciled_20261107/rows.json"
    rows=[r for r in json.loads(source.read_text()) if r["split"]=="train" and r["primary_label_kind"]=="numeric_Kd"]
    pack=np.load(ROOT/"data/staging/morgan_radius2_2048_chiral.npz");fp=dict(zip(pack["ids"],pack["bits"]))
    x=np.stack([fp[r["compound"]] for r in rows]);y=np.array([r["numeric_pKd"] for r in rows])
    previous=json.loads((ROOT/"research/protein_chemical_kernel_20261107/results.json").read_text())
    expected={f["target"]:f["rmse"]["chemical"] for f in previous["folds"]}
    differences=[]
    for target in sorted({r["primary_protein"] for r in rows}):
        test=np.array([r["primary_protein"]==target for r in rows])
        m=TanimotoRegressor().fit(x[~test],y[~test]);rmse=float(np.sqrt(np.mean((m.predict(x[test])-y[test])**2)))
        differences.append(abs(rmse-expected[target]))
    assert max(differences)<1e-10
    out=ROOT/"data/development_models";out.mkdir(exist_ok=True)
    path=out/"numeric_chemical_tanimoto_20261107.json"
    meta={"scope":"Development baseline only; ignores protein identity","training_rows":len(rows),
          "rows_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
          "features":"Morgan radius2 2048bit includeChirality=True","label":"numeric pKd",
          "source_approved":False}
    model=TanimotoRegressor().fit(x,y);model.save(path,meta)
    assert np.allclose(model.predict(x),TanimotoRegressor.load(path).predict(x),rtol=0,atol=1e-12)
    report={**meta,"model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "fold_parity_max_absolute_error":max(differences),"save_load_prediction_parity":True}
    report_path=ROOT/"research/matched_neural_lopo_20261107/chemical_baseline_export.json"
    with report_path.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
