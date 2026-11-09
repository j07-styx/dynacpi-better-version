"""Fixed target-held-out chemical overlap challenge; numeric training pool only."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.tanimoto_regressor import TanimotoRegressor,tanimoto
OUT=ROOT/"research/dual_holdout_kernel_20261107"
def main():
    OUT.mkdir(exist_ok=False)
    src=ROOT/"data/engineering_pilot_primary_reconciled_20261107/rows.json"
    rows=[r for r in json.loads(src.read_text()) if r["split"]=="train" and r["primary_label_kind"]=="numeric_Kd"]
    assert len(rows)==106
    pack=np.load(ROOT/"data/staging/morgan_radius2_2048_chiral.npz");fp=dict(zip(pack["ids"],pack["bits"]))
    sm=json.loads((ROOT/"data/staging/compounds.json").read_text())
    x=np.stack([fp[r["compound"]] for r in rows]);y=np.array([r["numeric_pKd"] for r in rows])
    scaffold={c:MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(sm[c]),includeChirality=False) or "ACYCLIC" for c in {r["compound"] for r in rows}}
    protocol={"scope":"Training-only challenge, not independent benchmark","rows_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
      "model":"Fixed Tanimoto ridge alpha=1","regimes":["protein_only","protein_and_compound","protein_and_scaffold"],
      "selection":"Each of nine training targets held out in turn; remove every query compound or scaffold from other-target training rows",
      "minimum_training_rows":2,"same_query_rows_across_regimes":True,"no_hyperparameter_search":True,
      "limits":["Training size changes across regimes","Identifier separation is not homology separation","Numeric-only ascertainment","Repeated development use"]}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
    folds=[]
    for target in sorted({r["sequence"] for r in rows}):
        test=np.array([r["sequence"]==target for r in rows])
        compounds={r["compound"] for r in rows if r["sequence"]==target}
        scaffolds={scaffold[c] for c in compounds}
        for mode in protocol["regimes"]:
            train=np.array([r["sequence"]!=target and (mode=="protein_only" or
              (r["compound"] not in compounds if mode=="protein_and_compound" else scaffold[r["compound"]] not in scaffolds)) for r in rows])
            assert not any(r["sequence"]==target for r,keep in zip(rows,train) if keep)
            if mode!="protein_only":assert not ({r["compound"] for r,keep in zip(rows,train) if keep}&compounds)
            if mode=="protein_and_scaffold":assert not ({scaffold[r["compound"]] for r,keep in zip(rows,train) if keep}&scaffolds)
            rec={"target":rows[np.flatnonzero(test)[0]]["primary_protein"],"mode":mode,"train_rows":int(train.sum()),
                 "train_compounds":len({r["compound"] for r,keep in zip(rows,train) if keep}),"query_rows":int(test.sum())}
            if train.sum()<2:rec["status"]="INSUFFICIENT_TRAINING"
            else:
                m=TanimotoRegressor().fit(x[train],y[train]);pred=m.predict(x[test])
                rec.update(status="SCORED",kernel_rmse=float(np.sqrt(np.mean((pred-y[test])**2))),
                  global_mean_rmse=float(np.sqrt(np.mean((y[train].mean()-y[test])**2))),
                  query_mean_nearest_tanimoto=float(tanimoto(x[test],x[train]).max(1).mean()))
            folds.append(rec)
    common=set.intersection(*[{r["target"] for r in folds if r["mode"]==m and r["status"]=="SCORED"} for m in protocol["regimes"]])
    assert common
    summary={m:{"targets":len(common),"kernel_macro_rmse":float(np.mean([r["kernel_rmse"] for r in folds if r["mode"]==m and r["target"] in common])),
      "global_mean_macro_rmse":float(np.mean([r["global_mean_rmse"] for r in folds if r["mode"]==m and r["target"] in common]))} for m in protocol["regimes"]}
    result={"summary_common_targets":summary,"folds":folds,"all_disjointness_assertions_passed":True}
    with (OUT/"results.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
