"""Paired target-block analysis of completed neural fits, without model selection."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research/matched_neural_lopo_20261107"
def main():
    rows=json.loads((OUT/"fits.json").read_text())
    predictions=json.loads((OUT/"predictions_LOCAL.json").read_text())
    protocol=json.loads((OUT/"protocol.json").read_text())
    assert len(rows)==81 and len(predictions)==106*9
    targets=sorted({r["target"] for r in rows});modes=protocol["modes"]
    assert len(targets)==9
    kernel=json.loads((ROOT/"research/protein_chemical_kernel_20261107/results.json").read_text())
    assert json.loads((ROOT/"research/protein_chemical_kernel_20261107/protocol.json").read_text())["rows_sha256"]==protocol["rows_sha256"]
    kp={f["target"]:f["rmse"]["chemical"] for f in kernel["folds"]}
    bytarget={}
    for target in targets:
        bytarget[target]={m:float(np.mean([r["rmse"] for r in rows if r["target"]==target and r["mode"]==m])) for m in modes}
        bytarget[target]["chemical_kernel"]=kp[target]
    rng=np.random.default_rng(71113);comparisons=[]
    for a,b in (("real_protein","chemical_only_zero_protein"),("real_protein","deranged_protein"),
                ("real_protein","chemical_kernel"),("chemical_only_zero_protein","chemical_kernel")):
        d=np.array([bytarget[t][a]-bytarget[t][b] for t in targets])
        boot=d[rng.integers(0,9,size=(10000,9))].mean(1)
        comparisons.append({"comparison":a+" minus "+b,"delta_rmse":float(d.mean()),
         "target_bootstrap_95_interval":np.quantile(boot,[.025,.975]).tolist(),"targets_improved":int(sum(d<0))})
    result={"target_mean_rmse":bytarget,"comparisons":comparisons,
      "per_seed_macro":{str(s):{m:float(np.mean([r["rmse"] for r in rows if r["seed"]==s and r["mode"]==m])) for m in modes} for s in protocol["seeds"]},
      "mean_final_train_standardized_mse":{m:float(np.mean([r["final_standardized_mse"] for r in rows if r["mode"]==m])) for m in modes},
      "limits":["Bootstrap over target identifiers, not homology clusters","Seeds and repeated targets are not independent biological replicates","No model selected or deployed"]}
    with (OUT/"paired_analysis.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="target_mean_rmse"},indent=2))
if __name__=="__main__":main()
