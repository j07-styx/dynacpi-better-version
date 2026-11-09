"""Paired component-level comparison on the same frozen inner-development rows."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main(seed,candidate="transfer",chemical="chemical"):
    base=ROOT/"research/transfer_cpi_dev_20261108";models=(candidate,chemical,"perceiver")
    records={};protocols={};preds={}
    for model in models:
        out=base/f"{model}_seed{seed}"
        assert (out/"COMPLETE.json").exists(),str(out)
        protocols[model]=json.loads((out/"protocol.json").read_text())
        hist=json.loads((out/"history.json").read_text())
        records[model]={"best":min(hist,key=lambda r:r["component_macro_rmse"]),"last":hist[-1]}
        preds[model]={kind:json.loads((out/(kind+"_predictions_LOCAL.json")).read_text()) for kind in ("best","last")}
    assert len({p["split_sha256"] for p in protocols.values()})==1
    for key in ("epochs","seed","batch_size","lr","weight_decay","train_pairs","dev_pairs","sampling","normalization"):
        assert len({p[key] for p in protocols.values()})==1,key
    ref=preds[candidate]["best"]
    anchors=lambda rows:[(r["component"],r["protein"],r["compound"],r["y"]) for r in rows]
    for permodel in preds.values():
        for rows in permodel.values():assert anchors(rows)==anchors(ref)
    rng=np.random.default_rng(2602);comparisons=[]
    for kind in ("best","last"):
        a=records[candidate][kind]["component_rmse"];groups=sorted(a)
        for baseline in (chemical,"perceiver","train_mean"):
            b=protocols[candidate]["baseline"]["component_rmse"] if baseline=="train_mean" else records[baseline][kind]["component_rmse"]
            assert sorted(b)==groups
            delta=np.array([a[g]-b[g] for g in groups])
            boot=delta[rng.integers(0,len(groups),size=(10000,len(groups)))].mean(1)
            comparisons.append({"checkpoint":kind,"baseline":baseline,"delta_component_macro_rmse":float(delta.mean()),
              "relative_reduction_percent":float(-100*delta.mean()/np.mean(list(b.values()))),
              "paired_component_bootstrap_95_interval":np.quantile(boot,[.025,.975]).tolist(),
              "components_improved":int(sum(delta<0)),"components":len(groups)})
    summary={m:{k:{key:r[key] for key in ("epoch","pooled_rmse","component_macro_rmse")} for k,r in records[m].items()} for m in models}
    result={"seed":seed,"summary":summary,"comparisons":comparisons,"train_mean":protocols[candidate]["baseline"],
       "scope":"Single-seed inner-development comparison; selected epoch is exploratory, no test evaluation",
       "matched_row_order_verified":True,"matched_optimizer_schedule_verified":True}
    path=base/f"comparison_{candidate}_seed{seed}.json"
    with path.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="train_mean"},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=2601);p.add_argument("--candidate",default="transfer");p.add_argument("--chemical",default="chemical");a=p.parse_args();main(a.seed,a.candidate,a.chemical)
