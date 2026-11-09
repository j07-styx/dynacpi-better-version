"""Evaluate chemical-information retention and batch-invariance, no training.

Recomputes original molecules from the fixed DAVIS engineering pilot,
not BindingDB. Stores only aggregate chem representation statistics.
"""
import sys,json,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from scripts.train_v4_geometry_trainonly_20261109 import setup, SOURCE, STAGE, OUTPUT
from gpu_lease import gpu_lease
DEST=OUTPUT/"CHEMISTRY_PIPELINE_DIAGNOSIS_20261109.json"

def stats(a):return {"mean":float(np.mean(a)),"std":float(np.std(a)),"min":float(np.min(a)),"max":float(np.max(a))}
def main():
    if DEST.exists():raise RuntimeError("Refuse to overwrite previous diagnostic.")
    base,data,fit,inner,dev,mean,std,hashes=setup()
    rows=json.loads((SOURCE/"rows.json").read_text(encoding="utf-8"))
    smiles=json.loads((STAGE/"compounds.json").read_text(encoding="utf-8"))
    fz=np.load(STAGE/"morgan_radius2_2048_chiral.npz")
    fp=dict(zip(fz["ids"].astype(str),fz["bits"]))
    result={}
    with torch.inference_mode():
        for pid in dev:
            source_rows=[r for r in rows if r["sequence"]==pid and r["split"]=="val"]
            if len(source_rows)!=54:raise RuntimeError("Expected 54 pairs")
            token=data[pid]["tokens"]
            xs_single=[]
            preds_single=[]
            bits=(np.stack([fp[r["compound"]] for r in source_rows])>0).astype(np.int32)
            if len({str(r["compound"]) for r in source_rows})!=54:
                raise RuntimeError("Repeated compound IDs in DAVIS target.")
            for idx,r in enumerate(source_rows):
                mol=smiles[r["compound"]]
                f=torch.tensor(np.asarray(fp[r["compound"]]).copy(),device="cuda",dtype=torch.float32)[None,:]
                x=base.compound([mol],f)
                xs_single.append(x.squeeze(0).cpu().numpy())
                valid=torch.ones((1,len(token)),device="cuda",dtype=torch.bool)
                yp=base.predict_tokens(x,token[None,:,:],valid)
                preds_single.append(float(yp.item())*std+mean)
            single=np.stack(xs_single)
            cached=data[pid]["x"].cpu().numpy()
            b=np.asarray(data[pid]["base_predictions"])
            a=np.asarray(preds_single)
            intersection=bits @ bits.T
            sizes=bits.sum(axis=1)
            union=sizes[:,None]+sizes[None,:]-intersection
            tanimoto=intersection/np.maximum(union,1)
            pair=tanimoto[np.triu_indices(54,k=1)]
            normal=np.linalg.norm(single,axis=1,keepdims=True)
            u=single/np.maximum(normal,1e-9)
            c=u @ u.T
            embcos=c[np.triu_indices(54,k=1)]
            result[data[pid]["name"]]={
              "n_unique_compounds":54,"mean_pairwise_morgan_tanimoto":float(pair.mean()),
              "std_pairwise_morgan_tanimoto":float(pair.std()),
              "fraction_tanimoto_below_0_4":float(np.mean(pair<0.4)),
              "fraction_tanimoto_below_0_2":float(np.mean(pair<0.2)),
              "mean_single_embedding_cosine":float(embcos.mean()),
              "single_embedding_feature_std_across_compounds":float(single.std(axis=0).mean()),
              "batch_embedding_feature_std_across_compounds":float(cached.std(axis=0).mean()),
              "mean_abs_single_vs_batch_embedding_difference":float(np.mean(np.abs(single-cached))),
              "max_abs_single_vs_batch_embedding_difference":float(np.max(np.abs(single-cached))),
              "std_single_molecule_predicted_pkd":float(np.std(a)),
              "std_batch_molecule_predicted_pkd":float(np.std(b)),
              "mean_abs_single_vs_batch_prediction_pkd":float(np.mean(np.abs(a-b))),
              "max_abs_single_vs_batch_prediction_pkd":float(np.max(np.abs(a-b)))}
            print(json.dumps({"protein":data[pid]["name"],**result[data[pid]["name"]]}),flush=True)
    DEST.write_text(json.dumps({"status":"DAVIS_DEV_ONLY_CHEMICAL_INFO_DIAGNOSTIC",
       "sources":hashes,"proteins":result},indent=2),encoding="utf-8")
    print(json.dumps({"status":"DONE","path":str(DEST)}),flush=True)

if __name__=="__main__":
    with gpu_lease(ROOT):
        main()
