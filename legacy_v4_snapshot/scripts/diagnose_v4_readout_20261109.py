"""Read-only counterfactual: separate mean-pooling and residue-attention effects.

For each already-trained real geometry checkpoint, pass base sequence and
geometry-enhanced protein tokens through the exact original frozen downstream
predictor with one pathway changed at a time. Not a new trained model.
"""
import sys,json,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from scripts.train_v4_geometry_trainonly_20261109 import setup, OUTPUT, SEEDS, BATCH
from dynacpi.geometry_branch import GeometryBranch
from gpu_lease import gpu_lease
DEST=OUTPUT/"READOUT_PATH_DIAGNOSIS_20261109.json"

def macro(y,p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

def evaluate(base,geom,row,mean,std):
    tokens=row["tokens"]
    geo=geom.enhance(tokens,row["graph"])
    before=[]; full=[]; mean_only=[]; attention_only=[]
    for i in range(0,len(row["x"]),BATCH):
        compound=row["x"][i:i+BATCH]
        length=len(compound)
        s=tokens.unsqueeze(0).expand(length,-1,-1)
        g=geo.unsqueeze(0).expand(length,-1,-1)
        valid=torch.ones((length,len(tokens)),dtype=torch.bool,device="cuda")
        keymask=~valid
        sa=base.readout(compound.unsqueeze(1),s,s,key_padding_mask=keymask,need_weights=False)[0][:,0]
        ga=base.readout(compound.unsqueeze(1),g,g,key_padding_mask=keymask,need_weights=False)[0][:,0]
        sm=tokens.mean(dim=0).unsqueeze(0).expand(length,-1)
        gm=geo.mean(dim=0).unsqueeze(0).expand(length,-1)
        def head(mm,aa):
            protein=base.norm(mm+aa)
            p=base.head(base.channel(compound,protein,protein)).squeeze(-1)
            return (p.cpu().numpy()*std+mean).tolist()
        before+=head(sm,sa)
        full+=head(gm,ga)
        mean_only+=head(gm,sa)
        attention_only+=head(sm,ga)
    arrays={k:np.asarray(v,dtype=np.float64) for k,v in
            (("baseline",before),("full",full),("mean_only",mean_only),("attention_only",attention_only))}
    if not np.allclose(arrays["baseline"],np.asarray(row["base_predictions"]),atol=5e-6,rtol=0):
        raise RuntimeError("Counterfactual path failed to reproduce original base predictor.")
    def record(p):
        delta=arrays[p]-arrays["baseline"]
        return {"rmse":macro(row["y"],arrays[p]),
                "mean_shift":float(np.mean(delta)),
                "mean_abs_shift":float(np.mean(np.abs(delta))),
                "std_shift_across_compounds":float(np.std(delta)),
                "mae_to_full_predictions":float(np.mean(np.abs(arrays[p]-arrays["full"])))}
    return {"protein":row["name"],
            "token_mean_residual_l2":float(torch.norm((geo-tokens).mean(dim=0)).item()),
            "token_residual_rms":float(torch.sqrt(torch.mean((geo-tokens)**2)).item()),
            "n":len(row["y"]),
            "baseline":record("baseline"),"full":record("full"),
            "mean_only":record("mean_only"),"attention_only":record("attention_only")}

def main():
    if DEST.exists():raise RuntimeError("Refusing to overwrite existing diagnosis.")
    base,rows,fit,inner,dev,mean,std,hashes=setup()
    arms=[]
    with torch.inference_mode():
        for seed in SEEDS:
            ck=torch.load(OUTPUT/("real_seed"+str(seed))/"best_checkpoint.pt",
                          map_location="cpu",weights_only=False)
            geo=GeometryBranch().cuda().eval()
            geo.load_state_dict(ck["state"],strict=True)
            proteins={}
            for p in dev:
                result=evaluate(base,geo,rows[p],mean,std)
                proteins[result["protein"]]=result
                print(json.dumps({"seed":seed,"protein":result["protein"],
                 "full_shift":result["full"]["mean_shift"],
                 "mean_only_shift":result["mean_only"]["mean_shift"],
                 "attention_only_shift":result["attention_only"]["mean_shift"],
                 "mean_only_mae_to_full":result["mean_only"]["mae_to_full_predictions"],
                 "attention_only_mae_to_full":result["attention_only"]["mae_to_full_predictions"]}),flush=True)
            arms.append({"seed":seed,"per_protein":proteins})
            del geo
            torch.cuda.empty_cache()
    DEST.write_text(json.dumps({"status":"DIAGNOSTIC_ONLY_REUSED_DAVIS_DEV_NOT_SCIENCE",
               "checkpoint_hashes":hashes,"arms":arms},indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"DONE","output":str(DEST)}),flush=True)

if __name__=="__main__":
    with gpu_lease(ROOT):
        main()
