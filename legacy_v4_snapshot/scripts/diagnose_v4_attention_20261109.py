"""Read-only analysis of molecular discrimination and protein attention variability.

No labels/predictions saved per row. Captures target-level aggregate metadata on
the historical DAVIS pilot only; results are not an independent benchmark.
"""
import sys,json,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from scripts.train_v4_geometry_trainonly_20261109 import setup, OUTPUT, BATCH
from dynacpi.geometry_branch import GeometryBranch
from gpu_lease import gpu_lease
DEST=OUTPUT/"ATTENTION_DISCRIMINATION_DIAGNOSIS_20261109.json"

def metrics(t):
    return {"mean":float(t.float().mean()),"std":float(t.float().std(unbiased=False)),
       "min":float(t.min()),"max":float(t.max())}

def inspect(base,rows,ids,mean,std):
    report={}
    for pid in ids:
        r=rows[pid]
        x=r["x"].detach()
        xcenter=x-x.mean(dim=0)
        mol_nrm=x/(x.norm(dim=-1,keepdim=True)+1e-8)
        pair_cos=mol_nrm @ mol_nrm.T
        n=len(x)
        offdiag=(pair_cos.sum()-pair_cos.diag().sum())/(n*(n-1))
        seq=r["tokens"]
        weights=[]
        for i in range(0,n,BATCH):
            c=x[i:i+BATCH]
            n_batch=len(c)
            t=seq[None,:,:].expand(n_batch,-1,-1)
            valid=torch.zeros((n_batch,len(seq)),device="cuda",dtype=torch.bool)
            _,w=base.readout(c.unsqueeze(1),t,t,
                             key_padding_mask=valid,need_weights=True,average_attn_weights=False)
            # [batch, 4, query_len=1, residues]
            weights.append(w[:,:,0,:].detach().float().cpu().numpy())
        weights=np.concatenate(weights,axis=0)
        if weights.shape!=(n,4,len(seq)):
            raise RuntimeError("Attention shape mismatch")
        ent=-(weights*np.log(np.clip(weights,1e-12,None))).sum(axis=-1)
        eff=np.exp(ent)
        attention_across_compounds_std=float(np.mean(np.std(weights,axis=0)))
        attention_across_residues_std=float(np.mean(np.std(weights,axis=-1)))
        base_predictions=np.asarray(r["base_predictions"])
        yy=np.asarray(r["y"])
        report[r["name"]]={
          "n_compounds":n,"n_residues":len(seq),
          "compound_embedding_feature_std":float(xcenter.float().std(unbiased=False).item()),
          "compound_embedding_cosine_similarity_offdiag_mean":float(offdiag.item()),
          "baseline_predicted_pkd_std":float(np.std(base_predictions)),
          "baseline_predicted_pkd_range":float(np.ptp(base_predictions)),
          "observed_affinity_pkd_std":float(np.std(yy)),
          "observed_affinity_pkd_range":float(np.ptp(yy)),
          "mean_attention_effective_residue_count":float(np.mean(eff)),
          "effective_residue_count_fraction":float(np.mean(eff/len(seq))),
          "mean_attention_weight_std_across_compounds":attention_across_compounds_std,
          "mean_attention_weight_std_across_residues":attention_across_residues_std,
          "ratio_compound_to_residue_attention_variance":float(
             attention_across_compounds_std/max(attention_across_residues_std,1e-12))
        }
        print(json.dumps({"target":r["name"],**report[r["name"]]}),flush=True)
    return report

def main():
    if DEST.exists():raise RuntimeError("Refusing to overwrite prior diagnostic.")
    base,rows,fit,inner,dev,mean,std,hashes=setup()
    base.eval()
    with torch.inference_mode():
        target=inspect(base,rows,dev,mean,std)
    DEST.write_text(json.dumps({"status":"DEV_ONLY_FEATURE_DIAGNOSTICS","sources":hashes,"proteins":target},indent=2),encoding="utf-8")
    print(json.dumps({"status":"DONE","out":str(DEST)}),flush=True)

if __name__=="__main__":
    with gpu_lease(ROOT):
        main()
