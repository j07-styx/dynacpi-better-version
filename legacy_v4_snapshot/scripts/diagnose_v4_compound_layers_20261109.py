"""Identify which frozen molecule-encoding stage destroys chemical diversity.

No label-dependent fitting, no reserved/test data. Only 54 DAVIS pilot
compounds and one already frozen model. Stores aggregate representation stats.
"""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from scripts.train_v4_geometry_trainonly_20261109 import setup,SOURCE,STAGE,OUTPUT
from chemprop.features import mol2graph
from gpu_lease import gpu_lease
DEST=OUTPUT/"COMPOUND_ENCODER_LAYER_DIAGNOSIS_20261109.json"

def feature_info(t):
    t=t.detach().float()
    v=t.cpu().numpy()
    norm=t/(t.norm(dim=-1,keepdim=True)+1e-9)
    cos=norm @ norm.T
    n=len(t)
    average=(cos.sum()-cos.diagonal().sum())/(n*(n-1))
    return {"vector_width":int(v.shape[1]),
        "mean_vector_norm":float(np.linalg.norm(v,axis=-1).mean()),
        "mean_pairwise_cosine":float(average.item()),
        "mean_feature_std_across_compounds":float(v.std(axis=0).mean()),
        "centered_vector_rms":float(np.sqrt(np.mean((v-v.mean(axis=0))**2))),
        "absolute_vector_rms":float(np.sqrt(np.mean(v*v)))}

def main():
    if DEST.exists():raise RuntimeError("Refuse to overwrite earlier diagnostic")
    base,data,fit,inner,dev,mean,std,hashes=setup()
    pid=next(p for p in dev if data[p]["name"]=="ADCK4")
    orig=json.loads((SOURCE/"rows.json").read_text(encoding="utf-8"))
    ss=json.loads((STAGE/"compounds.json").read_text(encoding="utf-8"))
    fpz=np.load(STAGE/"morgan_radius2_2048_chiral.npz")
    fp=dict(zip(fpz["ids"].astype(str),fpz["bits"]))
    source_rows=[r for r in orig if r["sequence"]==pid and r["split"]=="val"]
    smiles=[ss[r["compound"]] for r in source_rows]
    bits=torch.as_tensor(np.stack([fp[r["compound"]] for r in source_rows]),
                         dtype=torch.float32,device="cuda")
    with torch.inference_mode():
        raw=base.mpn([mol2graph(smiles)])
        low=raw.amin(dim=-1,keepdim=True)
        scaled=(raw-low)/(raw.amax(dim=-1,keepdim=True)-low).clamp_min(1e-6)
        fp_projected=base.fp(bits)
        channel_one=base.channel(fp_projected,scaled,scaled)
        mixed=scaled+channel_one
        final=base.channel(mixed,mixed,mixed)
        recomputed=base.compound(smiles,bits)
        if not torch.allclose(final,recomputed,atol=1e-6,rtol=0):
            raise RuntimeError("Layer decomposition does not reproduce trained encoder")
        parts={"Morgan_input":feature_info(bits),
               "Raw_MPN_output":feature_info(raw),
               "Per_molecule_minmax_normalized_MPN":feature_info(scaled),
               "Fingerprint_projection":feature_info(fp_projected),
               "First_channel_attention":feature_info(channel_one),
               "Added_MPN_and_channel1":feature_info(mixed),
               "Final_channel_attention_embedding":feature_info(final)}
    print(json.dumps(parts),flush=True)
    DEST.write_text(json.dumps({"status":"DAVIS_DEV_ONLY_ENCODER_LAYER_DIAGNOSTIC","sources":hashes,
                               "n_compounds":len(smiles),"layers":parts},indent=2),encoding="utf-8")
    print(json.dumps({"status":"DONE","path":str(DEST)}),flush=True)

if __name__=="__main__":
    with gpu_lease(ROOT):main()
