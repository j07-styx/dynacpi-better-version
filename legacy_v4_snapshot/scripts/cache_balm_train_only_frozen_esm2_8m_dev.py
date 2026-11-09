"""Compute frozen ESM2 t6 8M protein representations for BALM TRAIN only.

No BALM development or reserved-test labels/inputs are used. Representation
pretraining is frozen. The output .npz is a local cache (not versioned).
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch,esm

from dynacpi.balm_public_data_dev import ROOT,get_train_only

CACHE=ROOT/"data/reference/balm_esm2_8m_train_20261108"
FEATURES=CACHE/"esm2_t6_8m_mean_layer6_w1022.npz"
MANIFEST=ROOT/"research/balm_esm2_t6_8m_train_only_feature_manifest_dev.json"
LAYER=6
WINDOW=1022
BATCH=8

def main():
    if MANIFEST.exists() or FEATURES.exists():
        raise FileExistsError("Cache or manifest exists; refuse accidental overwrite.")
    start=time.time()
    seqs={s for s in (r["protein"] for r in get_train_only())}
    assert len(seqs)==639
    pairs=[(hashlib.sha256(s.encode()).hexdigest(),s) for s in seqs]
    assert len(set(key for key,_ in pairs))==len(pairs)
    pairs.sort(key=lambda x:(min(len(x[1]),WINDOW),x[0]))
    model,alphabet=esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    converter=alphabet.get_batch_converter(truncation_seq_length=WINDOW)
    embedded={}
    with torch.inference_mode():
        for start_idx in range(0,len(pairs),BATCH):
            batch=pairs[start_idx:start_idx+BATCH]
            _,_,tokens=converter(batch)
            tokens=tokens.to(device,non_blocking=True)
            out=model(tokens,repr_layers=[LAYER],return_contacts=False)["representations"][LAYER]
            for j,(key,s) in enumerate(batch):
                number=min(len(s),WINDOW)
                pooled=out[j,1:1+number].mean(0).float().cpu().numpy()
                assert pooled.shape==(320,) and np.isfinite(pooled).all()
                embedded[key]=pooled
            if (start_idx//BATCH)%10==0 or start_idx+BATCH>=len(pairs):
                print(json.dumps({"embedded":len(embedded),"total":len(pairs),
                     "elapsed_s":round(time.time()-start,1),
                     "last_max_token":int(tokens.shape[1])}),flush=True)
    ordered=sorted(embedded)
    matrix=np.stack([embedded[key] for key in ordered]).astype(np.float32)
    assert matrix.shape==(639,320) and np.isfinite(matrix).all()
    CACHE.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(FEATURES,sequence_sha256=np.array(ordered),embedding=matrix)
    digest=hashlib.sha256(FEATURES.read_bytes()).hexdigest()
    pretrained=Path(torch.hub.get_dir())/"checkpoints/esm2_t6_8M_UR50D.pt"
    manifest={
      "status":"BALM_TRAIN_ONLY_PRETRAINED_ESM2_FROZEN_PROTEIN_FEATURE_CACHE",
      "upstream_pretrained":"esm2_t6_8M_UR50D",
      "model_parameter_count":sum(x.numel() for x in model.parameters()),
      "source_weight_sha256":hashlib.sha256(pretrained.read_bytes()).hexdigest(),
      "source_weight_local_file":pretrained.name,
      "max_nterm_aa_tokens":WINDOW,
      "representation_layer":LAYER,
      "pooling":"mean of non-special residue tokens 1..min(length,1022), no target labels",
      "protein_embeddings":len(ordered),
      "embedding_dimensions":matrix.shape[1],
      "proteins_truncated_to_window":sum(len(s)>WINDOW for s in seqs),
      "minimum_full_length":min(map(len,seqs)),
      "maximum_full_length":max(map(len,seqs)),
      "device":device,
      "cache_location_str":"data/reference/balm_esm2_8m_train_20261108/esm2_t6_8m_mean_layer6_w1022.npz",
      "cache_sha256":digest,
      "minutes":round((time.time()-start)/60,2),
      "caveats":[
        "Only 639 proteins from the frozen original TRAIN partition; not complete BALM.",
        "Frozen ESM2 8M encoder, no fine-tuning; 85 long proteins truncated to first 1022 residues.",
        "No frozen development or reserved test labels or protein sequences encoded.",
        "Protein similarity graph and assay provenance remain unverified."
      ]
    }
    MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("DONE",json.dumps(manifest),flush=True)

if __name__=="__main__":main()
