"""Cache frozen pretrained ESM2 embeddings for the existing BALM INNER DEV inputs.

NO dev target labels are read, used or included in the feature cache.
This dataset split has been repeatedly inspected, so these features support
exploratory comparisons only. Never process prospective reserved test data.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch,esm

from dynacpi.balm_public_data_dev import ROOT,get_train_dev
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import cache_feature_map

CACHE=ROOT/"data/reference/balm_esm2_8m_innerdev_20261108"
FEATURES=CACHE/"esm2_t6_8m_innerdev_mean_layer6_w1022.npz"
MANIFEST=ROOT/"research/balm_esm2_t6_8m_innerdev_input_only_manifest_dev_20261108.json"
WINDOW=1022
BATCH=8

def main():
    if FEATURES.exists() or MANIFEST.exists():
        raise FileExistsError("Existing inner-dev cache or manifest; refuse overwrite")
    start=time.time()
    subset=get_train_dev()["inner_dev"]
    assert len(subset)==3891
    sequences=set(r["protein"] for r in subset)
    assert len(sequences)==209
    _,train_meta=cache_feature_map()
    dev=[(hashlib.sha256(s.encode()).hexdigest(),s) for s in sequences]
    assert len({p[0] for p in dev})==209
    dev.sort(key=lambda p:(min(len(p[1]),WINDOW),p[0]))
    model,alphabet=esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    converter=alphabet.get_batch_converter(truncation_seq_length=WINDOW)
    embeds={}
    with torch.inference_mode():
        for i in range(0,len(dev),BATCH):
            batch=dev[i:i+BATCH]
            _,_,tokens=converter(batch)
            hidden=model(tokens.to(device),repr_layers=[6],return_contacts=False)["representations"][6]
            for j,(digest,seq) in enumerate(batch):
                n=min(len(seq),WINDOW)
                vector=hidden[j,1:n+1].mean(0).float().cpu().numpy()
                assert vector.shape==(320,) and np.isfinite(vector).all()
                vector/=np.linalg.norm(vector)
                embeds[digest]=vector
            if (i//BATCH)%4==0 or i+BATCH>=len(dev):
                print(json.dumps({
                    "encoded_innerdev_input_proteins":len(embeds),
                    "total":len(dev),"elapsed_seconds":round(time.time()-start,1)
                }),flush=True)
    sorted_keys=sorted(embeds)
    matrix=np.asarray([embeds[k] for k in sorted_keys],dtype=np.float32)
    assert matrix.shape==(209,320)
    CACHE.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(FEATURES,sequence_sha256=np.asarray(sorted_keys),embedding=matrix)
    sha=hashlib.sha256(FEATURES.read_bytes()).hexdigest()
    manifest={
        "status":"FROZEN_ESM2_INNER_DEV_PROTEIN_INPUT_ONLY_FEATURES_NO_TEST",
        "source":"BALM pre-existing inner_dev input protein sequences only, all 209 unique proteins",
        "inner_dev_rows":len(subset),
        "unique_dev_proteins":len(dev),
        "training_proteins_encoded_here":0,
        "dev_target_labels_used_for_embedding":False,
        "reserved_prospective_test_sequences_or_labels_accessed":False,
        "model":"esm2_t6_8M_UR50D",
        "model_weight_sha256":train_meta["source_weight_sha256"],
        "training_reference_cache_sha256":train_meta["cache_sha256"],
        "layer":6,"max_residue_window":WINDOW,
        "pooling":"mean of 1..min(length,1022) non-special residue tokens then l2 normalization",
        "shape":list(matrix.shape),
        "max_full_sequence_length":max(map(len,sequences)),
        "truncated_protein_count":sum(len(s)>WINDOW for s in sequences),
        "local_cache":"data/reference/balm_esm2_8m_innerdev_20261108/esm2_t6_8m_innerdev_mean_layer6_w1022.npz",
        "cache_sha256":sha,
        "seconds":round(time.time()-start,1),
        "limitations":[
            "Frozen inner development repeatedly inspected by research process; no independent confirmation.",
            "Per-residue features for long proteins are truncated; not full-length embeddings.",
            "No reserved prospective TEST protein input or label was accessed.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
    print("FINISHED",json.dumps(manifest),flush=True)

if __name__=="__main__":main()
