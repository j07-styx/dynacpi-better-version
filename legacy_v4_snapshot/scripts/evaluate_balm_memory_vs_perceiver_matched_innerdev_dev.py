"""Score the already fitted frozen-TRAIN memory model on original INNER DEV.

Runs once with fixed weights, representation and memory specifications from
previous TRAIN-only CV. Matched to saved PerceiverCPI and earlier locked
classical ensemble predictions on exactly the same 3891 inner-dev rows.
This is repeatedly inspected DEV, not an independent prospective test.
"""
import hashlib,json,glob,time
from pathlib import Path

import joblib
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import hstack
from sklearn.preprocessing import normalize

from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from dynacpi.balm_memory_predictor_dev import predict_with_embedding,sequence_hash
from dynacpi.cpi_similarity_memory_dev import MEMORY_SPECS,neighbor_weighted_labels
from scripts.compare_balm_perceiver_vs_ensemble_dev import (
    group_stats,paired_bootstrap,row_hash,
)
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    fingerprint_rows,metric,
)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import data_features

CHECKPOINT=ROOT/"data/models/balm_affinity_memory_prototype_exploratory_20261108.joblib"
CHECKPOINT_REPORT=ROOT/"research/balm_affinity_memory_research_inference_manifest_20261108.json"
EMBED_CACHE=ROOT/"data/reference/balm_esm2_8m_innerdev_20261108/esm2_t6_8m_innerdev_mean_layer6_w1022.npz"
EMBED_META=ROOT/"research/balm_esm2_t6_8m_innerdev_input_only_manifest_dev_20261108.json"
LOCKED=ROOT/"research/balm_ensemble_dev_predictions_20261108.json"
PERCEIVER=ROOT/"research/balm_perceiver_multiseed_dev_20261108"
OUT=ROOT/"research/balm_memory_vs_perceiver_matched_innerdev_20261108.json"
MODELS=("train_mean","locked_explicit_interaction_ensemble",
        "memory_research_base_no_memory",
        "memory_chemical_only_k64",
        "memory_joint_esm8_k64",
        "memory_joint_esm16_k64")

def verify_provenance(dev,checkpoint):
    devhash=row_hash(dev)
    locked=json.loads(LOCKED.read_text(encoding="utf8"))
    assert len(dev)==3891 and locked["dev_row_sha256"]==devhash
    checkpoint_meta=json.loads(CHECKPOINT_REPORT.read_text(encoding="utf8"))
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==checkpoint_meta["checkpoint_sha256"]
    assert checkpoint["model_kind"]=="EXPLORATORY_BALM_TRAIN_ONLY_MEMORY_CPI"
    assert checkpoint["train_rows_unique"]==12315
    assert checkpoint["train_rows_original"]==15538
    assert checkpoint["memory_spec"].name=="joint_esm8_k64"
    meta=json.loads(EMBED_META.read_text(encoding="utf8"))
    assert meta["unique_dev_proteins"]==209
    assert not meta["dev_target_labels_used_for_embedding"]
    assert not meta["reserved_prospective_test_sequences_or_labels_accessed"]
    assert hashlib.sha256(EMBED_CACHE.read_bytes()).hexdigest()==meta["cache_sha256"]
    with np.load(EMBED_CACHE,allow_pickle=False) as cached:
        keys=[str(k) for k in cached["sequence_sha256"]]
        vecs=normalize(np.asarray(cached["embedding"],dtype=np.float32))
    assert vecs.shape==(209,320)
    features={key:vec for key,vec in zip(keys,vecs)}
    actual={sequence_hash(r["protein"]) for r in dev}
    assert actual==set(features)
    assert actual.isdisjoint(checkpoint["known_embeddings"])
    return devhash,locked,checkpoint_meta,meta,features

def predict_in_batches(dev,model,embs,batch_size=128):
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    mu=float(model["train_mean"]);sd=float(model["train_sd"])
    res={k:[] for k in MODELS if k!="locked_explicit_interaction_ensemble"}
    for i in range(0,len(dev),batch_size):
        block=dev[i:i+batch_size]
        chemicals=fingerprint_rows(block,gen,{})
        pt=model["protein_kmer_vectorizer"].transform([r["protein"] for r in block])
        vectors=np.asarray([embs[sequence_hash(r["protein"])] for r in block],dtype=np.float32)
        assert vectors.shape==(len(block),320)
        X=hstack([chemicals,pt*.25,vectors*.25],format="csr")
        ridge_pred=np.asarray(model["ridge"].predict(X),dtype=np.float64)*sd+mu
        chemical,aa=data_features(block,{},{})
        boosted_pred=model["boosting"].predict(np.column_stack([chemical,aa]))
        baseline=.75*ridge_pred+.25*boosted_pred
        ligands=(chemicals@model["train_fingerprints"].T).toarray().astype(np.float32)
        proteins=vectors@model["train_embeddings"].T
        np.clip(ligands,0,1,out=ligands)
        np.clip(proteins,-1,1,out=proteins)
        variants={}
        for name in ("chemical_only_k64","joint_esm8_k64","joint_esm16_k64"):
            spec=next(s for s in MEMORY_SPECS if s.name==name)
            neighbor=neighbor_weighted_labels(ligands,proteins,model["train_labels"],mu,spec)
            memory=mu+.5*(neighbor-mu)
            variants["memory_"+name]=.75*baseline+.25*memory
        res["train_mean"].extend(np.full(len(block),mu).tolist())
        res["memory_research_base_no_memory"].extend(baseline.tolist())
        for name in variants:res[name].extend(variants[name].tolist())
        if i==0:
            check=predict_with_embedding(model,block[0]["smiles"],block[0]["protein"],vectors[0])
            np.testing.assert_allclose(variants["memory_joint_esm8_k64"][0],
                 check["predicted_source_affinity_label"],rtol=1e-5,atol=1e-4)
        if i%(batch_size*8)==0 or i+batch_size>=len(dev):
            print(json.dumps({"rows_scored":i+len(block),"total":len(dev)}),flush=True)
    return {name:np.asarray(pred,dtype=np.float64) for name,pred in res.items()}

def compare_groups(y,dev,model_preds,perceiver_runs):
    original_groups=sorted({r["group"] for r in dev})
    index={g:i for i,g in enumerate(original_groups)}
    gidx=np.asarray([index[r["group"]] for r in dev],dtype=np.int32)
    ng=len(original_groups)
    sse={}
    for name,pred in model_preds.items():
        assert len(pred)==len(y) and np.isfinite(pred).all()
        sse[name]=group_stats(np.square(pred-y),gidx,ng)
    pairs={}
    names=("locked_explicit_interaction_ensemble",
           "memory_research_base_no_memory","memory_chemical_only_k64",
           "memory_joint_esm8_k64","memory_joint_esm16_k64")
    for base in names:
        if base!="locked_explicit_interaction_ensemble":
            es,en=sse[base]
            compnames=("locked_explicit_interaction_ensemble","memory_research_base_no_memory")
            if base.startswith("memory_joint_esm"):compnames+=("memory_chemical_only_k64",)
            for other in compnames:
                if base==other:continue
                os,on=sse[other]
                pairs[f"{base}_minus_{other}"]=paired_bootstrap(es,en,
                    np.expand_dims(os,0),np.expand_dims(on,0))
        for epochs,runs in perceiver_runs.items():
            per=[]
            for r in runs:
                yhat=np.asarray(r["dev_predictions_by_epoch"][-1],dtype=np.float64)
                per.append(group_stats(np.square(yhat-y),gidx,ng))
            per_sq=np.asarray([s for s,n in per])
            per_n=np.asarray([n for s,n in per])
            es,en=sse[base]
            pairs[f"{base}_minus_perceiver_{epochs}ep_final_seedmean"]=paired_bootstrap(
                es,en,per_sq,per_n)
    return pairs

def main():
    if OUT.exists():raise FileExistsError(OUT)
    started=time.time()
    dev=get_train_dev()["inner_dev"]
    model=joblib.load(CHECKPOINT)
    devhash,locked,checkpoint_meta,esm_meta,embs=verify_provenance(dev,model)
    y=np.asarray([r["label"] for r in dev],dtype=np.float64)
    groups=np.asarray([r["group"] for r in dev])
    pred=predict_in_batches(dev,model,embs)
    selected=locked["selected_by_train_cv_from_existing_report"]
    assert selected=="ridge_interaction_blend_0.25"
    pred["locked_explicit_interaction_ensemble"]=np.asarray(locked["predictions"][selected])
    assert set(pred)==set(MODELS)
    eval_metrics={name:metric(y,v,groups) for name,v in pred.items()}
    runs={}
    for epoch_count in (10,30):
        paths=sorted(PERCEIVER.glob(f"perceiver_seed*_ep{epoch_count}.json"))
        runs[epoch_count]=[json.loads(path.read_text(encoding="utf8")) for path in paths]
        assert len(runs[epoch_count])==({10:6,30:2}[epoch_count])
        assert len(set(r["seed"] for r in runs[epoch_count]))==len(runs[epoch_count])
        assert all(r["dev_row_sha256"]==devhash for r in runs[epoch_count])
        assert all(r["epochs"]==epoch_count for r in runs[epoch_count])
    paired=compare_groups(y,dev,pred,runs)
    output={
        "status":"EXPLORATORY_BALM_LOCKED_FULL_TRAIN_MEMORY_INNER_DEV_COMPARISON_NO_TEST",
        "dev_rows":len(dev),"protein_component_count":len(set(groups)),
        "dev_row_sha256":devhash,
        "locked_previous_reference_name":selected,
        "prototype_checkpoint_sha256":checkpoint_meta["checkpoint_sha256"],
        "pretrained_esm_cache_sha256":esm_meta["cache_sha256"],
        "prototype_source":"full original frozen TRAIN 15538, collapsed to 12315 unique protein/raw-SMILES pairs, plus frozen pretrained ESM2 8M",
        "dev_protein_cache_status":"input-only unsupervised pretrained features, never fit to dev labels",
        "no_reserved_test_data_access":True,
        "memory_controls":"same Ridge + descriptor booster for all branches; separate fixed chemical-only and ESM-protein-weighted neighbor labels",
        "model_metrics":eval_metrics,
        "perceiver_seeds_by_budget":{str(k):[r["seed"] for r in v] for k,v in runs.items()},
        "paired_group_bootstrap":paired,
        "predictions_by_model":{name:[round(float(v),5) for v in arr] for name,arr in pred.items()},
        "elapsed_seconds":round(time.time()-started,2),
        "warnings":[
            "Previously inspected development split; this is NOT an independently heldout outcome.",
            "Prototype full-train beta=1, baseline as previously fixed for TRAIN-only exploratory model.",
            "Different neural baselines have different parameter counts and GPU training compute budgets.",
            "Sources lack assay DOI/construct/method and discordant repeated target labels.",
            "Protein sequence groups derive from approximate earlier 3-mer graph, not verified homology.",
            "Novel protein ESM vectors use first 1022 residues; 30 dev proteins truncated.",
            "The locked explicit-interaction ensemble differs from the simple descriptor booster in the memory prototype.",
            "Multiple architecture experiments and candidate comparisons were examined; bootstrap intervals are descriptive.",
            "Do not use this report to authorize the blocked reserved prospective test.",
        ],
    }
    # Correct name for the embedding checksum.
    output["pretrained_esm_cache_sha256"]=output.pop("pretrained_es m_cache_sha256")
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf8")
    print("FINAL",json.dumps({
        "metrics":eval_metrics,
        "joint_vs_locked":paired["memory_joint_esm8_k64_minus_locked_explicit_interaction_ensemble"],
        "joint_vs_chemical":paired["memory_joint_esm8_k64_minus_memory_chemical_only_k64"],
        "joint_vs_perceiver_10":paired["memory_joint_esm8_k64_minus_perceiver_10ep_final_seedmean"],
        "joint_vs_perceiver_30":paired["memory_joint_esm8_k64_minus_perceiver_30ep_final_seedmean"],
        "seconds":output["elapsed_seconds"],
    }),flush=True)

if __name__=="__main__":
    main()

