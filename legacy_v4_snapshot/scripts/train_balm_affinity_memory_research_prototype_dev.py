"""Fit a LOCAL, explicitly exploratory SMILES+protein memory-CPI prototype.

Fitting only frozen BALM TRAIN: not an approved scientific benchmark.
Neither frozen original inner dev nor prospective test outcomes are touched.
Checkpoint is local and intentionally excluded from Git.
"""
import hashlib,json,time
from pathlib import Path
import joblib
import numpy as np

from dynacpi.balm_public_data_dev import ROOT,get_train_only
from dynacpi.scientific_cohort_gate import current_status
from dynacpi.balm_memory_predictor_dev import (
    fit_experimental_model,predict_with_embedding,sequence_hash,AA_ALPHABET
)
from scripts.compare_balm_train_esm2_8m_vs_kmer_scaffold_cold_dev import cache_feature_map

CHECKPOINT=ROOT/"data/models/balm_affinity_memory_prototype_exploratory_20261108.joblib"
MANIFEST=ROOT/"research/balm_affinity_memory_research_inference_manifest_20261108.json"


def main():
    if CHECKPOINT.exists() or MANIFEST.exists():
        raise FileExistsError("Do not overwrite prior research checkpoint or manifest")
    started=time.time()
    rows=get_train_only()
    assert len(rows)==15538
    embeddings,source=cache_feature_map()
    model=fit_experimental_model(rows,embeddings)
    smoke=[]
    for sample in rows:
        seq=sample["protein"]
        if len(seq)>=20 and set(seq).issubset(AA_ALPHABET):
            s=model["known_embeddings"][sequence_hash(seq)]
            prediction=predict_with_embedding(model,sample["smiles"],seq,s)
            assert np.isfinite(prediction["predicted_source_affinity_label"])
            smoke.append({
                "smiles_sha256":hashlib.sha256(sample["smiles"].encode()).hexdigest(),
                "protein_sha256":sequence_hash(seq),
                "estimated_source_p_affinity":prediction["predicted_source_affinity_label"],
                "training_row_label_for_smoke_only":float(sample["label"]),
                "protein_length":len(seq),
                "note":"TRAIN-row functionality smoke, not unseen-set validation",
            })
            if len(smoke)==3:break
    assert len(smoke)==3
    CHECKPOINT.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,CHECKPOINT,compress=3)
    digest=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    gate=current_status(str(ROOT))
    manifest={
        "status":"RESEARCH_ONLY_PROTOTYPE_NOT_APPROVED_TRAINING_OR_DEPLOYMENT",
        "model_family":"Ridge chemical Morgan + protein k-mer/ESM2-8M; joint RDKit/AA HistGradientBoosting; protein/chemical similarity memory",
        "source_dataset":"BALM BindingDB_filtered frozen TRAIN only",
        "source_protein_embedding_cache_sha256":source["cache_sha256"],
        "train_original_pairs":model["train_rows_original"],
        "train_unique_protein_smiles_pairs":model["train_rows_unique"],
        "protein_count_with_frozen_esm":len(embeddings),
        "train_mean":model["train_mean"],
        "train_label_std":model["train_sd"],
        "ridge_share_of_base":.75,
        "booster_share_of_base":.25,
        "memory_mix":model["memory_mix"],
        "memory_shrinkage_to_train_mean":model["memory_lambda"],
        "memory_spec":vars(model["memory_spec"]),
        "fixed_beta_for_exploratory_inference":model["beta"],
        "smoke_predictions_train_only":smoke,
        "local_checkpoint_path":"data/models/balm_affinity_memory_prototype_exploratory_20261108.joblib",
        "checkpoint_size_bytes":CHECKPOINT.stat().st_size,
        "checkpoint_sha256":digest,
        "train_only_end_to_end_seconds":round(time.time()-started,2),
        "scientific_gate_status":gate,
        "warnings":[
            "Research-only local predictor. Scientific gate remains blocked; not fit for scientific or clinical use.",
            "SMOKE only uses training rows and is NOT a validation accuracy estimate.",
            "No original inner-development or reserved test labels/samples used.",
            "Underlying BALM labels have incomplete assay DOI/method and construct provenance.",
            "Unseen protein sequences require frozen ESM2 8M inference and may be truncated to 1022 residues.",
            "Checkpoint contains model and TRAIN labels/representations; intentionally not pushed to Git.",
        ]
    }
    MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
    print("DONE",json.dumps(manifest,indent=2),flush=True)

if __name__=="__main__":main()
