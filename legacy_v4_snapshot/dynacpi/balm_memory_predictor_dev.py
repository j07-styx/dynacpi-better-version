"""Exploratory, runnable BALM compound-protein affinity-memory model.

All supervised fits use the frozen BALM TRAIN partition only. This model is
NOT scientifically approved or validated for deployment. ESM2 is pretrained
and frozen; unseen protein embeddings are inferred without label access.
"""
import hashlib
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from dynacpi.cpi_similarity_memory_dev import MEMORY_SPECS,neighbor_weighted_labels
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,fingerprint_rows
)
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import (
    HIST_PARAMS,data_features
)
from scripts.train_bindingdb_descriptor_protein_boosting_dev import (
    molecule_features,protein_features
)

ESM_WINDOW=1022
ESM_DIM=320
MEMORY_SPEC=next(x for x in MEMORY_SPECS if x.name=="joint_esm8_k64")
AA_ALPHABET=set("ACDEFGHIKLMNPQRSTVWYX")


def clean_protein(seq):
    seq="".join(seq.upper().split())
    if len(seq)<20 or any(c not in AA_ALPHABET for c in seq):
        raise ValueError("Protein must have >=20 valid uppercase amino-acid or X residues")
    return seq


def canonical_smiles(s):
    mol=Chem.MolFromSmiles(s)
    if mol is None or mol.GetNumAtoms()==0:
        raise ValueError("Invalid SMILES")
    return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)


def sequence_hash(seq):
    return hashlib.sha256(seq.encode()).hexdigest()


def encode_new_protein(seq,device=None):
    """Frozen ESM2 8M feature for an unknown sequence, first 1022 residues."""
    import esm
    import torch
    seq=clean_protein(seq)
    model,alphabet=esm.pretrained.esm2_t6_8M_UR50D()
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    converter=alphabet.get_batch_converter(truncation_seq_length=ESM_WINDOW)
    _,_,tokens=converter([(sequence_hash(seq),seq)])
    with torch.inference_mode():
        output=model(tokens.to(device),repr_layers=[6],return_contacts=False)
        n=min(len(seq),ESM_WINDOW)
        vec=output["representations"][6][0,1:1+n].mean(0).float().cpu().numpy()
    if vec.shape!=(ESM_DIM,) or not np.isfinite(vec).all():
        raise ValueError("Invalid ESM embedding")
    vec/=np.linalg.norm(vec)
    return vec


def fit_experimental_model(raw_rows,embedding_map):
    """Construct a fitted local research model using original TRAIN rows only."""
    training=collapse_training(raw_rows)
    y=np.asarray([r["label"] for r in training],dtype=np.float64)
    mu=float(y.mean());std=float(y.std())
    assert len(raw_rows)==15538 and len(training)==12315 and std>0
    unique_proteins={r["protein"] for r in training}
    all_hashes={sequence_hash(s) for s in unique_proteins}
    assert set(embedding_map)==all_hashes and len(unique_proteins)==639
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    chem_cache={}
    drug_matrix=fingerprint_rows(training,gen,chem_cache)
    vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(3,4),
                              max_features=4096,sublinear_tf=True,dtype=np.float32)
    protein_kmers=vectorizer.fit_transform([r["protein"] for r in training])
    protein_esm=np.asarray([embedding_map[sequence_hash(r["protein"])]
                            for r in training],dtype=np.float32)
    assert protein_esm.shape==(12315,ESM_DIM)
    X=hstack([drug_matrix,protein_kmers*.25,protein_esm*.25],format="csr")
    ridge=Ridge(alpha=1.,solver="lsqr")
    ridge.fit(X,(y-mu)/std)
    molecular,aa=data_features(training,{},{})
    boosting=HistGradientBoostingRegressor(**HIST_PARAMS)
    boosting.fit(np.column_stack([molecular,aa]),y)
    return {
        "model_kind":"EXPLORATORY_BALM_TRAIN_ONLY_MEMORY_CPI",
        "beta":1.0,"memory_lambda":.5,"memory_mix":.25,
        "train_rows_original":len(raw_rows),"train_rows_unique":len(training),
        "train_mean":mu,"train_sd":std,
        "protein_window":ESM_WINDOW,
        "memory_spec":MEMORY_SPEC,
        "ridge":ridge,"boosting":boosting,"protein_kmer_vectorizer":vectorizer,
        "train_fingerprints":drug_matrix,
        "train_embeddings":protein_esm,
        "train_labels":y,
        "known_embeddings":dict(embedding_map),
        "train_smiles":[r["smiles"] for r in training],
        "warnings":[
            "Not scientifically approved or externally validated; do not use for experimental decisions.",
            "Training source lacks per-measurement assay and construct provenance.",
            "Predicts source p-affinity-like numeric label, not validated physical Kd.",
            "No reserved test labels or predictions evaluated."
        ],
    }


def predict_with_embedding(model,smiles,protein,embedding):
    """Predict using validated, precomputed or frozen-inferred ESM representation."""
    smiles=canonical_smiles(smiles)
    protein=clean_protein(protein)
    vec=np.asarray(embedding,dtype=np.float32)
    if vec.shape!=(ESM_DIM,) or not np.isfinite(vec).all():
        raise ValueError("Expected finite ESM2 8M vector of dimension 320")
    length=np.linalg.norm(vec)
    if length<=0:raise ValueError("ESM representation has zero norm")
    vec=vec/length
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    drug=fingerprint_rows([{"smiles":smiles}],gen,{})
    sequence_features=model["protein_kmer_vectorizer"].transform([protein])
    row=hstack([drug,sequence_features*.25,vec.reshape(1,-1)*.25],format="csr")
    mu=model["train_mean"];std=model["train_sd"]
    ridge_pred=float(model["ridge"].predict(row)[0]*std+mu)
    chemical=molecule_features(smiles)
    pfeatures=protein_features(protein)
    boosted=float(model["boosting"].predict(np.r_[chemical,pfeatures].reshape(1,-1))[0])
    beta=model["beta"]
    original=mu+beta*(.75*ridge_pred+.25*boosted-mu)
    ligand_cos=(drug@model["train_fingerprints"].T).toarray().astype(np.float32)
    protein_cos=vec.reshape(1,-1)@model["train_embeddings"].T
    nearest=float(neighbor_weighted_labels(
        ligand_cos,protein_cos,model["train_labels"],mu,model["memory_spec"])[0])
    memory=mu+model["memory_lambda"]*(nearest-mu)
    pred=(1-model["memory_mix"])*original+model["memory_mix"]*memory
    return {
        "predicted_source_affinity_label":float(pred),
        "ridge_component":ridge_pred,
        "boosting_component":boosted,
        "memory_component":float(memory),
        "train_mean":float(mu),
        "nearest_molecular_fingerprint_cosine":float(ligand_cos.max()),
        "nearest_esm_protein_cosine":float(protein_cos.max()),
        "protein_length":len(protein),
        "protein_was_truncated":len(protein)>ESM_WINDOW,
        "status":"EXPLORATORY_UNVALIDATED_RESEARCH_ONLY",
        "note":"BALM labels contain unresolved assay differences; estimate is not a calibrated physical Kd.",
    }


def predict(model,smiles,protein,device=None):
    protein=clean_protein(protein)
    key=sequence_hash(protein)
    if key in model["known_embeddings"]:
        vec=model["known_embeddings"][key]
    else:
        vec=encode_new_protein(protein,device=device)
    return predict_with_embedding(model,smiles,protein,vec)
