"""Nonlinear physicochemical + protein composition baseline on frozen BindingDB inner dev.

No hyperparameter search or model selection on validation. No reserved test metrics.
"""
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "audit"
OUT = ROOT / "research" / "bindingdb_descriptor_protein_boosting_component_weighted_inner_dev.json"
RDLogger.DisableLog("rdApp.*")
AMINO = "ACDEFGHIKLMNPQRSTVWY"
PARAMS = dict(max_iter=160, learning_rate=0.05, max_leaf_nodes=15, max_depth=5,
              min_samples_leaf=80, l2_regularization=15.0, early_stopping=False,
              random_state=2103)

def records(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)

def molecule_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles[:40]}")
    return np.array([
        Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHAcceptors(mol), Lipinski.NumHDonors(mol),
        Lipinski.NumRotatableBonds(mol), Lipinski.RingCount(mol),
        Lipinski.NumAromaticRings(mol), Lipinski.NumAliphaticRings(mol),
        Lipinski.NumSaturatedRings(mol), rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(), mol.GetNumAtoms(),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),
        Descriptors.HeavyAtomMolWt(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
    ], dtype=np.float32)

def protein_features(seq):
    length = len(seq)
    if not length:
        raise ValueError("Empty protein sequence")
    counts = np.array([seq.count(ch) for ch in AMINO], dtype=np.float32) / length
    return np.r_[counts, np.float32(np.log1p(length))].astype(np.float32)

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.time()
    manifest = json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    tr_ids = set(manifest["train_components"])
    va_ids = set(manifest["inner_dev_components"])
    te_ids = set(manifest["test_components"])
    assert tr_ids.isdisjoint(va_ids | te_ids) and va_ids.isdisjoint(te_ids)
    mapping = {}
    for group in records(AUDIT/"bindingdb_full_kd_protein_components.jsonl"):
        for seq_hash in group["members"]:
            mapping[seq_hash] = group["component_id"]
    allowed_tasks = {(x["sequence_sha256"],x["assay_key"])
                     for x in records(AUDIT/"bindingdb_full_kd_task_summary.jsonl")
                     if x["unique_canonical_compounds"] >= 15}
    train, valid = [], []
    for item in records(AUDIT/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        group = mapping[item["protein_seq_sha256"]]
        if group not in tr_ids and group not in va_ids:
            continue  # reserved test excluded before accessing any label
        if (item["protein_seq_sha256"], item["assay_key"]) not in allowed_tasks:
            continue
        if item["range_pKd"] > 0.1:
            continue
        (train if group in tr_ids else valid).append(item)
    assert len(train) == manifest["train_pairs"] == 27118
    assert len(valid) == manifest["inner_dev_pairs"] == 7387

    needed = {x["protein_seq_sha256"] for x in train + valid}
    seqs = {}
    for item in records(AUDIT/"bindingdb_full_kd_candidates.jsonl"):
        seq = item["sequence"]
        key = hashlib.sha256(seq.encode()).hexdigest()
        if key in needed:
            seqs[key] = seq
            if len(seqs) == len(needed):
                break
    assert len(seqs) == len(needed), (len(seqs), len(needed))
    chemicals = {}
    proteins = {key: protein_features(seq) for key, seq in seqs.items()}

    def matrices(items):
        c = np.zeros((len(items), 21), dtype=np.float32)
        p = np.zeros((len(items), 21), dtype=np.float32)
        for i, item in enumerate(items):
            smi = item["smiles_isomeric"]
            if smi not in chemicals:
                chemicals[smi] = molecule_features(smi)
            c[i] = chemicals[smi]
            p[i] = proteins[item["protein_seq_sha256"]]
        assert np.isfinite(c).all() and np.isfinite(p).all()
        return c, p

    xtr, ptr = matrices(train)
    xva, pva = matrices(valid)
    ytr = np.asarray([r["median_pKd"] for r in train], dtype=np.float32)
    yva = np.asarray([r["median_pKd"] for r in valid], dtype=np.float32)
    mean = float(ytr.astype(np.float64).mean())
    comps = np.array([mapping[r["protein_seq_sha256"]] for r in valid])
    unique_comps = sorted(set(comps))
    def metrics(pred):
        err = (pred.astype(np.float64) - yva.astype(np.float64))**2
        baseline = (mean - yva.astype(np.float64))**2
        rmse = math.sqrt(float(err.mean()))
        per_component = [math.sqrt(float(err[comps==g].mean())) for g in unique_comps]
        wins = int(sum(err[comps==g].mean() < baseline[comps==g].mean() for g in unique_comps))
        return {"pooled_rmse":rmse,
                "macro_component_rmse":float(np.mean(per_component)),
                "components_better_than_mean":wins}
    baseline_metrics = metrics(np.full(len(valid), mean, dtype=np.float64))
    group_counts = {}
    for item in train:
        component = mapping[item["protein_seq_sha256"]]
        group_counts[component] = group_counts.get(component, 0) + 1
    sample_weights = np.asarray(
        [group_counts[mapping[item["protein_seq_sha256"]]] ** (-0.5) for item in train],
        dtype=np.float64
    )
    sample_weights /= sample_weights.mean()
    results = {}
    for name, xtrain, xvalid in [
        ("chemical_descriptors_only", xtr, xva),
        ("chemical_plus_protein_composition", np.column_stack([xtr,ptr]), np.column_stack([xva,pva]))
    ]:
        print("FITTING", name, "shape", xtrain.shape, flush=True)
        model = HistGradientBoostingRegressor(**PARAMS)
        model.fit(xtrain, ytr, sample_weight=sample_weights)
        results[name] = metrics(model.predict(xvalid))
        print(json.dumps({"mode":name, **results[name]}),flush=True)
    output = {
        "status":"EXPLORATORY_PROVISIONAL_FROZEN_INNER_DEV_ONLY_NO_TEST",
        "train_pairs":len(train),"inner_dev_pairs":len(valid),
        "train_components":len(tr_ids),"inner_dev_components":len(va_ids),
        "model":"sklearn HistGradientBoostingRegressor, train-only fitted","train_weighting":"inverse-sqrt component frequency, normalized to mean 1",
        "fixed_parameters":PARAMS,
        "chemical_feature_count":xtr.shape[1],
        "protein_feature_count":ptr.shape[1],
        "unique_compounds_featurized":len(chemicals),
        "baseline_train_mean":baseline_metrics,
        "models":results,"seconds":round(time.time()-start,2),
        "warnings":["Exploratory repeatedly inspected development cohort; no reserved test evaluated",
                    "Assay and construct QC incomplete; no claim of benchmark superiority",
                    "No validation-based hyperparameter optimization or checkpoint selection"]
    }
    OUT.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2),flush=True)

if __name__=="__main__":
    main()


