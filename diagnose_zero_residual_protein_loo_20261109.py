"""Read-only v4.3 diagnostics: original zero-residual train loss and protein LOO.
Does not train or tune a model, open original DEV labels, or score protected TEST.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from profile_model import AffinityProfile, normalize

HERE = Path(__file__).resolve().parent
OLD = Path(r"C:\Users\J0124\dynacpi_v4")
RESULTS_PATH = HERE / "interaction_study_20261109" / "study_results.json"
OUT = HERE / "interaction_study_20261109" / "zero_residual_protein_loo_diagnostic_20261109"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def macro_rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth))**2, axis=1)).mean())


def center_per_protein(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    return a - a.mean(axis=-1, keepdims=True)


def main():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite existing diagnostic artifacts")
    original = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if len(original["folds"]) != 3:
        raise RuntimeError("Expected exact saved historical three-fold v4.3 result")
    rawpath = OLD / "data" / "engineering_pilot" / "rows.json"
    fppath = OLD / "data" / "staging" / "morgan_radius2_2048_chiral.npz"
    if digest(rawpath) != original["row_sha256"] or digest(fppath) != original["fingerprint_sha256"]:
        raise RuntimeError("Input row or fingerprint hashes disagree with historical study")
    train = [r for r in json.loads(rawpath.read_text(encoding="utf-8")) if r["split"] == "train"]
    pids, cids = original["protein_hashes"], original["compound_hashes"]
    assert len(pids) == 24 and len(cids) == 54 and len(train) == 1296
    targets = {(r["sequence"], r["compound"]): float(r["y"]) for r in train}
    y = np.array([[targets[p, c] for c in cids] for p in pids], dtype=np.float64)
    assert np.isfinite(y).all()

    with np.load(fppath, allow_pickle=False) as z:
        fp = dict(zip(z["ids"].astype(str), z["bits"]))
    bits = np.stack([fp[c] for c in cids]).astype(np.float64)

    features = []
    for p in pids:
        path = OLD / "data" / "esm2_t33_650M_layer33_w1000_o128_fp16" / (p + ".pt")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if obj["sequence_hash"] != p:
            raise RuntimeError("Wrong ESM protein cache hash")
        features.append(obj["embeddings"].float().mean(dim=0).numpy())
    features = np.stack(features).astype(np.float64)
    featnorm = normalize(features)
    sim = featnorm @ featnorm.T

    # LOO warm-compound target-centered selectivity. The oracle uses the
    # held-out target's TRUE labels to choose its neighbor: unattainable in
    # deployment, not a conventional upper bound on all regression models.
    per_target = []
    for q in range(len(pids)):
        remaining = [i for i in range(len(pids)) if i != q]
        truth = center_per_protein(y[q])
        chem_prior = center_per_protein(y[remaining].mean(axis=0))
        base_mse = float(np.mean((chem_prior - truth)**2))
        candidate_dist = np.array([
            np.mean((center_per_protein(y[i]) - truth)**2) for i in remaining
        ], dtype=np.float64)
        oracle_idx = remaining[int(np.argmin(candidate_dist))]
        nearest_idx = remaining[int(np.argmax(sim[q, remaining]))]
        per_target.append({
            "chemical_mean_centered_mse": base_mse,
            "label_informed_best_single_protein_centered_mse": float(candidate_dist.min()),
            "esm_cosine_nearest_single_protein_centered_mse":
                float(np.mean((center_per_protein(y[nearest_idx]) - truth)**2)),
            "esm_cosine_to_nearest": float(sim[q, nearest_idx]),
            "oracle_matches_esm_nearest": bool(oracle_idx == nearest_idx),
        })
    loo_summary = {
        "protocol": "Leave-one-protein-out among original 24 DAVIS TRAIN proteins; all 54 chemicals warm",
        "base_centered_mse": float(np.mean([v["chemical_mean_centered_mse"] for v in per_target])),
        "oracle_best_single_centered_mse": float(np.mean([v["label_informed_best_single_protein_centered_mse"] for v in per_target])),
        "esm_cosine_nearest_centered_mse": float(np.mean([v["esm_cosine_nearest_single_protein_centered_mse"] for v in per_target])),
        "oracle_matched_esm_nearest_count": sum(v["oracle_matches_esm_nearest"] for v in per_target),
        "nearest_esm_cosine_min": float(min(v["esm_cosine_to_nearest"] for v in per_target)),
        "nearest_esm_cosine_max": float(max(v["esm_cosine_to_nearest"] for v in per_target)),
        "nearest_esm_cosine_mean": float(np.mean([v["esm_cosine_to_nearest"] for v in per_target])),
        "oracle_limitation": "Chosen using heldout protein's measured y vector; only an optimistic label-informed nearest SINGLE donor diagnostic, not a deployable model or upper bound on mixtures.",
    }
    zero_records = []
    fold_parity = []
    for f, rec in enumerate(original["folds"]):
        pi = sorted(set(range(24)) - set(original["outer_protein_folds"][f]))
        ci = sorted(set(range(54)) - set(original["outer_chemical_folds"][f]))
        po = original["outer_protein_folds"][f]
        co = original["outer_chemical_folds"][f]
        assert len(pi) == 16 and len(ci) == 36 and len(po) == 8 and len(co) == 18
        choice = rec["inner_choices"]["chemical"]
        selected_ridge = float(choice["chemical_ridge"])
        residual_ridge = float(choice["residual_ridge"])
        prior = AffinityProfile(
            chemical_ridge=selected_ridge, residual_ridge=residual_ridge,
            mode="chemical"
        ).fit(features[pi], bits[ci], y[np.ix_(pi, ci)])
        allchem = prior.chemical(bits)
        pred = np.broadcast_to(allchem[co], (len(po), len(co)))
        observed = macro_rmse(pred, y[np.ix_(po, co)])
        historical = rec["models"]["chemical_profile"]["double_cold"]["macro_rmse"]
        if abs(observed - historical) > 1e-10:
            raise RuntimeError("Reconstructed prior did not match historical fold " +
                               str((f, observed, historical)))
        fold_parity.append({"fold": f, "original_macro_rmse": historical,
                            "recomputed_macro_rmse": observed})
        train_y = y[np.ix_(pi, ci)]
        train_sd = float(train_y.std())
        zero_loss = float(np.mean(((train_y - allchem[ci][None, :]) / train_sd)**2))
        step1 = []
        step5 = []
        per_seed = []
        for seed in (2931, 2932, 2933):
            h = rec["training_history"][f"anchored_neural_seed{seed}"]
            first, last = h[0]["standardized_train_loss"], h[-1]["standardized_train_loss"]
            step1.append(first)
            step5.append(last)
            per_seed.append({"seed": seed, "epoch1_train_mse": first,
                             "epoch5_train_mse": last,
                             "loss_curve": [z["standardized_train_loss"] for z in h],
                             "zero_residual_train_mse": zero_loss,
                             "epoch5_minus_zero": last - zero_loss})
        zero_records.append({
            "fold": f, "train_proteins": len(pi), "train_compounds": len(ci),
            "fit_only_y_sd": train_sd,
            "chemical_prior_ridge": selected_ridge,
            "original_training_zero_residual_mse": zero_loss,
            "mean_anchored_epoch1_train_mse": float(np.mean(step1)),
            "mean_anchored_epoch5_train_mse": float(np.mean(step5)),
            "epoch5_minus_zero_residual": float(np.mean(step5) - zero_loss),
            "per_seed": per_seed
        })
    overall = {
        "zero_residual_standardized_train_mse": float(np.mean([v["original_training_zero_residual_mse"] for v in zero_records])),
        "anchored_epoch1_mean_train_mse": float(np.mean([v["mean_anchored_epoch1_train_mse"] for v in zero_records])),
        "anchored_epoch5_mean_train_mse": float(np.mean([v["mean_anchored_epoch5_train_mse"] for v in zero_records])),
        "epoch5_minus_zero": float(np.mean([v["epoch5_minus_zero_residual"] for v in zero_records])),
    }
    results = {
        "status": "PASS_READ_ONLY_HISTORICAL_TRAIN_DIAGNOSTIC",
        "new_training": False,
        "unseen_external_test": False,
        "protein_count": 24, "chemical_count": 54,
        "train_data_sha256": digest(rawpath),
        "fingerprint_sha256": digest(fppath),
        "source_results_sha256": digest(RESULTS_PATH),
        "source_script_sha256": digest(HERE / "diagnose_zero_residual_protein_loo_20261109.py"),
        "zero_residual_folds": zero_records,
        "overall_zero_residual": overall,
        "loo": loo_summary,
        "old_saved_cv_chemical_parity": fold_parity,
        "limitations": [
            "The zero-residual loss is an initial fixed-prior TRAIN target comparator, not an inner validation metric.",
            "Epoch averages are over proteins and epoch minibatches, while zero-loss is all fit pairs.",
            "The oracle uses heldout target affinity labels to pick the best SINGLE neighbor: not a deployable predictor.",
            "LOO among known training proteins and known compounds is exploratory, not a family-disjoint test.",
            "Uses frozen mean-pooled ESM embeddings; model choices or feature tuning on these 24 proteins would be exploratory."
        ],
    }
    OUT.mkdir(parents=True)
    (OUT / "RESULTS.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "overall_zero_residual": overall,
        "fold_zero_residual": [
            {"fold": v["fold"], "zero": v["original_training_zero_residual_mse"],
             "epoch1": v["mean_anchored_epoch1_train_mse"],
             "epoch5": v["mean_anchored_epoch5_train_mse"]}
            for v in zero_records],
        "loo": loo_summary,
        "parity": fold_parity
    }, indent=2))
    print("PASS_ZERO_RESIDUAL_AND_PROTEIN_LOO")


if __name__ == "__main__":
    main()
