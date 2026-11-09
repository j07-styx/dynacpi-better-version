"""Cold-protein nearest-sequence prior with train-only grouped CV.

The prospective test component labels and features are not used. All model
selection uses train groups only; inner-development is reported once afterward.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "audit"
REPORT = ROOT / "research" / "bindingdb_cold_protein_sequence_prior_inner_dev.json"


def read(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def metric(y, pred, groups):
    residual = (np.asarray(y) - np.asarray(pred)) ** 2
    pooled = float(np.sqrt(np.mean(residual)))
    group_macro = float(np.mean([
        np.sqrt(np.mean(residual[np.asarray(groups) == g]))
        for g in np.unique(groups)
    ]))
    return {"pooled_rmse": pooled, "component_macro_rmse": group_macro}


def predict_prior(train_rows, eval_rows, seqs, power, gate_power, scale, neighbors=3):
    # Protein-component centroids avoid weighting very large assay groups by their pair count.
    by_comp = defaultdict(list)
    for row in train_rows:
        by_comp[row["component"]].append(row)
    labels = np.asarray([x["label"] for x in train_rows], dtype=np.float64)
    global_mean = float(labels.mean())
    components = sorted(by_comp)
    training_seqs = sorted({r["protein"] for r in train_rows})
    validation_seqs = sorted({r["protein"] for r in eval_rows})
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 4),
                                 max_features=8192, sublinear_tf=True,
                                 dtype=np.float32)
    xtrain = vectorizer.fit_transform([seqs[s] for s in training_seqs])
    xval = vectorizer.transform([seqs[s] for s in validation_seqs])
    xtrain = normalize(xtrain, norm="l2")
    xval = normalize(xval, norm="l2")
    index = {s: i for i, s in enumerate(training_seqs)}
    prot_to_component = {r["protein"]: r["component"] for r in train_rows}
    centroid = []
    component_means = []
    for c in components:
        proteins = sorted({r["protein"] for r in by_comp[c]})
        centroid.append(xtrain[[index[s] for s in proteins]].mean(axis=0))
        # Equal weighting per protein avoids dominating the prior with large assays.
        protein_mean = [np.mean([r["label"] for r in by_comp[c] if r["protein"] == p])
                        for p in proteins]
        component_means.append(float(np.mean(protein_mean)))
    from scipy.sparse import csr_matrix
    centroids = normalize(vstack([csr_matrix(m) for m in centroid]), norm="l2")
    sim = (xval @ centroids.T).toarray()
    component_means = np.asarray(component_means)
    val_pred = {}
    for i, protein in enumerate(validation_seqs):
        row = sim[i]
        best = np.argsort(-row, kind="stable")[:neighbors]
        strengths = np.maximum(row[best], 0.0)
        weights = np.maximum(strengths ** power, 1e-12)
        adjacent_mean = float(np.average(component_means[best], weights=weights))
        maxsim = float(np.max(strengths))
        val_pred[protein] = global_mean + scale * maxsim ** gate_power * (adjacent_mean - global_mean)
    return np.asarray([val_pred[r["protein"]] for r in eval_rows], dtype=np.float64)


def main():
    m = json.loads((ROOT / "research" / "bindingdb_frozen_component_split_manifest_dev.json").read_text())
    train_groups = set(m["train_components"])
    dev_groups = set(m["inner_dev_components"])
    test_groups = set(m["test_components"])
    assert not train_groups & dev_groups and not dev_groups & test_groups and not train_groups & test_groups
    components = {}
    for group in read(AUDIT / "bindingdb_full_kd_protein_components.jsonl"):
        for key in group["members"]:
            components[key] = group["component_id"]
    tasks = {
        (r["sequence_sha256"], r["assay_key"])
        for r in read(AUDIT / "bindingdb_full_kd_task_summary.jsonl")
        if r["unique_canonical_compounds"] >= 15
    }
    train = []
    dev = []
    for row in read(AUDIT / "bindingdb_full_kd_compound_median_candidates.jsonl"):
        if ((row["protein_seq_sha256"], row["assay_key"]) not in tasks or
                row["range_pKd"] > .1):
            continue
        group = components[row["protein_seq_sha256"]]
        # Never materialize test samples, target labels, or predictions.
        if group not in train_groups and group not in dev_groups:
            continue
        sample = {"component": group, "protein": row["protein_seq_sha256"],
                  "label": float(row["median_pKd"])}
        if group in train_groups:
            train.append(sample)
        else:
            dev.append(sample)
    assert len(train) == m["train_pairs"] and len(dev) == m["inner_dev_pairs"]
    needed = {r["protein"] for r in train + dev}
    seqs = {}
    for row in read(AUDIT / "bindingdb_full_kd_candidates.jsonl"):
        s = row["sequence"]
        key = hashlib.sha256(s.encode()).hexdigest()
        if key in needed:
            seqs[key] = s
        if len(seqs) == len(needed):
            break
    assert len(seqs) == len(needed)
    groups = np.asarray([r["component"] for r in train])
    cv = GroupShuffleSplit(n_splits=3, test_size=.2, random_state=20261108)
    configs = [(pw, gate, scale) for pw in (1, 4)
               for gate in (0, 2, 4) for scale in (.25, .5, 1.)]
    results = {str(c): [] for c in configs}
    baselines = []
    for fold, (itr, iva) in enumerate(cv.split(groups, groups=groups), 1):
        tr = [train[int(i)] for i in itr]
        va = [train[int(i)] for i in iva]
        train_mean = float(np.mean([r["label"] for r in tr]))
        y = np.asarray([r["label"] for r in va])
        g = np.asarray([r["component"] for r in va])
        baseline = metric(y, np.full_like(y, train_mean), g)
        baselines.append(baseline)
        print(json.dumps({"fold": fold, "train": len(tr), "val": len(va),
                          "train_mean": baseline}), flush=True)
        # Compute expensive sequence neighbor scores once per fold per weight power.
        for power in (1, 4):
            prior = predict_prior(tr, va, seqs, power, 0, 1.)
            delta = prior - train_mean
            # To compute similarity-aware gate we need per-protein similarity.
            # Gated predictions are computed independently below.
            for gate in (0, 2, 4):
                if gate == 0:
                    gated = prior
                else:
                    gated = predict_prior(tr, va, seqs, power, gate, 1.)
                for scale in (.25, .5, 1.):
                    pred = train_mean + scale * (gated - train_mean)
                    results[str((power, gate, scale))].append(metric(y, pred, g))
    def mean_score(xs, name):
        return float(np.mean([x[name] for x in xs]))
    sorted_results = []
    for c in configs:
        xs = results[str(c)]
        score = .5 * (mean_score(xs, "pooled_rmse") / mean_score(baselines, "pooled_rmse") +
                      mean_score(xs, "component_macro_rmse") / mean_score(baselines, "component_macro_rmse"))
        sorted_results.append({"power": c[0], "gate_power": c[1], "scale": c[2],
                               "cv_combined_relative_error": float(score),
                               "cv_pooled_mean": mean_score(xs, "pooled_rmse"),
                               "cv_macro_mean": mean_score(xs, "component_macro_rmse")})
    sorted_results.sort(key=lambda r: r["cv_combined_relative_error"])
    chosen = sorted_results[0]
    # Only compare with baseline after the hyperparameters are selected on training CV.
    baseline_cv_score = 1.0
    if chosen["cv_combined_relative_error"] < baseline_cv_score:
        best = (chosen["power"], chosen["gate_power"], chosen["scale"])
    else:
        best = None
    dev_y = np.asarray([r["label"] for r in dev])
    dev_g = np.asarray([r["component"] for r in dev])
    global_mean = float(np.mean([r["label"] for r in train]))
    dev_mean_metrics = metric(dev_y, np.full_like(dev_y, global_mean), dev_g)
    if best:
        pred = predict_prior(train, dev, seqs, *best)
        dev_metrics = metric(dev_y, pred, dev_g)
    else:
        pred = np.full_like(dev_y, global_mean)
        dev_metrics = dev_mean_metrics
    # A second configuration is selected by minimizing pooled grouped-CV error,
    # subject to BOTH pooled and macro grouped-CV error improving on the mean.
    pooled_ref = mean_score(baselines, "pooled_rmse")
    macro_ref = mean_score(baselines, "component_macro_rmse")
    safe_candidates = sorted([
        r for r in sorted_results
        if r["cv_pooled_mean"] <= pooled_ref and r["cv_macro_mean"] <= macro_ref
    ], key=lambda r: (r["cv_pooled_mean"], r["cv_macro_mean"]))
    safe_choice = safe_candidates[0] if safe_candidates else None
    safe_params = (
        safe_choice["power"], safe_choice["gate_power"], safe_choice["scale"]
    ) if safe_choice else None
    if safe_params:
        safe_pred = predict_prior(train, dev, seqs, *safe_params)
    else:
        safe_pred = np.full_like(dev_y, global_mean)
    safe_metrics = metric(dev_y, safe_pred, dev_g)
    per_component = []
    for c in np.unique(dev_g):
        mask = dev_g == c
        per_component.append({
            "pair_count": int(mask.sum()),
            "train_mean_rmse": float(np.sqrt(np.mean((dev_y[mask] - global_mean)**2))),
            "macro_selected_rmse": float(np.sqrt(np.mean((dev_y[mask] - pred[mask])**2))),
            "pooled_safe_rmse": float(np.sqrt(np.mean((dev_y[mask] - safe_pred[mask])**2)))
        })
    per_component.sort(key=lambda r: r["pair_count"], reverse=True)
    out = {
        "status": "EXPLORATORY_TRAIN_ONLY_GROUP_CV_AND_FROZEN_INNER_DEV_NO_TEST",
        "train_pairs": len(train), "dev_pairs": len(dev),
        "train_components": len(train_groups), "dev_components": len(dev_groups),
        "method": "Nearest-component protein 3-4mer TF-IDF priors with similarity shrinkage",
        "train_only_group_folds": len(baselines),
        "cv_baseline_per_fold": baselines,
        "cv_candidate_ranking": sorted_results,
        "cv_selected_configuration_or_train_mean": list(best) if best else "train_mean",
        "dev_train_mean": dev_mean_metrics, "dev_selected_model": dev_metrics,
        "cv_safe_pooled_choice": list(safe_params) if safe_params else "train_mean",
        "dev_safe_pooled_model": safe_metrics,
        "component_error_diagnostic_anonymous": per_component,
        "macro_selected_components_better_than_mean": int(sum(
            z["macro_selected_rmse"] < z["train_mean_rmse"] for z in per_component)),
        "safe_components_better_than_mean": int(sum(
            z["pooled_safe_rmse"] < z["train_mean_rmse"] for z in per_component)),
        "warnings": [
            "Frozen dev set has been inspected repeatedly by other experiments.",
            "Protein sequence similarity estimates need not reflect experimental construct equivalence.",
            "Assay provenance remains incomplete.",
            "Reserved test set was not evaluated or accessed as a feature or label source.",
            "Hyperparameters chosen on grouped train-only CV; inner-dev metrics diagnostic only."
        ]
    }
    REPORT.write_text(json.dumps(out, indent=2))
    print(json.dumps({"chosen": out["cv_selected_configuration_or_train_mean"],
                      "dev_baseline": dev_mean_metrics, "dev_model": dev_metrics,
                      "safe_cv_selected": list(safe_params) if safe_params else "train_mean",
                      "dev_safe_pooled_model": safe_metrics,
                      "components_better_macro": out["macro_selected_components_better_than_mean"],
                      "components_better_safe": out["safe_components_better_than_mean"],
                      "largest_components": per_component[:3],
                      "top_cv": sorted_results[:4]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
