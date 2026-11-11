"""Read-only KIBA/METZ frozen-protocol integrity audit.

No model inference, no training, no held-out labels or predictions loaded.
Hashes private paired-prediction files as opaque bytes; inspects public-facing
aggregate metrics only for scenarios already officially scored once.
Uses built-in Python only to avoid competing for GPU/system RAM.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(r"C:\Users\J0124\DynaCPI-Backend")
OUT = ROOT / "external_benchmarks_20261110" / "locked_external_protocol_v1"
ONCE = OUT / "INDIVIDUAL_EXTERNAL_SAME_SPLIT_TESTS_ONCE_20261111"
SEEDS = [2931, 2932]
SCENARIOS = ("protein_cold", "compound_cold", "double_cold")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def audit() -> dict:
    rows = []
    for ds in ("kiba", "metz"):
        feature_hash = digest(OUT / ds / "source_features.npz")
        for scenario in SCENARIOS:
            label = f"{ds}/{scenario}"
            folder = OUT / ds / scenario
            split_file = folder / "SPLIT_LOCKED.json"
            split_hash = digest(split_file)
            split = read(split_file)
            frozen = read(folder / "dynacpi_fixed_sparse_train" / "VALIDATION_FROZEN.json")
            check(split["dataset"] == ds and split["scenario"] == scenario,
                  f"{label}: mislabeled split")
            check(split["source_features_sha256"].lower() == feature_hash,
                  f"{label}: source features changed")
            check(frozen["source_features_sha256"].lower() == feature_hash,
                  f"{label}: frozen selection uses changed data")
            check(frozen["frozen_split_sha256"].lower() == split_hash,
                  f"{label}: frozen selection split changed")
            check(bool(split["test_labels_not_exported"]), f"{label}: test export flag")
            check(frozen["training_seeds"] == SEEDS, f"{label}: Dyna seeds changed")
            check(frozen["status"] == "VALIDATION_LOCKED_TWO_SEEDS_NO_TEST",
                  f"{label}: Dyna validation not frozen")
            sets = {}
            for part in ("train", "val", "test"):
                vals = split[f"pair_{part}_row_indices"]
                indices = set(map(int, vals))
                check(len(indices) == len(vals), f"{label}: repeated {part} rows")
                sets[part] = indices
                check(len(indices) == split["n_pairs"][part],
                      f"{label}: inconsistent {part} count")
            check(not (sets["train"] & sets["val"]
                       or sets["train"] & sets["test"]
                       or sets["val"] & sets["test"]),
                  f"{label}: observed pair split overlap")
            result_dir = ONCE / f"{ds}_{scenario}"
            result_file = result_dir / "TEST_RESULTS.json"
            row = {"dataset": ds, "scenario": scenario,
                   "train_pairs": len(sets["train"]),
                   "validation_pairs": len(sets["val"]),
                   "planned_test_pairs": len(sets["test"]),
                   "split_sha256": split_hash, "feature_sha256": feature_hash}
            if result_file.exists():
                result = read(result_file)
                check(result["status"] == "SAME_SPLIT_TWO_SEED_ONCE_TEST_COMPLETE",
                      f"{label}: incomplete result")
                check(result["dataset"] == ds and result["scenario"] == scenario,
                      f"{label}: mislabeled result")
                check(result["source_features_sha256"].lower() == feature_hash
                      and result["frozen_split_sha256"].lower() == split_hash,
                      f"{label}: scored provenance mismatch")
                check(result["n_test_pairs"] == len(sets["test"]),
                      f"{label}: scored count mismatched")
                opaque = result_dir / "paired_raw_predictions_private.npz"
                check(digest(opaque) == result["paired_raw_pred_sha256"].lower(),
                      f"{label}: scored private-prediction hash mismatch")
                boot = read(result_dir / "PAIRED_CLUSTER_BOOTSTRAP.json")
                check(boot["original_test_result_sha256"].lower() == digest(result_file)
                      and boot["original_raw_predictions_sha256"].lower() == digest(opaque),
                      f"{label}: bootstrap provenance mismatch")
                row.update(status="SCORED_ONCE",
                           delta_macro=result["delta_macro_dyna_minus_perceiver"],
                           delta_pooled=result["delta_pooled_dyna_minus_perceiver"],
                           ci_macro=boot["bootstrap"]["macro_delta_95pct_interval"],
                           ci_pooled=boot["bootstrap"]["pooled_delta_95pct_interval"])
            else:
                check(not result_dir.exists(), f"{label}: partial once-only test!")
                row["status"] = "SEALED_NOT_SCORED"
            rows.append(row)
    return {"audit": "PASS", "no_heldout_test_values_opened": True,
            "model_training_or_test_scoring_performed": False,
            "datasets": ["kiba", "metz"], "scenarios": rows,
            "scored_scenarios": sum(x["status"] == "SCORED_ONCE" for x in rows),
            "sealed_scenarios": sum(x["status"] == "SEALED_NOT_SCORED" for x in rows)}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
