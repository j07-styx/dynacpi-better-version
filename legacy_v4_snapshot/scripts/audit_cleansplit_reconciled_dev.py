"""Source-aware split and label audit for the reconciled DAVIS development cohort.

Inspired by CleanSplit's emphasis on contamination-aware evaluation.
This script does not approve or generate a test set.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "engineering_pilot_primary_reconciled_20261107"
OUT = ROOT / "research" / "reconciled_cleansplit_audit_dev.json"


def audit(rows):
    splits = {k: [r for r in rows if r["split"] == k] for k in ("train", "val")}
    if sum(map(len, splits.values())) != len(rows):
        raise ValueError("Unexpected split role")
    for split, rr in splits.items():
        pairs = [(r["compound"], r["sequence"]) for r in rr]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate compound-protein pairs in " + split)
    train, val = splits["train"], splits["val"]
    fields = ("compound", "sequence", "primary_compound", "primary_protein")
    overlap = {}
    for field in fields:
        a, b = ({r[field] for r in train}, {r[field] for r in val})
        overlap[field] = {"train": len(a), "val": len(b),
                          "shared": len(a & b), "shared_values": sorted(a & b)}
    pair_overlap = sorted(set((r["compound"], r["sequence"]) for r in train) &
                          set((r["compound"], r["sequence"]) for r in val))
    strata = {}
    for split, rr in splits.items():
        strata[split] = dict(Counter(r["primary_label_kind"] for r in rr))
        for r in rr:
            kind = r["primary_label_kind"]
            if kind == "numeric_Kd" and r["numeric_pKd"] is None:
                raise ValueError("Numeric label missing source numeric_pKd")
            if kind != "numeric_Kd" and r["numeric_pKd"] is not None:
                raise ValueError("Non-numeric source label has numeric_pKd")
    by_target = defaultdict(lambda: Counter())
    for r in rows:
        by_target[r["primary_protein"]][r["split"] + ":" + r["primary_label_kind"]] += 1
    return {
        "status": "DEVELOPMENT_AUDIT_ONLY_NOT_INDEPENDENT_TEST",
        "rows": len(rows), "strata": strata,
        "overlap": overlap, "pair_overlap_count": len(pair_overlap),
        "same_compound_seen_in_both_splits": bool(overlap["compound"]["shared"]),
        "same_sequence_seen_in_both_splits": bool(overlap["sequence"]["shared"]),
        "numeric_target_strata": {k: dict(v) for k, v in sorted(by_target.items())},
        "protocol_warnings": [
            "Existing development validation has been repeatedly inspected.",
            "Shared compounds prevent cold-compound claims.",
            "No experimentally verified protein construct identity.",
            "Kinase group STE is present in both splits; not family-disjoint.",
            "Publisher blank/weak-binding cells are not quantitative Kd or validated censoring bounds.",
            "No independently frozen holdout/test set exists.",
            "This audit cannot detect structural or chemical similarity leakage without additional source verification."
        ],
        "recommendation": "Freeze a source-verified, label-blind, protein-component-disjoint external evaluation before comparing against published PerceiverCPI metrics."
    }


def main():
    rows = json.loads((DATA / "rows.json").read_text(encoding="utf-8"))
    result = audit(rows)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("numeric_target_strata", "overlap")}, indent=2))
    print("Audit saved:", OUT)


if __name__ == "__main__":
    main()

