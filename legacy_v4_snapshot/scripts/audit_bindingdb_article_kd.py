"""Preliminary, source-preserving Kd eligibility scan of the November 2026
BindingDB *Articles* subset. Not a final cleaned cohort or a train/test split.
Does not normalize chemistry, certify assay compatibility, or resolve constructs.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bindingdb_202611"
OUT = ROOT / "data" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
EXACT = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
CANONICAL_AA = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+\Z")
csv.field_size_limit(2_147_483_647)
USE_FULL = '--full' in sys.argv
DATA_STEM = 'BindingDB_All_202611_tsv' if USE_FULL else 'BindingDB_BindingDB_Articles_202611_tsv'
OUTPUT_STEM = 'bindingdb_full_kd' if USE_FULL else 'bindingdb_articles_kd'

def open_rows(stem):
    z = zipfile.ZipFile(DATA / (stem + ".zip"))
    member = next(m for m in z.namelist() if m.lower().endswith(".tsv"))
    f = io.TextIOWrapper(z.open(member), encoding="utf-8-sig", errors="replace", newline="")
    return z, f, csv.DictReader(f, delimiter="\t")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    stats = collections.Counter()
    candidates = {}  # reactant ID -> (construct seq, raw SMILES, pKd, DOI)
    z, f, rows = open_rows(DATA_STEM)
    chain_col = "Number of Protein Chains in Target (>1 implies a multichain complex)"
    try:
        for row in rows:
            stats["source_rows"] += 1
            if stats["source_rows"] % 250000 == 0:
                print(json.dumps({"scanned": stats["source_rows"], "kd_present": stats["kd_nonempty"]}), flush=True)
            kd = (row.get("Kd (nM)") or "").strip()
            if not kd:
                continue
            stats["kd_nonempty"] += 1
            if not EXACT.fullmatch(kd):
                if kd.lstrip().startswith(("<", ">", "≤", "≥")):
                    stats["kd_qualified_bound"] += 1
                else:
                    stats["kd_other_nonexact"] += 1
                continue
            try:
                kd_nM = float(kd)
            except (ValueError, OverflowError):
                stats["kd_unparseable"] += 1
                continue
            if not math.isfinite(kd_nM) or kd_nM <= 0:
                stats["kd_nonpositive_or_nonfinite"] += 1
                continue
            stats["kd_positive_exact_lexical"] += 1
            if (row.get(chain_col) or "").strip() != "1":
                stats["excluded_non_single_chain_or_unknown"] += 1
                continue
            seq = "".join((row.get("BindingDB Target Chain Sequence 1") or "").split()).upper()
            if not seq or not CANONICAL_AA.fullmatch(seq):
                stats["excluded_missing_or_noncanonical_seq"] += 1
                continue
            smi = (row.get("Ligand SMILES") or "").strip()
            if not smi:
                stats["excluded_missing_smiles"] += 1
                continue
            rid = (row.get("BindingDB Reactant_set_id") or "").strip()
            if not rid:
                stats["excluded_missing_reactant_id"] += 1
                continue
            if rid in candidates:
                stats["duplicate_reactant_ids"] += 1
                continue
            candidates[rid] = (seq, smi, 9.0 - math.log10(kd_nM), (row.get("Article DOI") or "").strip())
            stats["preliminary_candidate_rows"] += 1
    finally:
        f.close()
        z.close()

    # Join only the candidate reactant IDs, retaining no unrelated assay data.
    mapped = {}
    z, f, rows = open_rows("BindingDB_rsid_eaids_202611_tsv")
    try:
        for row in rows:
            rid = (row.get("REACTANT_SET_ID") or "").strip()
            if rid in candidates:
                assay = (row.get("ENTRYID_ASSAYID") or "").strip()
                if assay and assay != "-":
                    mapped.setdefault(rid, assay)
    finally:
        f.close()
        z.close()

    tasks = collections.defaultdict(lambda: collections.defaultdict(list))
    for rid, (seq, smi, pkd, doi) in candidates.items():
        assay = mapped.get(rid)
        if not assay:
            stats["excluded_no_assay_link"] += 1
            continue
        tasks[(seq, assay)][smi].append((pkd, rid))
        stats["candidate_assay_linked_rows"] += 1

    # Preserve source rows for later RDKit and protein-cluster QC without re-scanning the archive.
    with (OUT / (OUTPUT_STEM + "_candidates.jsonl")).open("w", encoding="utf-8") as out:
        for (seq, assay), compounds in sorted(tasks.items()):
            for smi, observations in sorted(compounds.items()):
                for pkd, rid in observations:
                    source_doi = candidates[rid][3]
                    out.write(json.dumps({"reactant_set_id": rid, "assay_key": assay,
                                          "sequence": seq, "smiles_raw": smi,
                                          "pKd": pkd, "article_doi": source_doi}) + "\n")

    sizes = [len(compounds) for compounds in tasks.values()]
    viable15 = [(seq, assay, compounds) for (seq, assay), compounds in tasks.items() if len(compounds) >= 15]
    viable30 = [(seq, assay, compounds) for (seq, assay), compounds in tasks.items() if len(compounds) >= 30]
    conflicting = 0
    dup_records = 0
    for comp in tasks.values():
        for obs in comp.values():
            if len(obs) > 1:
                dup_records += len(obs) - 1
                if max(x[0] for x in obs) - min(x[0] for x in obs) > 0.1:
                    conflicting += 1
    result = {
        "scope": ("Full November 2026 BindingDB export" if USE_FULL else "BindingDB Articles subset only; NOT full BindingDB"),
        "status": "PRELIMINARY: exact lexical Kd, single chain and canonical amino acids, mapped assay keys; NO RDKit or independent construct/assay QC",
        "input_archive_sha256": sha256(DATA / (DATA_STEM + ".zip")),
        "assay_map_sha256": sha256(DATA / "BindingDB_rsid_eaids_202611_tsv.zip"),
        "counts": dict(stats),
        "construct_assay_groups_preliminary": len(tasks),
        "unique_construct_sequences_preliminary": len(set(seq for seq, _ in tasks)),
        "groups_ge_15_raw_smiles": len(viable15),
        "groups_ge_30_raw_smiles": len(viable30),
        "constructs_with_ge15_group": len(set(seq for seq, _, _ in viable15)),
        "constructs_with_ge30_group": len(set(seq for seq, _, _ in viable30)),
        "duplicate_group_smiles_records": dup_records,
        "duplicate_group_smiles_disagree_gt_0_1_pKd": conflicting,
        "largest_group_sizes": sorted(sizes, reverse=True)[:20],
        "top_20_groups": [
            {"seq_sha256": hashlib.sha256(seq.encode()).hexdigest(), "assay_key": assay, "raw_smiles_count": len(compounds)}
            for seq, assay, compounds in sorted(viable15, key=lambda x: len(x[2]), reverse=True)[:20]
        ],
        "limitations": [
            "A numeric Kd field is only a lexical exact candidate: original publication assay semantics are not verified.",
            "Raw SMILES identity is not stereochemistry-aware RDKit standardization and can overcount unique compounds.",
            "Matching target sequence strings does not establish biological construct equivalence.",
            "Assay linkage identifies an ENTRYID_ASSAYID identifier, not necessarily sufficient experimental-condition compatibility.",
            "Unknown assay fields, duplicates, source overlap, pH/temperature and all required leakage-safe splits remain to be audited.",
            "No train/test split, affinity training or performance comparison is made from this audit.",
        ],
    }
    (OUT / (OUTPUT_STEM + "_eligibility.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("top_20_groups", "limitations")}, indent=2), flush=True)

if __name__ == "__main__":
    main()
