"""Read-only DAVIS development audit. Never promotes source approval or test status."""
import argparse, collections, hashlib, json, math
from pathlib import Path
from Bio.Align import PairwiseAligner
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def overlap(rows, key):
    groups = {s: {key(r) for r in rows if r["split"] == s} for s in ("train", "val")}
    return {"train": len(groups["train"]), "val": len(groups["val"]),
            "intersection": len(groups["train"] & groups["val"])}
def audit(root):
    pilot = root / "data/engineering_pilot"
    rows = json.loads((pilot/"rows.json").read_text())
    seq = json.loads((root/"data/staging/sequences.json").read_text())
    compounds = json.loads((root/"data/staging/compounds.json").read_text())
    manifest = json.loads((pilot/"limited_alpha_3d_graph_manifest.json").read_text())
    records = [r for r in manifest["records"] if r["status"] == "GRAPH_MAPPED_ENGINEERING_ONLY"]
    ids = {r["sequence_sha256"] for r in records}
    mapped = [r for r in rows if r["sequence"] in ids]
    canonical, scaffolds = {}, {}
    for c in {r["compound"] for r in rows}:
        mol = Chem.MolFromSmiles(compounds[c])
        if mol is None: raise ValueError("Invalid molecule " + c)
        canonical[c] = Chem.MolToSmiles(mol, isomericSmiles=True)
        scaffolds[c] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    result = {"status": "DEVELOPMENT_ONLY", "inputs": {
        str(p.relative_to(root)): digest(p) for p in (pilot/"rows.json", pilot/"limited_alpha_3d_graph_manifest.json",
        root/"data/staging/sequences.json", root/"data/staging/compounds.json")}}
    for name, subset in (("full_pilot", rows), ("mapped_pilot", mapped)):
        labels = {}
        for split in ("train", "val"):
            y = [float(r["y"]) for r in subset if r["split"] == split]
            if not all(math.isfinite(v) for v in y): raise ValueError("Nonfinite labels")
            labels[split] = {"n": len(y), "minimum": min(y), "maximum": max(y),
                            "at_numeric_5": sum(v == 5 for v in y)}
        result[name] = {"sequences": overlap(subset, lambda r: seq[r["sequence"]]),
            "compounds": overlap(subset, lambda r: canonical[r["compound"]]),
            "scaffolds_including_empty": overlap(subset, lambda r: scaffolds[r["compound"]]),
            "pairs": overlap(subset, lambda r: (seq[r["sequence"]], canonical[r["compound"]])),
            "labels": labels}
    aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                              open_gap_score=-10, extend_gap_score=-0.5)
    train = sorted({r["sequence"] for r in rows if r["split"] == "train"})
    valid = sorted({r["sequence"] for r in rows if r["split"] == "val"})
    similarities = []
    for a in train:
        for b in valid:
            aln = aligner.align(seq[a], seq[b])[0]
            paired = matches = 0
            for (s, e), (u, v) in zip(*aln.aligned):
                paired += int(e-s)
                matches += sum(x == y for x, y in zip(seq[a][s:e], seq[b][u:v]))
            similarities.append({"train": a, "val": b, "identity_paired": matches/max(1, paired),
                "coverage_train": paired/len(seq[a]), "coverage_val": paired/len(seq[b]),
                "both_geometry_mapped": a in ids and b in ids})
    result["cross_split_global_alignments"] = similarities
    result["homology_screen"] = {"rule": "paired identity >=0.3 and coverage >=0.8 for BOTH sequences; screening only, not a family assignment",
        "flagged": [x for x in similarities if x["identity_paired"] >= .3 and min(x["coverage_train"], x["coverage_val"]) >= .8]}
    result["structure_provenance"] = {"mapped": len(records), "experimentally_verified": 0,
        "coverage_min": min(r["backbone_coverage"] for r in records),
        "note": "Mapping manifest asserts sequence substring matching; no independent construct approval is inferred."}
    result["label_policy"] = "Retain original numeric labels for legacy parity only. Values at 5 are counted, not independently adjudicated as exact or censored. No relabeling."
    result["decision"] = "No independent holdout exists in this pilot. Shared compounds permitted for target transfer but preclude compound-disjoint claims. Alignment screening is not a verified family split."
    return result
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = audit(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f: json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cross_split_global_alignments", "inputs")}, indent=2))
