"""Verify every staged ESM-2 feature file before downstream model training."""
from pathlib import Path
import hashlib
import json
import math
import torch

root = Path(__file__).resolve().parents[1]
seqs = json.loads((root / "data/staging/sequences.json").read_text(encoding="utf-8"))
cache = root / "data/esm2_t33_650M_layer33_w1000_o128_fp16"
meta = json.loads((cache / "settings.json").read_text(encoding="utf-8"))
counts = {"sequences": len(seqs), "present": 0, "valid": 0, "missing": 0,
          "bad": 0, "total_residues": 0, "total_windows": 0}
errors = []
for seqhash, sequence in sorted(seqs.items()):
    path = cache / (seqhash + ".pt")
    if not path.exists():
        counts["missing"] += 1
        errors.append([seqhash, "file_missing"])
        continue
    counts["present"] += 1
    try:
        if hashlib.sha256(sequence.encode()).hexdigest() != seqhash:
            raise ValueError("staging sequence hash mismatch")
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["sequence_hash"] != seqhash:
            raise ValueError("cached sequence hash mismatch")
        if blob["meta"] != meta:
            raise ValueError("feature settings mismatch")
        embs = blob["embeddings"]
        if embs.shape != (len(sequence), 1280):
            raise ValueError("wrong embedding shape " + str(embs.shape))
        if embs.dtype != torch.float16 or not torch.isfinite(embs).all():
            raise ValueError("wrong dtype or nonfinite embedding")
        covers = [0] * len(sequence)
        windows = blob["windows"]
        if not windows:
            raise ValueError("no windows")
        for start, end in windows:
            if not 0 <= start < end <= len(sequence) or end - start > 1000:
                raise ValueError("invalid window boundaries")
            for pos in range(start, end):
                covers[pos] += 1
        if min(covers) < 1:
            raise ValueError("gaps in window coverage")
        counts["valid"] += 1
        counts["total_residues"] += len(sequence)
        counts["total_windows"] += len(windows)
    except Exception as exc:
        counts["bad"] += 1
        errors.append([seqhash, str(exc)])
result = {"scope": "staged sequences ESM-2 cache, real extractor files",
          "model": meta, "counts": counts, "errors": errors[:25],
          "note": "Validates shapes, masks/windows, metadata and finite values; does not validate biological predictive accuracy"}
dest = root / "data/audit/esm_cache_integrity.json"
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"counts": counts, "errors": errors[:10]}, indent=2), flush=True)
if counts["missing"] or counts["bad"]:
    raise SystemExit(1)
