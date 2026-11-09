"""No-GPU integrity tests for new DynaCPI v4 geometry training protocol."""
import hashlib
import sys
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from train_v4_geometry_trainonly_20261109 import (
    inner_partition, permute_residue_map, OUTPUT, SEEDS, MODES, sha_file, MIN_INNER_GAIN
)

fake=[hashlib.sha256(("protein-%d"%i).encode()).hexdigest() for i in range(9)]
fit,inner=inner_partition(fake)
assert len(fit)==7 and len(inner)==2
assert set(fit).isdisjoint(set(inner)) and set(fit)|set(inner)==set(fake)
assert inner_partition(list(reversed(fake)))==(fit,inner)
print("PASS deterministic label-blind train-only 7/2 partition",flush=True)

graph={"indices":torch.arange(17),"node_s":torch.ones(17,14),
       "edges":torch.tensor([[0,1],[1,2]]).T}
g1=permute_residue_map(graph,fake[0])
g2=permute_residue_map(graph,fake[0])
assert torch.equal(g1["indices"],g2["indices"])
assert not torch.equal(graph["indices"],g1["indices"])
assert set(g1["indices"].tolist())==set(graph["indices"].tolist())
assert torch.equal(graph["indices"],torch.arange(17))
assert g1["node_s"] is graph["node_s"] and g1["edges"] is graph["edges"]
print("PASS stable mismatched mapping without altering graph/source",flush=True)

assert 0<MIN_INNER_GAIN<0.01
summary=json.loads((OUTPUT/"SUMMARY.json").read_text(encoding="utf-8"))
assert summary["status"]=="COMPLETE_DEVELOPMENT_ONLY"
assert len(summary["arms"])==len(SEEDS)*len(MODES)==6
seen={(x["seed"],x["arm"]) for x in summary["arms"]}
assert seen=={(seed,mode) for seed in SEEDS for mode in MODES}
for record in summary["arms"]:
    assert 0<=record["epoch"]<=6
    p=OUTPUT/(record["arm"]+"_seed"+str(record["seed"]))
    result=json.loads((p/"COMPLETE.json").read_text(encoding="utf-8"))
    assert result["status"]=="COMPLETE_DEVELOPMENT_ONLY_NOT_SCIENTIFIC_VALIDATION"
    assert result["fit_pair_count"]==378 and result["inner_pair_count"]==108 and result["dev_pair_count"]==216
    assert result["input_hashes"]==summary["data_hashes"]
    assert result["checkpoint_sha256"]==sha_file(p/"best_checkpoint.pt")
    assert abs(result["historically_inspected_dev_diagnostic"]["macro_geometry_rmse"]-record["dev_rmse"])<1e-10
print("PASS six saved checkpoints, identities, hashes and metrics",flush=True)
print("3/3 validation groups PASS — engineering only, NOT scientific generalization",flush=True)
