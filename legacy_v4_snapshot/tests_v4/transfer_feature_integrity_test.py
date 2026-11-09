"""Actual train/dev feature coverage, manifest identity and test exclusion."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.transfer_data import load_development
rows,seq,split_hash=load_development()
manifest=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
assert {r["component"] for r in rows}.isdisjoint(manifest["test_components"])
assert {r["protein"] for r in rows if r["split"]=="train"}.isdisjoint({r["protein"] for r in rows if r["split"]=="dev"})
cache=ROOT/"data/transfer_esm64_dev"
for pid,s in seq.items():
    o=torch.load(cache/(pid+".pt"),weights_only=True)
    assert o["sequence_hash"]==pid and o["length"]==len(s)
    assert o["tokens"].shape==(min(64,len(s)),1280) and torch.isfinite(o["tokens"]).all()
    assert o["source_meta"]["checkpoint_sha256"]=="ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c"
g=torch.load(ROOT/"data/transfer_compounds_dev.pt",weights_only=True)
assert g["meta"]["split_sha256"]==split_hash
assert set(g["graphs"])=={r["compound"] for r in rows}
train_ids=sorted({r["compound"] for r in rows if r["split"]=="train"})
props=torch.stack([g["graphs"][c]["properties"] for c in train_ids])
assert torch.equal(props.mean(0),g["property_mean"])
assert torch.equal(props.std(0,unbiased=False).clamp_min(.1),g["property_std"])
print("PASS full feature coverage, source weight identity, train-only scaling, prospective-test exclusion")
