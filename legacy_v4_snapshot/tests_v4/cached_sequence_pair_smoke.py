"""Real staged compound + verified cached ESM integration check.
No training on affinity measurements or scientific performance claim.
"""
from pathlib import Path
from itertools import islice
import csv
import json
import sys
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/"scripts"))
from dynacpi.sequence_model import SequenceFoundation
from gpu_lease import gpu_lease

def main():
    with (root/"data/staging/legacy_observations.csv").open(newline="",encoding="utf-8") as f:
        rows=list(islice(csv.DictReader(f),2))
    compounds=json.loads((root/"data/staging/compounds.json").read_text(encoding="utf-8"))
    sequences=json.loads((root/"data/staging/sequences.json").read_text(encoding="utf-8"))
    cache=root/"data/esm2_t33_650M_layer33_w1000_o128_fp16"
    settings=json.loads((cache/"settings.json").read_text(encoding="utf-8"))
    smiles=[compounds[r["compound_hash"]] for r in rows]
    loaded=[torch.load(cache/(r["sequence_hash"]+".pt"),map_location="cpu",weights_only=True) for r in rows]
    lengths=[len(sequences[r["sequence_hash"]]) for r in rows]
    assert all(x["embeddings"].shape==(n,1280) and x["meta"]==settings for x,n in zip(loaded,lengths))
    t=torch.nn.utils.rnn.pad_sequence([x["embeddings"].float() for x in loaded],batch_first=True).cuda()
    valid=(torch.arange(t.size(1),device="cuda")[None,:]<torch.tensor(lengths,device="cuda")[:,None])
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    mols=[Chem.MolFromSmiles(s) for s in smiles]
    assert all(m is not None for m in mols)
    fp=torch.tensor(np.stack([gen.GetFingerprintAsNumPy(m) for m in mols]),dtype=torch.float32,device="cuda")
    torch.manual_seed(1105)
    model=SequenceFoundation().cuda().eval()
    with torch.no_grad():
        batch=model(smiles,fp,t,valid)
        singles=torch.stack([model([smiles[i]],fp[i:i+1],t[i:i+1,:lengths[i]],valid[i:i+1,:lengths[i]])[0] for i in range(len(rows))])
        diff=float((batch-singles).abs().max())
        torch.testing.assert_close(batch,singles,rtol=1e-5,atol=1e-5)
    model.train()
    model.zero_grad(set_to_none=True)
    loss=(model(smiles,fp,t,valid)-torch.tensor([0.3,-0.2],device="cuda")).square().mean()
    loss.backward()
    grad=model.sequence[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
    result={"scope":"Two real staged DAVIS compounds and sequences with genuine cached frozen ESM-2 650M residue embeddings; random initialized extension head and synthetic loss targets",
            "status":"engineering_smoke_only", "pairs":len(rows),
            "sequence_lengths":lengths,
            "max_batch_vs_single_difference":diff,
            "sequence_projection_grad_norm":float(grad.norm()),
            "device":torch.cuda.get_device_name(),
            "note":"Input legacy labels quarantined; no affinity learning, validated target semantics, trained checkpoint or scientific result"}
    dest=root/"research/cached_sequence_pair_smoke.json"
    dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":
    with gpu_lease(root):
        main()
