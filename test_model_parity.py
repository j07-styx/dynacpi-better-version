"""Check independent model-class port against the original causal repair source."""
from pathlib import Path
import sys, json
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
OLD=Path(r"C:\Users\J0124\dynacpi_v4")
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(OLD/"scripts"))
sys.path.insert(0,str(HERE/"causal_diagnosis_20261109"))
from dynacpi.sequence_model import SequenceFoundation
from causal_diagnosis import Repair, prediction
from repaired_model import RepairedSequenceFoundation
from gpu_lease import gpu_lease

with gpu_lease(OLD):
    ck=torch.load(HERE/"checkpoints/seed_2833/checkpoint.pt",map_location="cpu",weights_only=False)
    assert ck["kind"]=="repaired_sequence_v4_2"
    original=SequenceFoundation().cuda()
    comparison=Repair(original.state_dict(),encoder=True,readout=True).cuda().eval()
    comparison.load_state_dict(ck["model_state"],strict=True)
    standalone=RepairedSequenceFoundation().cuda().eval()
    standalone.load_state_dict(ck["model_state"],strict=True)
    smiles=json.loads((OLD/"data/staging/compounds.json").read_text(encoding="utf-8"))
    pack=np.load(OLD/"data/staging/morgan_radius2_2048_chiral.npz")
    fps=dict(zip(pack["ids"].astype(str),pack["bits"]))
    ids=list(smiles)[:2]
    s=[smiles[ids[0]],smiles[ids[1]]]
    b=torch.as_tensor(np.stack([fps[i] for i in ids]),device="cuda",dtype=torch.float32)
    rows=json.loads((OLD/"data/engineering_pilot/rows.json").read_text())
    pid=next(r["sequence"] for r in rows if r["split"]=="val")
    obj=torch.load(OLD/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(pid+".pt"),
                   map_location="cpu",weights_only=True)
    e=obj["embeddings"].float().cuda()
    with torch.no_grad():
        ca=standalone.compound(s,b)
        cb=comparison.compound(s,b)
        assert torch.allclose(ca,cb,atol=1e-7,rtol=0)
        ya=prediction(standalone,ca,e)
        yb=prediction(comparison,cb,e)
        assert torch.allclose(ya,yb,atol=1e-7,rtol=0)
        assert (ya[0]-ya[1]).abs().item()>1e-7
    print("PASS: repaired model port exactly reproduces supplied experimental Repair class")
    print("PASS: 2 chemically distinct compounds produce different predictions",flush=True)
