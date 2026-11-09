"""Build isolated 12-epoch gated geometry optimization experiment."""
from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/"scripts/train_confidence_gated_geometry_dev.py").read_text()
changes=[
 ('if seed not in (1108,): raise ValueError("Only predeclared gated DEV seed 1108 allowed")','if seed not in (1110,): raise ValueError("Only extended gated DEV seed 1110 allowed")'),
 ('optimizer=torch.optim.AdamW(geom.parameters(),lr=3e-4,weight_decay=1e-4)','optimizer=torch.optim.AdamW(geom.parameters(),lr=1e-4,weight_decay=1e-3)'),
 ('directory=root/"research/pilot_gated_geometry_dev"/("seed_"+str(seed))','directory=root/"research/pilot_gated_geometry_extended_dev"/("seed_"+str(seed))'),
 ('for epoch in range(1,3):','for epoch in range(1,13):'),
 ('history=[]\n    best=float("inf")','history=[]\n    best=float("inf")\n    stale=0'),
 ('if g_macro<best:\n            best=g_macro','if g_macro<best-1e-5:\n            best=g_macro\n            stale=0'),
 ('            os.replace(tmp,directory/"best_checkpoint.pt")','            os.replace(tmp,directory/"best_checkpoint.pt")\n        else:\n            stale+=1\n        if stale>=4:\n            print("EARLY_STOP: four epochs without macro RMSE improvement",flush=True)\n            break'),
 ('"architecture":"confidence_gated_geometry_v1"','"architecture":"confidence_gated_geometry_extended_v1"')
]
for a,b in changes:
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
out=r/"scripts/train_confidence_gated_geometry_extended_dev.py"
compile(s,str(out),"exec");out.write_text(s)
print("READY",out)
