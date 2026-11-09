"""Create a new regularized compound-conditioned geometry development experiment."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/"scripts/train_compound_conditioned_geometry_dev.py").read_text()
changes=[
('if seed not in (1112,): raise ValueError("Only compound-conditioned DEV seed 1112 allowed")','if seed not in (1114,): raise ValueError("Only anchored compound-conditioned DEV seed 1114 allowed")'),
('geom=CompoundConditionedGeometry(GeometryBranch()).cuda().train()','geom=CompoundConditionedGeometry(GeometryBranch(),gate_bias=-3.5).cuda().train()'),
('optimizer=torch.optim.AdamW(geom.parameters(),lr=1e-4,weight_decay=1e-3)','optimizer=torch.optim.AdamW(geom.parameters(),lr=5e-5,weight_decay=3e-3)'),
('            return output\n        with torch.no_grad():','            with torch.no_grad():\n                raw=base.predict_tokens(c,tokens.unsqueeze(0).expand(batch,-1,-1),mask)\n            return output,raw.detach()\n        with torch.no_grad():'),
('directory=root/"research/pilot_compound_conditioned_geometry_dev"/("seed_"+str(seed))','directory=root/"research/pilot_compound_conditioned_geometry_anchored_dev"/("seed_"+str(seed))'),
('            pred=predict_pair(rr,p,train=True)','            pred,baseline=predict_pair(rr,p,train=True)'),
('            loss=(pred-targets).square().mean()','            fit_loss=(pred-targets).square().mean()\n            anchor_loss=(pred-baseline).square().mean()\n            loss=fit_loss+0.25*anchor_loss'),
('"architecture":"compound_conditioned_geometry_v1"','"architecture":"compound_conditioned_geometry_anchored_v1"'),
('"status":"ENGINEERING_GATED_GEOMETRY_DAVIS_NOT_SCIENCE_BENCHMARK"','"status":"ENGINEERING_ANCHORED_COMPOUND_GATE_DAVIS_NOT_SCIENCE_BENCHMARK"')
]
for old,new in changes:
 assert s.count(old)==1,(old,s.count(old))
 s=s.replace(old,new)
out=root/"scripts/train_compound_conditioned_geometry_anchored_dev.py"
compile(s,str(out),"exec");out.write_text(s)
print("READY",out)
