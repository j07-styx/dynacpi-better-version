"""Generate independent compound-conditioned DAVIS development trainer."""
from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/"scripts/train_confidence_gated_geometry_extended_dev.py").read_text()
changes=[
("from dynacpi.confidence_gated_geometry import ConfidenceGatedGeometry","from dynacpi.compound_conditioned_geometry import CompoundConditionedGeometry"),
('if seed not in (1110,): raise ValueError("Only extended gated DEV seed 1110 allowed")','if seed not in (1112,): raise ValueError("Only compound-conditioned DEV seed 1112 allowed")'),
("geom=ConfidenceGatedGeometry(GeometryBranch()).cuda().train()","geom=CompoundConditionedGeometry(GeometryBranch()).cuda().train()"),
('optimizer=torch.optim.AdamW(geom.parameters(),lr=1e-4,weight_decay=1e-3)','optimizer=torch.optim.AdamW(geom.parameters(),lr=1e-4,weight_decay=1e-3)'),
('result=geom(tokens,input_cache[p]["graph"])','result=geom(tokens,input_cache[p]["graph"],c)'),
('q=result.unsqueeze(0).expand(batch,-1,-1)','q=result'),
('directory=root/"research/pilot_gated_geometry_extended_dev"/("seed_"+str(seed))','directory=root/"research/pilot_compound_conditioned_geometry_dev"/("seed_"+str(seed))'),
('"architecture":"confidence_gated_geometry_extended_v1"','"architecture":"compound_conditioned_geometry_v1"'),
('"Only two epochs and one seed; insufficient to infer benefit of geometry or publish any result."','"Single-seed small-cohort development pilot; insufficient to infer benefit of geometry or publish any result."'),
]
for a,b in changes:
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
out=r/"scripts/train_compound_conditioned_geometry_dev.py"
compile(s,str(out),"exec")
out.write_text(s)
print("READY",out)
