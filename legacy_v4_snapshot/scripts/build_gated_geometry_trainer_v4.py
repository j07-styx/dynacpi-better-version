"""Create isolated learnable-gate DAVIS engineering trainer from verified seed-sweep baseline."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
src=(root/"scripts/train_geometry_seed_sweep_dev.py").read_text(encoding="utf-8")
edits=[
 ('from dynacpi.geometry_branch import GeometryBranch','from dynacpi.geometry_branch import GeometryBranch\nfrom dynacpi.confidence_gated_geometry import ConfidenceGatedGeometry'),
 ('if seed not in (1106,1107): raise ValueError("Only predeclared auxiliary DEV seeds 1106,1107 are allowed in this script")','if seed not in (1108,): raise ValueError("Only predeclared gated DEV seed 1108 allowed")'),
 ('geom=GeometryBranch().cuda().train()','geom=ConfidenceGatedGeometry(GeometryBranch()).cuda().train()'),
 ('result=geom.enhance(tokens,input_cache[p]["graph"])','result=geom(tokens,input_cache[p]["graph"])'),
 ('directory=root/"research/pilot_geometry_seed_sweep_dev"/("seed_"+str(seed))','directory=root/"research/pilot_gated_geometry_dev"/("seed_"+str(seed))'),
 ('"state":geom.state_dict(),"epoch":epoch,','"state":geom.state_dict(),"architecture":"confidence_gated_geometry_v1","epoch":epoch,'),
 ('"status":"ENGINEERING_LEGACY_DAVIS_NOT_SCIENCE_BENCHMARK"','"status":"ENGINEERING_GATED_GEOMETRY_DAVIS_NOT_SCIENCE_BENCHMARK"')
]
for a,b in edits:
 assert src.count(a)==1,(a,src.count(a))
 src=src.replace(a,b)
dest=root/"scripts/train_confidence_gated_geometry_dev.py"
dest.write_text(src,encoding="utf-8")
compile(src,str(dest),"exec")
print("CREATED isolated gated geometry training script",dest)
