"""Build isolated conservative geometry gate engineering experiment; never overwrite earlier pilots."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
src=root/'scripts/train_compound_conditioned_geometry_anchored_dev.py'
out=root/'scripts/train_compound_conditioned_geometry_strong_anchor_dev.py'
s=src.read_text(encoding='utf-8')
changes={
 'if seed not in (1114,)':'if seed not in (1115,)',
 'gate_bias=-3.5':'gate_bias=-4.5',
 'lr=5e-5,weight_decay=3e-3':'lr=3e-5,weight_decay=5e-3',
 'loss=fit_loss+0.25*anchor_loss':'loss=fit_loss+0.5*anchor_loss',
 'pilot_compound_conditioned_geometry_anchored_dev':'pilot_compound_conditioned_geometry_strong_anchor_dev',
 'compound_conditioned_geometry_anchored_v1':'compound_conditioned_geometry_strong_anchor_v1',
 'ENGINEERING_ANCHORED_COMPOUND_GATE_DAVIS_NOT_SCIENCE_BENCHMARK':'ENGINEERING_STRONG_ANCHOR_COMPOUND_GATE_DAVIS_NOT_SCIENCE_BENCHMARK',
}
for old,new in changes.items():
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
assert not out.exists(),f'Experiment script already exists: {out}'
compile(s,str(out),'exec')
out.write_text(s,encoding='utf-8')
print('CREATED',out)
