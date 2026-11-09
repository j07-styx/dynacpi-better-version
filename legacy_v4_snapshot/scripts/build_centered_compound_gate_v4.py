"""Build opt-in centered structural residual experiment from anchored baseline."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'scripts/train_compound_conditioned_geometry_anchored_dev.py').read_text(encoding='utf-8')
changes={
 'from dynacpi.geometry_branch import GeometryBranch':'from dynacpi.geometry_branch import GeometryBranch\nfrom dynacpi.centered_geometry import CenteredGeometry',
 'if seed not in (1114,)':'if seed not in (1117,)',
 'geom=CompoundConditionedGeometry(GeometryBranch(),gate_bias=-3.5)':'geom=CompoundConditionedGeometry(CenteredGeometry(GeometryBranch()),gate_bias=-3.5)',
 'pilot_compound_conditioned_geometry_anchored_dev':'pilot_compound_conditioned_geometry_centered_dev',
 'compound_conditioned_geometry_anchored_v1':'compound_conditioned_geometry_centered_v1',
 'ENGINEERING_ANCHORED_COMPOUND_GATE_DAVIS_NOT_SCIENCE_BENCHMARK':'ENGINEERING_CENTERED_COMPOUND_GATE_DAVIS_NOT_SCIENCE_BENCHMARK',
}
for old,new in changes.items():
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
out=root/'scripts/train_compound_conditioned_geometry_centered_dev.py'
assert not out.exists(),str(out)
compile(s,str(out),'exec')
out.write_text(s,encoding='utf-8')
print('CREATED',out)
