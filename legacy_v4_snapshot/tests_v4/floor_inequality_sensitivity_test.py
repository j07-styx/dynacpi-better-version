"""Check hypothetical inequality penalty without claiming assay semantics."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'scripts/sensitivity_floor_inequality_dev.py'
spec=importlib.util.spec_from_file_location('floor_sensitivity',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert m.loss(5.0,4.0,'screen_negative_or_weak_binding')==0
assert m.loss(5.0,6.0,'screen_negative_or_weak_binding')==1
assert m.loss(6.0,5.0,'numeric_Kd')==1
assert len(m.val)==200
assert m.output['compound_average']['hypothetical_screen_upper_bound_violations']==87
assert m.output['archived_sequence']['hypothetical_screen_upper_bound_violations']==129
print('PASS hypothetical upper-bound sensitivity and explicit label semantics')
