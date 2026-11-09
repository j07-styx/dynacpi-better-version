"""Ensure training-protein-only model selection never reads validation labels."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_only_compound_shrinkage_dev import COHORT, fit, predict, run
import json
rows=json.loads((COHORT/'rows.json').read_text())
r=run(rows)
assert r['selected_alpha']==3.0
assert len(r['train_cv'])==8
assert len(r['train_cv'][0]['folds'])==9
assert r['validation']['rows']==200
assert abs(r['validation']['unshrunk_macro_rmse']-0.8365181844053747)<1e-10
changed=copy.deepcopy(rows)
for row in changed:
    if row['split']=='val':row['y']=12345.0
r2=run(changed)
assert r2['selected_alpha']==r['selected_alpha']
assert r2['train_cv']==r['train_cv']
assert r2['validation']['selected_macro_rmse']!=r['validation']['selected_macro_rmse']
assert predict(fit([{'compound':'a','y':5},{'compound':'a','y':7}],0),{'compound':'a'})==6
print('PASS train-only CV selection invariant to validation label mutation')
