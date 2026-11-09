"""Exploratory floor-inequality sensitivity, NOT validated assay censoring.

Treat a screen-negative/weak-binding pKd=5 benchmark code as hypothetical
upper bound, and report exact-Kd squared error separately. Read-only.
"""
import collections
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'data/engineering_pilot_primary_reconciled_20261107/rows.json').read_text())
val=[r for r in rows if r['split']=='val']
train=[r for r in rows if r['split']=='train']
archived=json.loads((ROOT/'research/bounded_fusion_study_20261107/baseline.json').read_text())['predictions']
seq={(r['sequence'],r['compound']):float(r['pred']) for r in archived}
compound=collections.defaultdict(list)
for r in train:compound[r['compound']].append(float(r['y']))
means={k:sum(v)/len(v) for k,v in compound.items()}

def loss(y,p,kind,floor=5.0):
    if kind=='numeric_Kd':return (p-y)**2
    if kind=='screen_negative_or_weak_binding':return max(0.0,p-floor)**2
    raise ValueError('unrecognized label kind')

def analyze(pred):
    exact=[r for r in val if r['primary_label_kind']=='numeric_Kd']
    screen=[r for r in val if r['primary_label_kind']=='screen_negative_or_weak_binding']
    get=lambda r:pred[(r['sequence'],r['compound'])]
    exact_rmse=math.sqrt(sum(loss(r['y'],get(r),r['primary_label_kind']) for r in exact)/len(exact))
    upper_violations=sum(get(r)>5.0 for r in screen)
    screen_hinge=math.sqrt(sum(loss(r['y'],get(r),r['primary_label_kind']) for r in screen)/len(screen))
    combined=math.sqrt(sum(loss(r['y'],get(r),r['primary_label_kind']) for r in val)/len(val))
    return {'numeric_rmse':exact_rmse,'hypothetical_screen_upper_bound_violations':upper_violations,
            'hypothetical_screen_upper_bound_hinge_rmse':screen_hinge,
            'hypothetical_combined_hinge_rmse':combined}

assert len(val)==200 and len(train)==450
assert all((r['sequence'],r['compound']) in seq for r in val)
assert all(r['compound'] in means for r in val)
output={'status':'EXPLORATORY_HYPOTHETICAL_FLOOR_SENSITIVITY_NOT_ASSAY_VALIDATED',
        'assumption':'Only for sensitivity: screening-negative/weak-binding code y=5 interpreted as pKd <= 5, NOT a verified censoring rule',
        'validation_rows':len(val),
        'compound_average':analyze({(r['sequence'],r['compound']):means[r['compound']] for r in val}),
        'archived_sequence':analyze(seq),
        'warnings':['No claim that all source blanks imply quantitative pKd <= 5.',
                    'Do not use this hypothetical score for model selection or a paper benchmark.',
                    'Previously used validation cohort; archived neural model trained on legacy records.']}
print(json.dumps(output,indent=2))
