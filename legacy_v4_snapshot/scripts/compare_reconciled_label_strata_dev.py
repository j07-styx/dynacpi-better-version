"""Compare archived sequence predictions with train-only compound means by source label kind."""
import collections
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'data/engineering_pilot_primary_reconciled_20261107/rows.json').read_text())
old=json.loads((ROOT/'research/bounded_fusion_study_20261107/baseline.json').read_text())['predictions']
oldmap={(r['sequence'],r['compound']):float(r['pred']) for r in old}
train=[r for r in rows if r['split']=='train']
val=[r for r in rows if r['split']=='val']
means=collections.defaultdict(list)
for r in train:means[r['compound']].append(float(r['y']))
means={k:sum(v)/len(v) for k,v in means.items()}
assert len(val)==200 and all((r['sequence'],r['compound']) in oldmap for r in val)

def score(group,method):
    if not group:return None
    return math.sqrt(sum((float(r['y'])-method(r))**2 for r in group)/len(group))

out={'status':'ARCHIVED_CHECKPOINT_FILTERED_DEVELOPMENT_DIAGNOSTIC_NOT_CLEAN_RETRAINING', 'rows':len(val),'groups':{}}
for name,group in [('all',val),('numeric_Kd',[r for r in val if r['primary_label_kind']=='numeric_Kd']),('screen_negative_or_weak_binding',[r for r in val if r['primary_label_kind']=='screen_negative_or_weak_binding'])]:
    out['groups'][name]={'n':len(group),'sequence_pooled_rmse':score(group,lambda r:oldmap[(r['sequence'],r['compound'])]),'compound_mean_pooled_rmse':score(group,lambda r:means[r['compound']])}
print(json.dumps(out,indent=2))
