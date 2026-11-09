"""Read-only compare fresh mixed-label and numeric-only sequence training runs."""
import json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for name in ('reconciled_clean_sequence_seed_2101_dev','reconciled_numeric_only_sequence_seed_2102_dev'):
    p=root/'research'/name
    history=json.loads((p/'history.json').read_text())
    rows=json.loads((p/'val_predictions.json').read_text())
    assert len(history)==4 and len(rows)==200
    print(name,'best_dev_macro',min(x['target_macro_rmse'] for x in history),'last_epoch',history[-1]['epoch'])
    for kind in ('numeric_Kd','screen_negative_or_weak_binding'):
        a=[r for r in rows if r['primary_label_kind']==kind]
        print(' ',kind,'n',len(a),'last_epoch_pooled_rmse',math.sqrt(sum((r['pred']-r['y'])**2 for r in a)/len(a)))
