"""Matched seed objective comparison on repeatedly inspected DEV validation.
Report all four epochs; no validation-based checkpoint selection.
"""
import json,math,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
results={}
for seed in (2103,2104,2105):
    results[seed]={}
    for mode in ('mixed','numeric'):
        path=root/'research'/f'reconciled_{mode}_sequence_seed_{seed}_dev'
        hist=json.loads((path/'history.json').read_text())
        rows=json.loads((path/'val_predictions.json').read_text())
        assert len(hist)==4 and len(rows)==200
        numeric=[r for r in rows if r['primary_label_kind']=='numeric_Kd']
        assert len(numeric)==71
        results[seed][mode]={'epoch4_numeric_rmse':math.sqrt(sum((r['pred']-r['y'])**2 for r in numeric)/len(numeric)),
                             'epoch4_macro_rmse':hist[-1]['target_macro_rmse'],
                             'history':hist}
for seed in results:
    print('seed',seed,'mixed',results[seed]['mixed']['epoch4_numeric_rmse'],'numeric',results[seed]['numeric']['epoch4_numeric_rmse'])
for mode in ('mixed','numeric'):
    scores=[results[s][mode]['epoch4_numeric_rmse'] for s in results]
    print('summary',mode,'numeric_kd_epoch4_rmse_mean',statistics.mean(scores),'std',statistics.stdev(scores))
assert all(results[s]['numeric']['epoch4_numeric_rmse']<results[s]['mixed']['epoch4_numeric_rmse'] for s in results)
