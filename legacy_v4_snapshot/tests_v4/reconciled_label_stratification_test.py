"""Regression test for read-only source-reconciled label stratification."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_reconciled_baseline_dev import audit

r=audit()
assert r['train_rows']==450 and r['validation_rows']==200
assert r['train_targets']==9 and r['validation_targets']==4
assert r['validation_numeric_kd_count']==71
assert r['validation_screen_negative_count']==129
assert abs(r['validation_target_macro_rmse']-0.8365181844053746)<1e-10
assert r['validation_numeric_kd_pooled_rmse']>r['validation_screen_negative_pooled_rmse']
assert sum(t['numeric_kd_count'] for t in r['per_target'].values())==71
assert sum(t['screen_negative_count'] for t in r['per_target'].values())==129
print('PASS reconciled baseline: train-only compound means and label-type stratification')
