"""Read-only reconciled cohort baseline and label-type stratification audit.

No model training, source mutation, validation-based parameter selection or test claims.
"""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/engineering_pilot_primary_reconciled_20261107'


def rmse(rows,predictions):
    if not rows:
        return None
    return math.sqrt(sum((float(r['y'])-float(predictions[(r['sequence'],r['compound'])]))**2 for r in rows)/len(rows))


def audit():
    path=SOURCE/'rows.json'
    rows=json.loads(path.read_text(encoding='utf-8'))
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['new_rows_sha256']
    assert len(rows)==650
    train=[r for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='val']
    train_targets={r['sequence'] for r in train}
    val_targets={r['sequence'] for r in val}
    assert not train_targets&val_targets
    grouped=collections.defaultdict(list)
    for r in train:
        grouped[r['compound']].append(float(r['y']))
    means={c:sum(ys)/len(ys) for c,ys in grouped.items()}
    assert all(r['compound'] in means for r in val)
    preds={(r['sequence'],r['compound']):means[r['compound']] for r in val}
    by_target={}
    for target in sorted(val_targets):
        subset=[r for r in val if r['sequence']==target]
        by_target[target[:12]]={
            'n':len(subset),
            'all_rmse':rmse(subset,preds),
            'numeric_kd_count':sum(r['primary_label_kind']=='numeric_Kd' for r in subset),
            'numeric_kd_rmse':rmse([r for r in subset if r['primary_label_kind']=='numeric_Kd'],preds),
            'screen_negative_count':sum(r['primary_label_kind']=='screen_negative_or_weak_binding' for r in subset),
            'screen_negative_rmse':rmse([r for r in subset if r['primary_label_kind']=='screen_negative_or_weak_binding'],preds),
        }
    numeric=[r for r in val if r['primary_label_kind']=='numeric_Kd']
    screen=[r for r in val if r['primary_label_kind']=='screen_negative_or_weak_binding']
    report={
        'status':'RECONCILED_DEVELOPMENT_LABEL_STRATIFICATION_NOT_SCIENTIFIC_BENCHMARK',
        'source_rows_sha256':manifest['new_rows_sha256'],
        'train_rows':len(train),'validation_rows':len(val),
        'train_targets':len(train_targets),'validation_targets':len(val_targets),
        'baseline':'compound mean from training targets only',
        'validation_all_pooled_rmse':rmse(val,preds),
        'validation_target_macro_rmse':sum(r['all_rmse'] for r in by_target.values())/len(by_target),
        'validation_numeric_kd_count':len(numeric),
        'validation_numeric_kd_pooled_rmse':rmse(numeric,preds),
        'validation_screen_negative_count':len(screen),
        'validation_screen_negative_pooled_rmse':rmse(screen,preds),
        'per_target':by_target,
        'warnings':['Screen-negative/weak-binding rows are not exact quantitative affinity observations.',
                    'Subset is selected and existing validation targets have been repeatedly used.',
                    'This audit does not validate chemical structures or experimental constructs.',
                    'Stratified numeric RMSE is descriptive, not a replacement for a censor-aware evaluation.']
    }
    assert len(train)==450 and len(val)==200
    assert len(numeric)+len(screen)==len(val)
    return report


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
