"""Train-protein-only LOTO selection for regularized compound means.

This is an interpretable negative control, not a novel architecture or a test-set score.
No source files or checkpoints are modified. Validation is read only after
hyperparameter selection from the nine training proteins.
"""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COHORT=ROOT/'data/engineering_pilot_primary_reconciled_20261107'
ALPHAS=(0.,1.,3.,10.,30.,100.,300.,1000.)


def fit(rows,alpha):
    global_y=sum(float(r['y']) for r in rows)/len(rows)
    by=collections.defaultdict(list)
    for r in rows:by[r['compound']].append(float(r['y']))
    return global_y,{c:(sum(v)+alpha*global_y)/(len(v)+alpha) for c,v in by.items()}


def predict(model,row):
    default,by=model
    return by.get(row['compound'],default)


def mse(rows,model):
    return sum((float(r['y'])-predict(model,r))**2 for r in rows)/len(rows)


def score_by_label(rows,model):
    out={}
    for kind in ('numeric_Kd','screen_negative_or_weak_binding'):
        subset=[r for r in rows if r['primary_label_kind']==kind]
        out[kind]={'count':len(subset),'rmse':math.sqrt(mse(subset,model)) if subset else None}
    return out


def run(rows,alphas=ALPHAS):
    train=[r for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='val']
    assert train and val
    train_targets=sorted({r['sequence'] for r in train})
    val_targets=sorted({r['sequence'] for r in val})
    assert not set(train_targets)&set(val_targets)
    assert len({r['compound'] for r in train})==len({r['compound'] for r in val})
    cv=[]
    for alpha in alphas:
        folds=[]
        for held in train_targets:
            fold_train=[r for r in train if r['sequence']!=held]
            fold_val=[r for r in train if r['sequence']==held]
            model=fit(fold_train,alpha)
            folds.append({'target_sha256_prefix':held[:12], 'n':len(fold_val),'mse':mse(fold_val,model)})
        cv.append({'alpha':alpha,'train_target_macro_rmse':sum(math.sqrt(f['mse']) for f in folds)/len(folds),
                   'train_pooled_rmse':math.sqrt(sum(f['mse']*f['n'] for f in folds)/sum(f['n'] for f in folds)),
                   'folds':folds})
    selected=min(cv,key=lambda x:(x['train_target_macro_rmse'],x['alpha']))
    chosen=fit(train,selected['alpha'])
    unshrunk=fit(train,0.)
    return {'status':'DEVELOPMENT_ONLY_TRAIN_PROTEIN_LOTO_SELECTED',
            'selection':'9 training proteins leave-one-protein-out; macro RMSE, never validation-selected',
            'selected_alpha':selected['alpha'],'train_cv':cv,
            'validation':{'rows':len(val),'proteins':len(val_targets),
              'selected_pooled_rmse':math.sqrt(mse(val,chosen)),
              'unshrunk_pooled_rmse':math.sqrt(mse(val,unshrunk)),
              'selected_macro_rmse':sum(math.sqrt(mse([r for r in val if r['sequence']==t],chosen)) for t in val_targets)/len(val_targets),
              'unshrunk_macro_rmse':sum(math.sqrt(mse([r for r in val if r['sequence']==t],unshrunk)) for t in val_targets)/len(val_targets),
              'selected_by_label':score_by_label(val,chosen),
              'unshrunk_by_label':score_by_label(val,unshrunk)},
            'warnings':['Previously inspected development validation, not an independent holdout',
                        'Source screening-negative rows are not exact numeric affinities',
                        'Protein-family-disjointness and assayed constructs are unverified']}


if __name__=='__main__':
    path=COHORT/'rows.json'
    manifest=json.loads((COHORT/'manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['new_rows_sha256']
    rows=json.loads(path.read_text())
    print(json.dumps(run(rows),indent=2))
