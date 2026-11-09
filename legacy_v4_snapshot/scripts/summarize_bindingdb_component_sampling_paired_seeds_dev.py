import json,statistics
from pathlib import Path
r=Path('research')
rows=[]
for seed in (2103,2104):
    for mode,stem in [('uniform','bindingdb_deepdta_frozen_inner_dev_normalized_10epoch_seed'),('component_rebalanced','bindingdb_deepdta_component_weighted_10epoch_seed')]:
        obj=json.loads((r/(stem+str(seed)+'.json')).read_text())
        hist=obj['history'];best=min(hist,key=lambda h:h['val_rmse'])
        rows.append({'seed':seed,'mode':mode,'best_epoch':best['epoch'],'best_dev_rmse':best['val_rmse'],'epoch10_rmse':hist[-1]['val_rmse']})
by={mode:[x for x in rows if x['mode']==mode] for mode in ('uniform','component_rebalanced')}
result={'status':'EXPLORATORY_INNER_DEV_ONLY_NO_TEST','matched_seed_results':rows,'mean_best_dev_rmse':{k:statistics.mean(x['best_dev_rmse'] for x in v) for k,v in by.items()},'mean_epoch10_rmse':{k:statistics.mean(x['epoch10_rmse'] for x in v) for k,v in by.items()},'warnings':['Two seeds only, no uncertainty inference','Best epoch chosen by repeated inner-dev inspection','Component resampling also changes minibatch stochasticity','Train-only mean RMSE 1.803902 remains better than all four trained models','Provisional assay and construct QC; test untouched']}
(r/'bindingdb_deepdta_component_sampling_paired_seeds_dev.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
