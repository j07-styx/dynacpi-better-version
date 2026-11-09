import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
entries=[]
for seed in (2103,2104):
    for amp,stem in [(0.5,'bindingdb_bounded_residual_deepdta_10epoch_seed'),(1.5,'bindingdb_bounded_residual_amp15_10epoch_seed')]:
        obj=json.loads((root/'research'/f'{stem}{seed}.json').read_text())
        assert obj['train_pairs']==27118 and obj['validation_pairs']==7387 and len(obj['history'])==10
        best=min(obj['history'],key=lambda x:x['val_rmse'])
        entries.append({'seed':seed,'residual_amplitude':amp,'best_epoch':best['epoch'],'best_inner_dev_rmse':best['val_rmse'],'final_epoch_rmse':obj['history'][-1]['val_rmse']})
means={str(amp):{'best_rmse':statistics.mean(x['best_inner_dev_rmse'] for x in entries if x['residual_amplitude']==amp),'final_rmse':statistics.mean(x['final_epoch_rmse'] for x in entries if x['residual_amplitude']==amp)} for amp in (0.5,1.5)}
assert means['0.5']['best_rmse']<means['1.5']['best_rmse'] and means['0.5']['final_rmse']<means['1.5']['final_rmse']
result={'status':'EXPLORATORY_INNER_DEV_NO_TEST','matched_seed_results':entries,'mean_by_amplitude':means,'warnings':['Two seeds and repeatedly inspected development split; not an unbiased result','Neither configuration beats train-only mean RMSE 1.803902','No approved assay/construct QC or independent test evaluation']}
(root/'research/bindingdb_compact_cnn_matched_amplitude_ablation_dev.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
