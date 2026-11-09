"""Fixed internal target-holdout comparison for the candidate chemistry repair.

No pretrained supervised checkpoint is used: original and repaired models start
from identical random shared weights. Only historical TRAIN rows are read for
fitting, model selection, and scoring. This is engineering validation, not a new
independent scientific test. No production model/API/checkpoint is replaced.
"""
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from causal_diagnosis import ROOT, OUT, Repair, SequenceFoundation, prediction, metric, save, gpu_lease


def main():
    if (OUT/'holdout_results.json').exists():
        raise RuntimeError('Refusing to overwrite completed experiment')
    start=time.monotonic()
    torch.set_num_threads(2)
    rows=[r for r in json.loads((ROOT/'data/engineering_pilot/rows.json').read_text()) if r['split']=='train']
    pids=sorted({r['sequence'] for r in rows},key=lambda p:hashlib.sha256(('causal_repair_inner_v1|'+p).encode()).hexdigest())
    assert len(pids)==24
    fit,inner=pids[6:],pids[:6]
    cids=sorted({r['compound'] for r in rows})
    assert len(cids)==54
    target={}
    embeds={}
    for p in pids:
        d={r['compound']:r['y'] for r in rows if r['sequence']==p}
        assert set(d)==set(cids)
        target[p]=np.array([d[c] for c in cids],dtype=np.float64)
        obj=torch.load(ROOT/'data/esm2_t33_650M_layer33_w1000_o128_fp16'/(p+'.pt'),map_location='cpu',weights_only=True)
        assert obj['sequence_hash']==p
        embeds[p]=obj['embeddings'].float().cuda()
    yfit=np.stack([target[p] for p in fit])
    mean,std=float(yfit.mean()),float(yfit.std())
    y={p:torch.tensor((target[p]-mean)/std,device='cuda',dtype=torch.float32) for p in pids}
    stage=ROOT/'data/staging'
    smiles=json.loads((stage/'compounds.json').read_text())
    sm=[smiles[c] for c in cids]
    z=np.load(stage/'morgan_radius2_2048_chiral.npz')
    fps=dict(zip(z['ids'].astype(str),z['bits']))
    bits=torch.tensor(np.stack([fps[c] for c in cids]),device='cuda',dtype=torch.float32)
    lookup=yfit.mean(0)
    baselines={p:{'compound_mean':metric(lookup,target[p]),
                  'global_mean':metric(np.full(54,mean),target[p])} for p in inner}
    protocol={'status':'INTERNAL_HISTORICAL_TRAIN_TARGET_HOLDOUT_NOT_INDEPENDENT_SCIENCE',
        'fit_proteins':fit,'inner_proteins':inner,'fit_pairs':18*54,'inner_pairs':6*54,
        'shared_compounds':54,'seeds':[2831,2832,2833],'epochs':20,
        'optimizer_steps_per_arm':360,'lr':0.0003,'weight_decay':0.0001,
        'batch':'54 compounds for one protein, random target order each epoch',
        'dropout':'enabled during training, original inherited rates; new pair MLP has no dropout',
        'selection':'lowest six-target inner macro RMSE over epochs 0,5,10,15,20',
        'labels':'standardized only using the 18 fit targets',
        'initialization':'matched fresh random shared weights; ESM embeddings only pretrained features',
        'limitations':['Not capacity matched; repaired model adds direct chemical and pair feature paths',
                       'All 24 historical training targets were used in earlier engineering; no fresh final test',
                       'All 54 compounds occur in training; this is an unseen-target, known-compound check',
                       'Fixed one internal split; seeds are optimization repeats, not independent datasets'],
        'baseline_inner_macro_compound_mean':float(np.mean([v['compound_mean']['rmse'] for v in baselines.values()])),
        'baseline_inner_macro_global_mean':float(np.mean([v['global_mean']['rmse'] for v in baselines.values()])),
        'baseline_details':baselines,'arms':{}}
    save('holdout_protocol.json',protocol)
    def score(model,ids,shuffle=False):
        model.eval()
        vals={}
        with torch.no_grad():
            c=model.compound(sm,bits)
            if shuffle:
                # Change only compound-to-label association, keeping its marginal
                # feature distribution and the trained weights unchanged.
                c=c[torch.tensor(np.random.default_rng(7731).permutation(54),device='cuda')]
            for p in ids:
                pred=(prediction(model,c,embeds[p])*std+mean).cpu().numpy()
                vals[p]=metric(pred,target[p])
        return {'macro_rmse':float(np.mean([v['rmse'] for v in vals.values()])),
                'mean_pred_sd':float(np.mean([v['pred_sd'] for v in vals.values()])),
                'mean_centered_mse':float(np.mean([v['centered_mse'] for v in vals.values()])),
                'per_protein':vals}
    results=copy.deepcopy(protocol)
    for seed in protocol['seeds']:
        torch.manual_seed(seed)
        init=SequenceFoundation(device='cuda').cuda()
        state={k:v.detach().cpu().clone() for k,v in init.state_dict().items()}
        del init
        for name,repair in [('original',False),('both_bypasses',True)]:
            torch.manual_seed(seed)
            m=Repair(state,encoder=repair,readout=repair).cuda()
            opt=torch.optim.AdamW(m.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
            best=float('inf'); best_state=None; best_epoch=0;history=[]
            for epoch in range(21):
                if epoch:
                    m.train()
                    for idx in np.random.default_rng(seed*100+epoch).permutation(18):
                        p=fit[idx]
                        opt.zero_grad(set_to_none=True)
                        pred=prediction(m,m.compound(sm,bits),embeds[p])
                        loss=(pred-y[p]).square().mean()
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                        loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.)
                        opt.step()
                if epoch in [0,5,10,15,20]:
                    sc=score(m,inner)
                    tr=score(m,fit)
                    history.append({'epoch':epoch,'fit_macro_rmse':tr['macro_rmse'],
                                    'inner_macro_rmse':sc['macro_rmse'],
                                    'inner_mean_pred_sd':sc['mean_pred_sd']})
                    if sc['macro_rmse']<best:
                        best=sc['macro_rmse'];best_epoch=epoch
                        best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            m.load_state_dict(best_state,strict=True)
            item={'seed':seed,'arm':name,'selected_epoch':best_epoch,'history':history,
                  'parameters':sum(p.numel() for p in m.parameters()),
                  'selected_inner':score(m,inner),'selected_fit':score(m,fit),
                  'selected_inner_compound_shuffle':score(m,inner,shuffle=True)}
            results['arms'][name+'_seed'+str(seed)]=item
            save('holdout_progress.json',results)
            print(json.dumps({'seed':seed,'arm':name,'epoch':best_epoch,'inner_rmse':best,
                'fit_rmse':item['selected_fit']['macro_rmse'],
                'shuffle_inner_rmse':item['selected_inner_compound_shuffle']['macro_rmse'],
                'inner_prediction_sd':item['selected_inner']['mean_pred_sd']}),flush=True)
            del m,opt,best_state
            torch.cuda.empty_cache()
    results['elapsed_seconds']=time.monotonic()-start
    save('holdout_results.json',results)
    print(json.dumps({'status':'HOLDOUT_COMPLETE','elapsed_seconds':results['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    with gpu_lease(ROOT):
        main()
