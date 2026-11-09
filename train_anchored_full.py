"""Fit anchored candidates and matched controls, fixed five-epoch full-TRAIN budget."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
from train_repaired_full import load_data,forward,sha_file,OLD
from repaired_model import RepairedSequenceFoundation
from profile_model import AffinityProfile
from gpu_lease import gpu_lease
OUT=HERE/'interaction_study_20261109'


def dump(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def main():
    if (OUT/'anchored_full_results.json').exists():raise RuntimeError('Already completed')
    start=time.monotonic();torch.set_num_threads(2)
    study=json.loads((OUT/'study_results.json').read_text())
    priorpath=OUT/'chemical_checkpoint.npz'
    assert sha_file(priorpath)==study['full_train_profile_models']['chemical']['checkpoint_sha256']
    assert sha_file(HERE/'profile_model.py')==study['profile_model_sha256']
    info=load_data();assert info['row_sha256']==study['row_sha256']
    prior_model=AffinityProfile.load(priorpath)
    prior=prior_model.chemical(info['bits'].cpu().numpy())
    results={'status':'FULL_TRAIN_ENGINEERING_CANDIDATES_WITH_MATCHED_CONTROLS',
        'seeds':[2931,2932,2933],'epochs':5,'steps_per_arm':120,'lr':.0003,'weight_decay':.0001,
        'selection':'Fixed five epochs, no DEV epoch or seed selection',
        'prior_selection':'Nested selection on original 24 TRAIN proteins only; frozen chemical prior',
        'data_sha256':info['row_sha256'],'repair_source_sha256':info['repair_source_sha256'],
        'profile_source_sha256':sha_file(HERE/'profile_model.py'),'prior_sha256':sha_file(priorpath),
        'limitations':['Previously inspected eight-target DEV; no independent scientific test',
                       'All 54 DEV compounds are known to these full-data models',
                       'Compare new-chemistry generalization using separate nested CV, not this DEV score'],
        'arms':[]}
    dump(OUT/'anchored_full_protocol.json',results)
    for seed in results['seeds']:
        torch.manual_seed(seed)
        initial=RepairedSequenceFoundation().cuda()
        state={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()};del initial
        for anchored in [False,True]:
            name=('anchored' if anchored else 'matched')+'_seed_'+str(seed)
            path=OUT/name;path.mkdir(exist_ok=True)
            if (path/'checkpoint.pt').exists():raise RuntimeError('Existing checkpoint '+name)
            torch.manual_seed(seed)
            model=RepairedSequenceFoundation().cuda();model.load_state_dict(state,strict=True)
            if anchored:
                nn.init.zeros_(model.pair_head[-1].weight);nn.init.zeros_(model.pair_head[-1].bias)
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
            hist=[]
            for epoch in range(1,6):
                model.train();losses=[]
                for idx in np.random.default_rng(seed*100+epoch).permutation(24):
                    p=info['train_ids'][idx];opt.zero_grad(set_to_none=True)
                    pred=forward(model,model.compound(info['sm'],info['bits']),info['embs'][p])
                    offset=prior if anchored else info['mean']
                    y=torch.tensor((info['train'][p]-offset)/info['std'],device='cuda',dtype=torch.float32)
                    loss=(pred-y).square().mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                    losses.append(float(loss.detach()))
                hist.append({'epoch':epoch,'standardized_train_loss':float(np.mean(losses))})
            ck={'kind':'anchored_sequence_v4_3' if anchored else 'matched_repaired_v4_3',
                'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'seed':seed,'epoch':5,'label_std':info['std'],'label_mean':info['mean'],
                'row_sha256':info['row_sha256'],'fingerprint_sha256':info['fingerprint_sha256'],
                'repair_source_sha256':info['repair_source_sha256'],
                'profile_source_sha256':results['profile_source_sha256'],
                'prior_sha256':results['prior_sha256'],'architecture':'two_path_v1_plus_frozen_chemical_prior' if anchored else 'two_path_v1'}
            tmp=path/'checkpoint.pt.tmp';torch.save(ck,tmp);os.replace(tmp,path/'checkpoint.pt')
            # Score only after saving final weights; no dev-driven selection.
            model.eval();per={}
            with torch.no_grad():
                c=model.compound(info['sm'],info['bits'])
                for p in info['dev_ids']:
                    raw=forward(model,c,info['embs'][p]).cpu().numpy()*info['std']
                    pred=raw+(prior if anchored else info['mean'])
                    y=info['dev'][p];e=pred-y
                    per[p]={'rmse':float(np.sqrt(np.mean(e**2))),
                            'prediction_sd':float(np.std(pred)),
                            'bias_squared':float(np.mean(e)**2),
                            'centered_mse':float(np.mean((e-e.mean())**2))}
            item={'name':name,'seed':seed,'anchored':anchored,'history':hist,
                  'checkpoint_sha256':sha_file(path/'checkpoint.pt'),
                  'dev_macro_rmse':float(np.mean([v['rmse'] for v in per.values()])),
                  'per_target':per}
            dump(path/'COMPLETE.json',item);results['arms'].append(item)
            dump(OUT/'anchored_full_progress.json',results)
            print(json.dumps({'name':name,'dev_macro_rmse':item['dev_macro_rmse']}),flush=True)
            del model,opt;torch.cuda.empty_cache()
    results['elapsed_seconds']=time.monotonic()-start
    dump(OUT/'anchored_full_results.json',results)
    print(json.dumps({'status':'ANCHORED_FULL_COMPLETE','elapsed_seconds':results['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    with gpu_lease(OLD):main()
