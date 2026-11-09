"""Read-only diagnosis of current repaired checkpoints; no model selection."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from train_repaired_full import load_data, forward, dump, OLD, sha_file
from repaired_model import RepairedSequenceFoundation
from gpu_lease import gpu_lease


def metrics(pred,y):
    pred=np.asarray(pred,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    error=pred-y
    return {'macro_rmse':float(np.sqrt(np.mean(error**2,axis=1)).mean()),
            'pooled_rmse':float(np.sqrt(np.mean(error**2))),
            'mean_target_bias_squared':float(np.mean(error.mean(1)**2)),
            'centered_mse':float(np.mean((error-error.mean(1,keepdims=True))**2)),
            'mean_target_prediction_sd':float(pred.std(1).mean())}


def main():
    torch.set_num_threads(2)
    out=HERE/'interaction_study_20261109'
    out.mkdir(exist_ok=True)
    if (out/'diagnosis.json').exists():raise RuntimeError('Already diagnosed; preserve prior result')
    info=load_data()
    train_y=np.stack([info['train'][p] for p in info['train_ids']])
    dev_y=np.stack([info['dev'][p] for p in info['dev_ids']])
    prior=train_y.mean(0)
    result={'status':'PREVIOUSLY_INSPECTED_DEV_DIAGNOSTIC_NO_SELECTION',
            'data_sha256':info['row_sha256'],'train_proteins':info['train_ids'],
            'dev_proteins':info['dev_ids'],
            'baseline':metrics(np.broadcast_to(prior,dev_y.shape),dev_y),'models':{}}
    mu=train_y.mean();row=train_y.mean(1,keepdims=True)-mu;col=train_y.mean(0,keepdims=True)-mu
    interaction=train_y-mu-row-col
    result['train_label_variance']={'total':float(train_y.var()),'protein_main':float(np.mean(row**2)),
        'compound_main':float(np.mean(col**2)),'interaction':float(np.mean(interaction**2))}
    ensembles=[];etr=[]
    for seed in [2831,2832,2833]:
        ckpt=HERE/'checkpoints'/('seed_'+str(seed))/'checkpoint.pt'
        ck=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert ck['row_sha256']==info['row_sha256']
        assert ck['repair_source_sha256']==sha_file(HERE/'repaired_model.py')
        m=RepairedSequenceFoundation().cuda().eval();m.load_state_dict(ck['model_state'],strict=True)
        def score(ids):
            with torch.no_grad():
                c=m.compound(info['sm'],info['bits'])
                return np.stack([(forward(m,c,info['embs'][p])*ck['label_std']+ck['label_mean']).cpu().numpy() for p in ids]).astype(np.float64)
        tr=score(info['train_ids']);dv=score(info['dev_ids'])
        ensembles.append(dv);etr.append(tr)
        result['models'][str(seed)]={'original':metrics(dv,dev_y),
            'unit_weight_chemical_anchor':metrics(prior[None]+dv-tr.mean(0,keepdims=True),dev_y),
            'train_compound_mean_prediction_rmse':float(np.sqrt(np.mean((tr.mean(0)-prior)**2)))}
        del m;torch.cuda.empty_cache()
    dv=np.mean(ensembles,axis=0);tr=np.mean(etr,axis=0)
    result['ensemble']={'unweighted':metrics(dv,dev_y),
        'unit_weight_chemical_anchor':metrics(prior[None]+dv-tr.mean(0,keepdims=True),dev_y),
        'anchor_note':'One predeclared weight=1 counterfactual; DEV is not used to select a blending weight.'}
    result['per_target']={p:{'baseline':metrics(prior[None],dev_y[i:i+1]),
        'ensemble':metrics(dv[i:i+1],dev_y[i:i+1]),
        'anchored_ensemble':metrics(prior[None]+dv[i:i+1]-tr.mean(0,keepdims=True),dev_y[i:i+1])}
        for i,p in enumerate(info['dev_ids'])}
    dump(out/'diagnosis.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    with gpu_lease(OLD):main()
