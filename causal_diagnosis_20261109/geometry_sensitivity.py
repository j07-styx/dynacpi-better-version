"""Separate geometric residual size from frozen readout sensitivity, fixed seed.
All scoring is on the four already inspected development proteins. No fitting.
"""
import sys
import json
import numpy as np
import torch
from causal_diagnosis import ROOT, OUT, save, scalar, gpu_lease
from dynacpi.geometry_branch import GeometryBranch
from scripts.train_v4_geometry_trainonly_20261109 import setup


def main():
    if (OUT/'geometry_sensitivity.json').exists():raise RuntimeError('Already complete')
    base,data,fit,inner,dev,mean,std,hashes=setup()
    geom=GeometryBranch().cuda().eval()
    ck=torch.load(ROOT/'research/dynacpi_v4_trainonly_geometry_20261109/real_seed1901/best_checkpoint.pt',map_location='cpu',weights_only=False)
    geom.load_state_dict(ck['state'],strict=True)
    deltas={}
    with torch.no_grad():
        for p in data:
            t=data[p]['tokens']
            deltas[p]=(geom.enhance(t,data[p]['graph'])-t).detach().clone()
        common=torch.stack([deltas[p].mean(0) for p in fit]).mean(0)
    result={'status':'SAME_WEIGHTS_DEV_ONLY_PERTURBATION_DIAGNOSIS','seed':1901,
            'common_perturbation':'Mean latent geometry residual across seven fit proteins, broadcast equally at all residues',
            'common_vector_norm':scalar(common.norm()),'input_hashes':hashes,'proteins':{}}
    for p in dev:
        t=data[p]['tokens'].detach().clone().requires_grad_()
        x=data[p]['x'].detach().clone()
        c=x.mean(0,keepdim=True)
        mask=torch.ones((1,len(t)),device='cuda',dtype=torch.bool)
        yp=base.predict_tokens(c,t[None],mask)*std+mean
        grad=torch.autograd.grad(yp.sum(),t)[0]
        d=deltas[p]
        with torch.no_grad():
            fullmask=torch.ones((len(x),len(t)),device='cuda',dtype=torch.bool)
            def pred(tokens):return base.predict_tokens(x,tokens[None].expand(len(x),-1,-1),fullmask)*std+mean
            before=pred(t)
            actual=pred(t+d)-before
            constant=pred(t+d.mean(0)[None])-before
            centered=pred(t+d-d.mean(0)[None])-before
            shared=pred(t+common[None])-before
            record={'protein_length':len(t),'mapped_residues':len(data[p]['graph']['indices']),
                    'delta_rms':scalar(d.square().mean().sqrt()),
                    'delta_mean_vector_norm':scalar(d.mean(0).norm()),
                    'uniform_shift_gradient_norm':scalar(grad.sum(0).norm()),
                    'first_order_delta_pKd':scalar((grad*d).sum()),
                    'actual_mean_delta_pKd':scalar(actual.mean()),
                    'constant_mean_residual_delta_pKd':scalar(constant.mean()),
                    'centered_residual_delta_pKd':scalar(centered.mean()),
                    'identical_common_perturbation_delta_pKd':scalar(shared.mean()),
                    'actual_delta_sd':scalar(actual.std(unbiased=False))}
            result['proteins'][data[p]['name']]=record
    save('geometry_sensitivity.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    with gpu_lease(ROOT):main()
