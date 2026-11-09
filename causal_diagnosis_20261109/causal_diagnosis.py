"""Read-only checkpoint diagnostics and isolated train-set repair experiments.

Run on the user's device with its existing DynaCPI environment. Existing model,
data, checkpoints, APIs and git checkout are never modified. Outputs go beside
this script. Uses only the historical engineering pilot (no prospective test).
"""
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(r"C:\Users\J0124\dynacpi_v4")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dynacpi.sequence_model import SequenceFoundation
from chemprop.features import mol2graph
from gpu_lease import gpu_lease


def save(name, data):
    path = OUT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def scalar(x):
    return float(x.detach().cpu())


def stats(x):
    x = x.detach().float()
    xc = x - x.mean(0, keepdim=True)
    normed = nn.functional.normalize(x, dim=1)
    cs = normed @ normed.T
    sv = torch.linalg.svdvals(xc)
    e = sv.square()
    prob = e / e.sum().clamp_min(1e-30)
    return {"cosine": scalar((cs.sum() - cs.diag().sum()) / (len(x)*(len(x)-1))),
            "centered_rms": scalar(xc.square().mean().sqrt()),
            "rms": scalar(x.square().mean().sqrt()),
            "centered_effective_rank": scalar(torch.exp(-(prob * prob.clamp_min(1e-30).log()).sum())),
            "norm_sd": scalar(x.norm(dim=1).std(unbiased=False))}


def cab_stats(layer, query, key, value):
    q, k, v = layer.q(query), layer.k(key), layer.v(value)
    logits = q.unsqueeze(-1) * k.unsqueeze(-2) / layer.scale
    w = logits.softmax(-1)
    entropy = -(w * w.clamp_min(1e-30).log()).sum(-1)
    actual = layer(query, key, value)
    uniform = layer.out(v.mean(-1, keepdim=True).expand_as(v))
    derivative = ((w*k[:,None,:]*v[:,None,:]).sum(-1)
                  -(w*k[:,None,:]).sum(-1)*(w*v[:,None,:]).sum(-1)) / layer.scale
    return {"logit_sd": scalar(logits.std(unbiased=False)),
            "effective_channel_fraction": scalar(entropy.exp().mean()/q.shape[-1]),
            "mean_abs_weight_deviation_from_uniform": scalar((w-1/q.shape[-1]).abs().mean()),
            "rms_difference_from_uniform_output": scalar((actual-uniform).square().mean().sqrt()),
            "relative_difference_from_uniform_output": scalar((actual-uniform).norm()/actual.norm().clamp_min(1e-20)),
            "mean_abs_scalar_query_derivative": scalar(derivative.abs().mean()),
            "max_abs_scalar_query_derivative": scalar(derivative.abs().max()),
            "output": stats(actual)}


def parts(m, sm, bits):
    raw = m.mpn([mol2graph(sm)])
    lo = raw.amin(-1, keepdim=True)
    graph = (raw-lo)/(raw.amax(-1, keepdim=True)-lo).clamp_min(1e-6)
    fp = m.fp(bits)
    first = m.channel(fp, graph, graph)
    mixed = graph + first
    final = m.channel(mixed, mixed, mixed)
    return raw, graph, fp, first, mixed, final


def prediction(m, c, emb):
    t = m.sequence(emb.float())
    mask = torch.ones((len(c), len(t)), device=t.device, dtype=torch.bool)
    return m.predict_tokens(c, t[None].expand(len(c),-1,-1), mask)


def metric(pred, y):
    p, y = np.asarray(pred), np.asarray(y)
    centered = (p-p.mean())-(y-y.mean())
    den = np.linalg.norm(p-p.mean())*np.linalg.norm(y-y.mean())
    return {"rmse": float(np.sqrt(np.mean((p-y)**2))),
            "pred_mean": float(p.mean()), "y_mean": float(y.mean()),
            "pred_sd": float(p.std()), "y_sd": float(y.std()),
            "bias_squared": float((p.mean()-y.mean())**2),
            "centered_mse": float(np.mean(centered**2)),
            "pearson": float(np.dot(p-p.mean(),y-y.mean())/den) if den>1e-14 else None,
            "floor_fraction": float(np.mean(y==5.0))}


class Repair(SequenceFoundation):
    """Two independently switchable bypasses; experimental, not promoted."""
    def __init__(self, state, encoder=False, readout=False):
        super().__init__(device="cuda")
        self.load_state_dict(state, strict=True)
        self.repair_encoder = encoder
        self.repair_readout = readout
        if encoder:
            self.chemical_norm = nn.LayerNorm(256)
        if readout:
            self.pair_head = nn.Sequential(nn.Linear(256*4,256), nn.ReLU(), nn.Linear(256,1))

    def compound(self, smiles, fingerprints):
        if not self.repair_encoder:
            return super().compound(smiles, fingerprints)
        graph = self.mpn([mol2graph(list(smiles))])
        # Preserve molecular features directly, omit both channel attention gates
        # and the per-molecule min/max operation in this candidate repair.
        return self.chemical_norm(self.fp(fingerprints) + graph)

    def predict_tokens(self, c, tokens, valid):
        if not self.repair_readout:
            return super().predict_tokens(c, tokens, valid)
        residue = self.readout(c.unsqueeze(1),tokens,tokens,
                              key_padding_mask=~valid,need_weights=False)[0][:,0]
        mean = (tokens*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True)
        p = self.norm(mean+residue)
        return self.pair_head(torch.cat([c,p,c*p,(c-p).abs()],dim=-1)).squeeze(-1)


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    if (OUT / "causal_results.json").exists():
        raise RuntimeError("Refusing to overwrite completed diagnostic")
    torch.manual_seed(1105)
    device = "cuda"
    rowfile = ROOT/"data/engineering_pilot/rows.json"
    rows = json.loads(rowfile.read_text())
    assert {r['split'] for r in rows} == {'train','val'}
    train = [r for r in rows if r['split']=='train']
    checkpoint = ROOT/"research/pilot_sequence/checkpoint.pt"
    ck = torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert ck['row_sha256']==hashlib.sha256(rowfile.read_bytes()).hexdigest()
    base = SequenceFoundation(device=device).to(device).eval()
    base.load_state_dict(ck['state'],strict=True)
    fresh = SequenceFoundation(device=device).to(device).eval()
    mean, std = ck['label_mean'],ck['label_std']
    stage = ROOT/"data/staging"
    smiles = json.loads((stage/"compounds.json").read_text())
    fpz = np.load(stage/"morgan_radius2_2048_chiral.npz")
    fp = dict(zip(fpz['ids'].astype(str),fpz['bits']))
    cids = sorted({r['compound'] for r in train})
    cid_idx = {c:i for i,c in enumerate(cids)}
    sm = [smiles[c] for c in cids]
    bits = torch.tensor(np.stack([fp[c] for c in cids]),dtype=torch.float32,device=device)
    manifest = json.loads((ROOT/"data/engineering_pilot/limited_alpha_3d_graph_manifest.json").read_text())
    names = {r['sequence_sha256']:r['source_name'] for r in manifest['records']}
    embs = {}
    targets = {}
    for pid in sorted({r['sequence'] for r in rows}):
        rr = [r for r in rows if r['sequence']==pid]
        d = {r['compound']:r['y'] for r in rr}
        assert len(d)==54 and set(d)==set(cids)
        targets[pid] = np.array([d[c] for c in cids])
        obj = torch.load(ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(pid+'.pt'),map_location="cpu",weights_only=True)
        assert obj['sequence_hash']==pid
        embs[pid] = obj['embeddings']
    result = {"status":"ENGINEERING_DIAGNOSIS_NOT_INDEPENDENT_EVALUATION",
              "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "rows_sha256":ck['row_sha256'],"checkpoint_epoch":ck['epoch'],
              "label_mean":mean,"label_std":std,
              "optimizer_updates":math.ceil(len(train)/64)*3,
              "data":{s:{"pairs":sum(r['split']==s for r in rows),
                           "proteins":len({r['sequence'] for r in rows if r['split']==s}),
                           "floor_fraction":float(np.mean([r['y']==5 for r in rows if r['split']==s]))}
                      for s in ['train','val']}}
    result['representation']={}
    with torch.no_grad():
        for label,m in [('checkpoint',base),('random_init',fresh)]:
            raw,g,f,a,mix,c=parts(m,sm,bits)
            result['representation'][label]={
                'layers':{k:stats(v) for k,v in [('mpn',raw),('scaled_mpn',g),('fp',f),('first',a),('mixed',mix),('final',c)]},
                'channel1':cab_stats(m.channel,f,g,g),
                'channel2':cab_stats(m.channel,mix,mix,mix)}
        c=base.compound(sm,bits)
        result['per_protein']={}
        train_ids=sorted({r['sequence'] for r in train})
        compound_mean=np.stack([targets[p] for p in train_ids]).mean(0)
        result['baseline_comparison']={}
        for pid,emb in embs.items():
            ee=emb.to(device).float()
            pred=prediction(base,c,ee).cpu().numpy()*std+mean
            fixed=prediction(base,c.mean(0,keepdim=True).expand_as(c),ee).cpu().numpy()*std+mean
            result['per_protein'][pid]={'name':names.get(pid,pid[:12]),
                'split':'train' if pid in train_ids else 'val',**metric(pred,targets[pid]),
                'replace_all_compounds_by_mean_max_abs':float(np.max(np.abs(pred-fixed))),
                'compound_mean_baseline_rmse':metric(compound_mean,targets[pid])['rmse']}
        for split in ['train','val']:
            vals=[v for v in result['per_protein'].values() if v['split']==split]
            result['baseline_comparison'][split]={
                'checkpoint_macro_rmse':float(np.mean([v['rmse'] for v in vals])),
                'compound_mean_macro_rmse':float(np.mean([v['compound_mean_baseline_rmse'] for v in vals]))}
        adck=next(p for p in targets if names.get(p)=='ADCK4')
        ee=embs[adck].to(device).float()
        tokens=base.sequence(ee)[None].expand(54,-1,-1)
        mask=torch.ones(tokens.shape[:2],device=device,dtype=torch.bool)
        residue=base.readout(c[:,None],tokens,tokens,key_padding_mask=~mask,need_weights=False)[0][:,0]
        p=base.norm(tokens.mean(1)+residue)
        result['final_prediction_channel']=cab_stats(base.channel,c,p,p)
        result['scaling_counterfactual']={}
        for scale in [16.0,4.0,1.0]:
            base.channel.scale=scale
            cc=base.compound(sm,bits)
            pp=prediction(base,cc,ee).cpu().numpy()*std+mean
            result['scaling_counterfactual'][str(scale)]={'compound':stats(cc),'prediction_sd':float(pp.std()),
                    'note':'Same weights, altered scale; not a trained repair or accuracy estimate.'}
        base.channel.scale=16.0
    # Select ONE original TRAIN protein, by its training-label spread, to test
    # capacity/optimization on 16 fixed pairs. This is deliberately not a benchmark.
    pid=max(train_ids,key=lambda p:targets[p].std())
    order=np.argsort(targets[pid],kind='stable')
    ids=order[np.linspace(0,53,16).round().astype(int)]
    small_sm=[sm[i] for i in ids]
    small_bits=bits[ids]
    small_emb=embs[pid].to(device).float()
    small_y=torch.tensor((targets[pid][ids]-mean)/std,device=device,dtype=torch.float32)
    base.eval();base.zero_grad(set_to_none=True)
    fpin=small_bits.detach().clone().requires_grad_()
    cc=base.compound(small_sm,fpin);cc.retain_grad()
    tt=base.sequence(small_emb);tt.retain_grad()
    vv=torch.ones((16,len(tt)),device=device,dtype=torch.bool)
    pp=base.predict_tokens(cc,tt[None].expand(16,-1,-1),vv)
    loss=(pp-small_y).square().mean();loss.backward()
    result['gradient_probe']={'protein':names.get(pid,pid),'training_only':True,
        'fp_input_grad_norm':scalar(fpin.grad.norm()),'compound_grad_norm':scalar(cc.grad.norm()),
        'protein_token_grad_norm':scalar(tt.grad.norm()),'module_gradient_norms':{}}
    for prefix in ['mpn.','fp.','channel.q.','channel.k.','channel.v.','channel.out.','sequence.','readout.','head.']:
        weights=[p for n,p in base.named_parameters() if n.startswith(prefix) and p.grad is not None]
        result['gradient_probe']['module_gradient_norms'][prefix]={
            'grad_norm':math.sqrt(sum(scalar(p.grad.square().sum()) for p in weights)),
            'weight_norm':math.sqrt(sum(scalar(p.square().sum()) for p in weights))}
    base.zero_grad(set_to_none=True)
    with torch.no_grad():
        same_sm=[small_sm[0]]*16
        same_bits=small_bits[:1].expand(16,-1)
        base.train()
        dropout_values=[]
        for _ in range(4):
            dropout_values.extend((prediction(base,base.compound(same_sm,same_bits),small_emb)*std+mean).tolist())
        base.eval()
        real=(prediction(base,base.compound(sm,bits),small_emb)*std+mean).cpu().numpy()
        result['dropout_probe']={'same_pair_train_dropout_sd':float(np.std(dropout_values)),
                                 '54_distinct_compounds_eval_sd':float(real.std())}
    save('causal_results.json',result)
    print(json.dumps({'status':'CAUSAL_DIAGNOSTIC_COMPLETE',
        'data':result['data'],'baseline':result['baseline_comparison'],
        'cab1':result['representation']['checkpoint']['channel1'],
        'cab2':result['representation']['checkpoint']['channel2'],
        'final_cab':result['final_prediction_channel'],
        'gradient':result['gradient_probe'],'dropout':result['dropout_probe']}),flush=True)
    micro={'status':'TRAIN_ONLY_CAPACITY_TEST_NOT_GENERALIZATION',
           'protein':names.get(pid,pid),'protein_hash':pid,'compound_hashes':[cids[i] for i in ids],
           'training_pairs':16,'target_sd':float(targets[pid][ids].std()),
           'steps':200,'learning_rate':0.001,'dropout_disabled_all_arms':True,'arms':{}}
    # Equal update budget, all existing parameters unfrozen; fresh replacement
    # heads make this a repair smoke test, not a capacity-matched benchmark.
    for name,enc,read in [('original',False,False),('encoder_only',True,False),('readout_only',False,True),('both',True,True)]:
        torch.manual_seed(2831)
        m=Repair(ck['state'],encoder=enc,readout=read).to(device)
        m.eval()  # eval disables dropout, but autograd/optimizer remain enabled.
        opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0)
        hist=[]
        for step in range(201):
            opt.zero_grad(set_to_none=True)
            yy=prediction(m,m.compound(small_sm,small_bits),small_emb)
            loss=(yy-small_y).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite microfit')
            if step in [0,10,25,50,100,200]:
                vals=(yy.detach()*std+mean).cpu().numpy()
                hist.append({'step':step,**metric(vals,targets[pid][ids])})
            if step<200:
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(),1.)
                opt.step()
        micro['arms'][name]={'history':hist,'parameters':sum(p.numel() for p in m.parameters())}
        save('microfit_results.json',micro)
        print(json.dumps({'arm':name,'initial':hist[0],'final':hist[-1]}),flush=True)
        del m,opt
        torch.cuda.empty_cache()
    result['elapsed_seconds']=time.monotonic()-start
    save('causal_results.json',result)
    print(json.dumps({'status':'ALL_COMPLETE','elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with gpu_lease(ROOT):
        main()
