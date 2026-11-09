"""Nested training-only chemical/target holdouts plus fixed-budget neural controls.

Only the original 24 TRAIN targets and 54 compounds participate in CV. Each outer
fold fits 16 targets and 36 compounds, and scores cold-target, cold-compound, and
double-cold cells separately. Scaffold/near-neighbor groups remain disjoint.
The old 8-target DEV is diagnostic only, scored after all selections are frozen.
"""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE=Path(__file__).resolve().parent
OLD=Path(r'C:\Users\J0124\dynacpi_v4')
sys.path.insert(0,str(OLD));sys.path.insert(0,str(OLD/'scripts'))
from gpu_lease import gpu_lease
from repaired_model import RepairedSequenceFoundation
from profile_model import AffinityProfile,tanimoto

OUT=HERE/'interaction_study_20261109'
SEEDS=[2931,2932,2933]
EPOCHS=5


def dump(name,obj):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def metric(p,y):
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    e=p-y
    return {'macro_rmse':float(np.sqrt(np.mean(e*e,axis=1)).mean()),
        'pooled_rmse':float(np.sqrt(np.mean(e*e))),
        'target_bias_mse':float(np.mean(e.mean(1)**2)),
        'centered_mse':float(np.mean((e-e.mean(1,keepdims=True))**2)),
        'mean_pred_sd':float(p.std(1).mean()),'proteins':len(y),'compounds':y.shape[1]}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition_groups(groups,k,salt):
    bins=[[] for _ in range(k)]
    ordered=sorted(groups,key=lambda g:(-len(g),hashlib.sha256((salt+'|'+str(sorted(g))).encode()).hexdigest()))
    for g in ordered:
        dest=min(range(k),key=lambda i:(len(bins[i]),i));bins[dest].extend(g)
    return [sorted(b) for b in bins]


def chemistry_groups(smiles,fingerprints):
    scaff=[MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s),includeChirality=False) for s in smiles]
    # Empty ring scaffolds do not make all acyclic compounds one scaffold.
    scaff=[s if s else 'acyclic:'+smiles[i] for i,s in enumerate(scaff)]
    sim=tanimoto(fingerprints,fingerprints);parent=list(range(len(smiles)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(smiles)):
        for j in range(i):
            if scaff[i]==scaff[j] or sim[i,j]>=.5:parent[root(i)]=root(j)
    d={}
    for i in range(len(smiles)):d.setdefault(root(i),[]).append(i)
    return list(d.values()),scaff,sim


def inner_select(pfeat,bits,y,groups):
    """Two paired internal partitions; balance all three generalization cells."""
    pf=partition_groups([[i] for i in range(len(pfeat))],2,'inner_protein_v1')
    cf=partition_groups(groups,2,'inner_chemical_v1')
    allp=set(range(len(pfeat)));allc=set(range(len(bits)))
    candidates=[]
    for lc in [.0001,.1,1.0,10.0]:
        for lr in [.01,.1,1.0,10.0,100.0]:
            scores={m:[] for m in ['chemical','additive','interaction']}
            for f in range(2):
                pi=sorted(allp-set(pf[f]));ci=sorted(allc-set(cf[f]))
                m=AffinityProfile(lc,lr).fit(pfeat[pi],bits[ci],y[np.ix_(pi,ci)])
                for mode in scores:
                    # Equal weight per regime, then equal weight per partition.
                    vals=[]
                    for pp,cc in [(pf[f],ci),(pi,cf[f]),(pf[f],cf[f])]:
                        vals.append(metric(m.predict(pfeat[pp],bits[cc],mode),y[np.ix_(pp,cc)])['macro_rmse'])
                    scores[mode].append(float(np.mean(vals)))
            candidates.append({'chemical_ridge':lc,'residual_ridge':lr,
                'scores':{m:float(np.mean(v)) for m,v in scores.items()}})
    chosen={}
    for mode in ['chemical','additive','interaction']:
        best=min(candidates,key=lambda r:r['scores'][mode])
        chosen[mode]={'chemical_ridge':best['chemical_ridge'],'residual_ridge':best['residual_ridge'],
                      'inner_score':best['scores'][mode]}
    return chosen,candidates


def forward(m,c,e):
    t=m.sequence(e)
    valid=torch.ones((len(c),len(t)),device=e.device,dtype=torch.bool)
    return m.predict_tokens(c,t[None].expand(len(c),-1,-1),valid)


def main():
    start=time.monotonic();torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if (OUT/'study_results.json').exists():raise RuntimeError('Completed study already exists')
    rows_path=OLD/'data/engineering_pilot/rows.json'
    rows=json.loads(rows_path.read_text())
    train=[r for r in rows if r['split']=='train']
    pids=sorted({r['sequence'] for r in train});cids=sorted({r['compound'] for r in train})
    assert len(pids)==24 and len(cids)==54 and len(train)==1296
    smiles=json.loads((OLD/'data/staging/compounds.json').read_text());sm=[smiles[c] for c in cids]
    fpfile=OLD/'data/staging/morgan_radius2_2048_chiral.npz'
    z=np.load(fpfile);fp=dict(zip(z['ids'].astype(str),z['bits']))
    bits=np.stack([fp[c] for c in cids]).astype(np.float64)
    labels={(r['sequence'],r['compound']):float(r['y']) for r in train}
    y=np.array([[labels[p,c] for c in cids] for p in pids])
    emb=[]
    for p in pids:
        obj=torch.load(OLD/'data/esm2_t33_650M_layer33_w1000_o128_fp16'/(p+'.pt'),map_location='cpu',weights_only=True)
        assert obj['sequence_hash']==p
        emb.append(obj['embeddings'].float().cuda())
    pfeat=np.stack([e.mean(0).cpu().numpy() for e in emb]).astype(np.float64)
    groups,scaff,sim=chemistry_groups(sm,bits)
    cf=partition_groups(groups,3,'outer_compound_v1')
    pf=partition_groups([[i] for i in range(24)],3,'outer_protein_v1')
    for f in range(3):
        ci=sorted(set(range(54))-set(cf[f]))
        assert not set(scaff[i] for i in ci)&set(scaff[j] for j in cf[f])
        assert float(sim[np.ix_(ci,cf[f])].max())<.5
    protocol={'status':'HISTORICAL_TRAIN_ONLY_NESTED_CROSS_VALIDATION_NOT_PROSPECTIVE_TEST',
        'row_sha256':digest(rows_path),'fingerprint_sha256':digest(fpfile),
        'repaired_model_sha256':digest(HERE/'repaired_model.py'),
        'profile_model_sha256':digest(HERE/'profile_model.py'),
        'protein_hashes':pids,'compound_hashes':cids,'chemical_groups':groups,
        'unique_murcko_scaffolds':len(set(scaff)),
        'chemical_group_rule':'Connected components of identical Murcko scaffold OR Morgan Tanimoto >= 0.50',
        'outer_protein_folds':pf,'outer_chemical_folds':cf,
        'max_cross_chemical_similarity':[float(sim[np.ix_(sorted(set(range(54))-set(cf[f])),cf[f])].max()) for f in range(3)],
        'neural_seeds':SEEDS,'epochs':EPOCHS,'lr':.0003,'weight_decay':.0001,
        'neural_selection':'fixed epoch 5 for both models; no outer-score selection',
        'profile_selection':'two inner paired folds; mean RMSE over cold protein, cold compound and double cold',
        'limitations':['24 targets and 54 compounds only; historically used data, not a new independent test',
            'Protein identities are disjoint but protein families are not clustered',
            'Neural/profile model capacities and hyperparameter-selection budgets differ',
            'Three paired outer folds assess selected cells, not all nine cross-products',
            'Binary fingerprints, target ESM pooling, and no 3D or support labels'],
        'folds':[]}
    dump('study_protocol.json',protocol)
    result=copy.deepcopy(protocol)
    for fold in range(3):
        pi=sorted(set(range(24))-set(pf[fold]));ci=sorted(set(range(54))-set(cf[fold]))
        subgroups=[[ci.index(v) for v in g if v in ci] for g in groups]
        subgroups=[g for g in subgroups if g]
        choices,grid=inner_select(pfeat[pi],bits[ci],y[np.ix_(pi,ci)],subgroups)
        cells={'cold_protein':(pf[fold],ci),'cold_compound':(pi,cf[fold]),'double_cold':(pf[fold],cf[fold])}
        rec={'fold':fold,'train_proteins':len(pi),'train_compounds':len(ci),
             'inner_choices':choices,'inner_grid':grid,'models':{}}
        def score_model(func):return {reg:metric(func(pp,cc),y[np.ix_(pp,cc)]) for reg,(pp,cc) in cells.items()}
        mu=y[np.ix_(pi,ci)].mean()
        rec['models']['global_mean']=score_model(lambda pp,cc:np.full((len(pp),len(cc)),mu))
        # The compound-average lookup is defined only for compounds seen in fit.
        seen_prior=y[np.ix_(pi,ci)].mean(0)
        rec['seen_compound_mean_cold_protein']=metric(np.broadcast_to(seen_prior,(len(pf[fold]),len(ci))),y[np.ix_(pf[fold],ci)])
        fitted={}
        for mode,ch in choices.items():
            model=AffinityProfile(ch['chemical_ridge'],ch['residual_ridge'],mode).fit(pfeat[pi],bits[ci],y[np.ix_(pi,ci)])
            rec['models'][mode+'_profile']=score_model(lambda pp,cc,m=model:m.predict(pfeat[pp],bits[cc]))
            fitted[mode]=model
        # Chemical prior selected without outer labels, frozen throughout both neural arms.
        prior=fitted['chemical'].chemical(bits)
        scale=float(y[np.ix_(pi,ci)].std())
        bit_tensor=torch.tensor(bits,dtype=torch.float32,device='cuda')
        fit_sm=[sm[i] for i in ci]
        for seed in SEEDS:
            torch.manual_seed(seed)
            initial=RepairedSequenceFoundation().cuda()
            initial_state={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}
            del initial
            for anchored in [False,True]:
                name=('anchored_neural' if anchored else 'repaired_neural')+'_seed'+str(seed)
                torch.manual_seed(seed)
                m=RepairedSequenceFoundation().cuda();m.load_state_dict(initial_state,strict=True)
                if anchored:
                    nn.init.zeros_(m.pair_head[-1].weight);nn.init.zeros_(m.pair_head[-1].bias)
                opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001)
                hist=[]
                for epoch in range(1,EPOCHS+1):
                    m.train();losses=[]
                    for idx in np.random.default_rng(seed*100+epoch).permutation(len(pi)):
                        p=pi[idx];opt.zero_grad(set_to_none=True)
                        pred=forward(m,m.compound(fit_sm,bit_tensor[ci]),emb[p])
                        offset=prior[ci] if anchored else mu
                        label=torch.tensor((y[p,ci]-offset)/scale,device='cuda',dtype=torch.float32)
                        loss=(pred-label).square().mean()
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                        loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                        losses.append(float(loss.detach()))
                    hist.append({'epoch':epoch,'standardized_train_loss':float(np.mean(losses))})
                m.eval()
                with torch.no_grad():
                    cc=m.compound(sm,bit_tensor)
                    pred=np.stack([(forward(m,cc,emb[p])*scale).cpu().numpy() for p in range(24)])
                pred+=prior[None] if anchored else mu
                rec['models'][name]=score_model(lambda pp,cc,pred=pred:pred[np.ix_(pp,cc)])
                rec.setdefault('training_history',{})[name]=hist
                print(json.dumps({'fold':fold,'model':name,
                    'metrics':{k:v['macro_rmse'] for k,v in rec['models'][name].items()}}),flush=True)
                del m,opt;torch.cuda.empty_cache()
        result['folds'].append(rec);dump('study_progress.json',result)
        print(json.dumps({'fold':fold,'status':'FOLD_COMPLETE','profile':{mode:rec['models'][mode+'_profile'] for mode in choices}}),flush=True)
    # Fit one profile candidate using a fresh nested selection on all historical
    # TRAIN data; the old eight DEV proteins never participate in selection.
    choices,grid=inner_select(pfeat,bits,y,groups)
    full={}
    for mode,ch in choices.items():
        m=AffinityProfile(ch['chemical_ridge'],ch['residual_ridge'],mode).fit(pfeat,bits,y)
        path=OUT/(mode+'_checkpoint.npz');m.save(path)
        reload=AffinityProfile.load(path)
        assert np.allclose(m.predict(pfeat,bits),reload.predict(pfeat,bits),atol=1e-10,rtol=0)
        full[mode]={'choice':ch,'checkpoint':path.name,'checkpoint_sha256':digest(path)}
    result['full_train_profile_models']=full;result['full_train_inner_grid']=grid
    # Diagnostic only on previously viewed eight-target panel, after freezing all models.
    dev=[r for r in rows if r['split']=='val'];dpids=sorted({r['sequence'] for r in dev})
    assert len(dpids)==8
    dy=np.array([[next(r['y'] for r in dev if r['sequence']==p and r['compound']==c) for c in cids] for p in dpids])
    dfeat=[]
    for p in dpids:
        obj=torch.load(OLD/'data/esm2_t33_650M_layer33_w1000_o128_fp16'/(p+'.pt'),map_location='cpu',weights_only=True)
        assert obj['sequence_hash']==p
        dfeat.append(obj['embeddings'].float().mean(0).numpy())
    dfeat=np.array(dfeat)
    result['previously_inspected_dev']={'compound_mean':metric(np.broadcast_to(y.mean(0),dy.shape),dy)}
    for mode in full:
        m=AffinityProfile.load(OUT/full[mode]['checkpoint'])
        result['previously_inspected_dev'][mode]=metric(m.predict(dfeat,bits),dy)
    result['elapsed_seconds']=time.monotonic()-start
    dump('study_results.json',result)
    print(json.dumps({'status':'STUDY_COMPLETE','elapsed_seconds':result['elapsed_seconds'],
        'historical_dev':result['previously_inspected_dev']}),flush=True)


if __name__=='__main__':
    with gpu_lease(OLD):main()
