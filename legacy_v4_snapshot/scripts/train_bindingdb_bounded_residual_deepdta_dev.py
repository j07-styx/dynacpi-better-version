"""DeepDTA dual CNN training on provisional BindingDB component-disjoint DEV split."""
import json,hashlib,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from dynacpi.bounded_residual_deepdta_dev import ResidualDeepDTA
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data/audit'
def rows(p):
 with p.open(encoding='utf8') as f:
  for line in f:yield json.loads(line)
def main():
 torch.manual_seed(2103);np.random.seed(2103);random.seed(2103)
 comp={}
 for g in rows(A/'bindingdb_full_kd_protein_components.jsonl'):
  for s in g['members']:comp[s]=g['component_id']
 tasks={(x['sequence_sha256'],x['assay_key']) for x in rows(A/'bindingdb_full_kd_task_summary.jsonl') if x['unique_canonical_compounds']>=15}
 samples=[x for x in rows(A/'bindingdb_full_kd_compound_median_candidates.jsonl') if (x['protein_seq_sha256'],x['assay_key']) in tasks and x['range_pKd']<=.1]
 manifest=json.loads((ROOT/'research/bindingdb_frozen_component_split_manifest_dev.json').read_text())
 tr_groups=set(manifest['train_components']);val=set(manifest['inner_dev_components']);test=set(manifest['test_components'])
 assert tr_groups.isdisjoint(val|test) and val.isdisjoint(test)
 train=[x for x in samples if comp[x['protein_seq_sha256']] in tr_groups]
 valid=[x for x in samples if comp[x['protein_seq_sha256']] in val]
 assert len(train)==manifest['train_pairs'] and len(valid)==manifest['inner_dev_pairs']
 needed={x['protein_seq_sha256'] for x in samples};seq={}
 for x in rows(A/'bindingdb_full_kd_candidates.jsonl'):
  s=x['sequence'];key=hashlib.sha256(s.encode()).hexdigest()
  if key in needed:seq[key]=s
  if len(seq)==len(needed):break
 assert len(seq)==len(needed)
 drug_chars=sorted(set(''.join(x['smiles_isomeric'] for x in train)));prot_chars=sorted(set(''.join(seq[x['protein_seq_sha256']] for x in train)))
 assert len(drug_chars)<128 and len(prot_chars)<32
 dmap={c:i+1 for i,c in enumerate(drug_chars)};pmap={c:i+1 for i,c in enumerate(prot_chars)}
 def tensorize(items):
  d=np.zeros((len(items),100),dtype=np.int64);p=np.zeros((len(items),1000),dtype=np.int64);y=np.empty(len(items),dtype=np.float32)
  for i,x in enumerate(items):
   for j,c in enumerate(x['smiles_isomeric'][:100]):d[i,j]=dmap.get(c,0)
   for j,c in enumerate(seq[x['protein_seq_sha256']][:1000]):p[i,j]=pmap.get(c,0)
   y[i]=x['median_pKd']
  return TensorDataset(torch.from_numpy(d),torch.from_numpy(p),torch.from_numpy(y))
 tr=tensorize(train);va=tensorize(valid);counts={};[counts.__setitem__(comp[x["protein_seq_sha256"]],counts.get(comp[x["protein_seq_sha256"]],0)+1) for x in train];sample_weights=torch.tensor([counts[comp[x["protein_seq_sha256"]]]**(-0.5) for x in train],dtype=torch.double);sampler=WeightedRandomSampler(sample_weights,num_samples=len(train),replacement=True,generator=torch.Generator().manual_seed(2103));device='cuda' if torch.cuda.is_available() else 'cpu'
 model=ResidualDeepDTA().to(device);optim=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);loader=DataLoader(tr,batch_size=128,sampler=sampler,pin_memory=True);vloader=DataLoader(va,batch_size=128,pin_memory=True)
 print(json.dumps({'train':len(train),'val':len(valid),'train_components':len(tr_groups),'val_components':len(val),'device':device,'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
 mean=float(np.mean([x['median_pKd'] for x in train]));std=float(np.std([x['median_pKd'] for x in train]));assert std>0
 history=[]
 for epoch in range(1,11):
  model.train();losses=[];start=time.time()
  for d,p,y in loader:
   d,p,y=d.to(device),p.to(device),y.to(device);optim.zero_grad(set_to_none=True);out=model(d,p);loss=(out-(y-mean)/std).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();losses.append(loss.item())
  model.eval();sq=0;n=0
  with torch.no_grad():
   for d,p,y in vloader:
    pred=model(d.to(device),p.to(device)).cpu()*std+mean;sq+=(pred-y).square().sum().item();n+=len(y)
  entry={'epoch':epoch,'train_mse':float(np.mean(losses)),'val_rmse':float(np.sqrt(sq/n)),'seconds':round(time.time()-start,2)};history.append(entry);print(json.dumps(entry),flush=True)
 result={'status':'FROZEN_INNER_DEV_ONLY_NO_TEST_EVALUATION','dataset':'BindingDB preliminary exact-looking Kd; assay and construct QC incomplete','train_pairs':len(train),'reserved_test_pairs':manifest['test_pairs'],'validation_pairs':len(valid),'protein_cluster_disjoint':True,'seed':2103,'epochs':10,'batch_size':128,'lr':.0003,'optimizer':'AdamW','architecture':'bounded residual dual CNN compact head tanh amplitude 0.5 normalized targets','train_sampling':'inverse_sqrt_protein_component_frequency_replacement','target_normalization':'train_only_mean_std','gradient_clip_norm':1.0,'max_smiles_len':100,'max_protein_len':1000,'source_paper_batch_size':256,'source_paper_epochs':100,'note':'Train-only vocabulary and label normalization; short training feasibility experiment, not faithful full paper training or independent test; all epoch validation metrics exploratory','history':history,'train_only_mean_baseline_rmse':float(np.sqrt(np.mean((np.array([x['median_pKd'] for x in valid])-np.mean([x['median_pKd'] for x in train]))**2)))}
 (ROOT/'research/bindingdb_bounded_residual_deepdta_10epoch_seed2103.json').write_text(json.dumps(result,indent=2));print('SAVED',flush=True)
if __name__=='__main__':main()


