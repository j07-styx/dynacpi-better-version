"""Matched DeepDTA negative control: protein tokens randomly reassigned to rows."""
import json,hashlib,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from dynacpi.deepdta_reference_dev import DeepDTA
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
 groups=sorted({comp[x['protein_seq_sha256']] for x in samples});val=set(sorted(groups,key=lambda x:hashlib.sha256(('bindingdb-dev-20261107:'+x).encode()).hexdigest())[:len(groups)//5])
 train=[x for x in samples if comp[x['protein_seq_sha256']] not in val];valid=[x for x in samples if comp[x['protein_seq_sha256']] in val]
 needed={x['protein_seq_sha256'] for x in samples};seq={}
 for x in rows(A/'bindingdb_full_kd_candidates.jsonl'):
  s=x['sequence'];key=hashlib.sha256(s.encode()).hexdigest()
  if key in needed:seq[key]=s
  if len(seq)==len(needed):break
 assert len(seq)==len(needed)
 drug_chars=sorted(set(''.join(x['smiles_isomeric'] for x in samples)));prot_chars=sorted(set(''.join(seq.values())))
 assert len(drug_chars)<128 and len(prot_chars)<32
 dmap={c:i+1 for i,c in enumerate(drug_chars)};pmap={c:i+1 for i,c in enumerate(prot_chars)}
 def tensorize(items):
  d=np.zeros((len(items),100),dtype=np.int64);p=np.zeros((len(items),1000),dtype=np.int64);y=np.empty(len(items),dtype=np.float32)
  for i,x in enumerate(items):
   for j,c in enumerate(x['smiles_isomeric'][:100]):d[i,j]=dmap[c]
   for j,c in enumerate(seq[x['protein_seq_sha256']][:1000]):p[i,j]=pmap[c]
   y[i]=x['median_pKd']
  return TensorDataset(torch.from_numpy(d),torch.from_numpy(p),torch.from_numpy(y))
 tr=tensorize(train);va=tensorize(valid)
 # Negative control: shuffle protein encodings independently in train and val,
 # retaining compounds, affinity labels, and protein-token marginal distribution.
 # Deterministic independent permutation, no heldout label access.
 g=torch.Generator().manual_seed(991103)
 def shuffle_proteins(dataset):
  d,p,y=dataset.tensors
  return TensorDataset(d,p[torch.randperm(len(p),generator=g)],y)
 tr=shuffle_proteins(tr);va=shuffle_proteins(va)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 model=DeepDTA().to(device);optim=torch.optim.Adam(model.parameters(),lr=.001);loader=DataLoader(tr,batch_size=128,shuffle=True,pin_memory=True);vloader=DataLoader(va,batch_size=128,pin_memory=True)
 print(json.dumps({'train':len(train),'val':len(valid),'train_components':len(groups)-len(val),'val_components':len(val),'device':device,'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
 history=[]
 for epoch in range(1,11):
  model.train();losses=[];start=time.time()
  for d,p,y in loader:
   d,p,y=d.to(device),p.to(device),y.to(device);optim.zero_grad(set_to_none=True);out=model(d,p);loss=(out-y).square().mean();loss.backward();optim.step();losses.append(loss.item())
  model.eval();sq=0;n=0
  with torch.no_grad():
   for d,p,y in vloader:
    pred=model(d.to(device),p.to(device)).cpu();sq+=(pred-y).square().sum().item();n+=len(y)
  entry={'epoch':epoch,'train_mse':float(np.mean(losses)),'val_rmse':float(np.sqrt(sq/n)),'seconds':round(time.time()-start,2)};history.append(entry);print(json.dumps(entry),flush=True)
 result={'status':'DEEPDTA_PROTEIN_SHUFFLE_NEGATIVE_CONTROL_DEV','dataset':'BindingDB preliminary exact-looking Kd; assay and construct QC incomplete','train_pairs':len(train),'validation_pairs':len(valid),'protein_cluster_disjoint':True,'protein_inputs_shuffled':True,'shuffle_seed':991103,'seed':2103,'epochs':10,'batch_size':128,'lr':.001,'max_smiles_len':100,'max_protein_len':1000,'source_paper_batch_size':256,'source_paper_epochs':100,'note':'Short training feasibility experiment, not faithful full paper training or independent test; all epoch validation metrics exploratory','history':history,'train_only_mean_baseline_rmse':float(np.sqrt(np.mean((np.array([x['median_pKd'] for x in valid])-np.mean([x['median_pKd'] for x in train]))**2)))}
 (ROOT/'research/bindingdb_deepdta_protein_shuffle_10epoch_dev.json').write_text(json.dumps(result,indent=2));print('SAVED',flush=True)
if __name__=='__main__':main()
