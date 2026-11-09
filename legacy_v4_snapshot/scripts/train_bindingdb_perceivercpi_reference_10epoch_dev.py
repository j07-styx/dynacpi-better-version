"""DeepDTA dual CNN training on provisional BindingDB component-disjoint DEV split."""
import json,hashlib,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from types import SimpleNamespace
from chemprop.models.model import InteractionModel
from chemprop.features import mol2graph
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from tape import TAPETokenizer
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
 tok=TAPETokenizer(vocab="unirep")
 protein={k:torch.tensor((list(tok.encode(list(v)))+[0]*500)[:500],dtype=torch.long) for k,v in seq.items()}
 fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
 RDLogger.DisableLog('rdApp.*')
 smiles={x['compound_sha256']:x['smiles_isomeric'] for x in samples}
 fp={}
 for key,sm in smiles.items():
  mol=Chem.MolFromSmiles(sm)
  if mol is None:raise ValueError('invalid canonical SMILES '+key)
  a=np.zeros(2048,dtype=np.int8);DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol),a)
  fp[key]=torch.tensor(a,dtype=torch.float32)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 assert device=='cuda'
 args=SimpleNamespace(dataset_type="regression",vocab_size=31,prot_hidden=128,sequence_length=500,prot_1d_out=64,kernel_size=7,prot_1dcnn_num=3,hidden_size=300,alpha=.5,dropout=.1,num_tasks=1,checkpoint_frzn=None,features_only=False,number_of_molecules=1,use_input_features=False,atom_descriptors=None,ffn_num_layers=2,ffn_hidden_size=300,activation="ReLU",overwrite_default_atom_features=False,overwrite_default_bond_features=False,device=torch.device(device),mpn_shared=False,atom_messages=False,bias=False,depth=3,undirected=False,aggregation="mean",aggregation_norm=100)
 model=InteractionModel(args).to(device);optim=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
 mean=np.mean([x['median_pKd'] for x in train]);std=np.std([x['median_pKd'] for x in train]);assert std>0
 def forward(batch):
  keys=[x['compound_sha256'] for x in batch]
  graph=mol2graph([smiles[k] for k in keys])
  p=torch.stack([protein[x['protein_seq_sha256']] for x in batch]).to(device)
  f=torch.stack([fp[k] for k in keys]).to(device)
  return model([graph],p,f).flatten()
 print(json.dumps({'train':len(train),'val':len(valid),'train_components':len(groups)-len(val),'val_components':len(val),'device':device,'parameters':sum(p.numel() for p in model.parameters()),'model':'upstream PerceiverCPI InteractionModel'}),flush=True)
 history=[]
 for epoch in range(1,11):
  model.train();losses=[];start=time.time()
  order=np.random.default_rng(2103+epoch).permutation(len(train))
  for offset in range(0,len(train),32):
   batch=[train[int(i)] for i in order[offset:offset+32]]
   y=torch.tensor([(x['median_pKd']-mean)/std for x in batch],dtype=torch.float32,device=device)
   optim.zero_grad(set_to_none=True)
   pred=forward(batch);loss=(pred-y).square().mean()
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();losses.append(loss.item())
  model.eval();sq=0;n=0
  with torch.inference_mode():
   for offset in range(0,len(valid),32):
    batch=valid[offset:offset+32];pred=forward(batch).cpu().numpy()*std+mean
    y=np.array([x['median_pKd'] for x in batch]);sq+=float(np.sum((pred-y)**2));n+=len(batch)
  entry={'epoch':epoch,'train_mse_normalized':float(np.mean(losses)),'val_rmse':float(np.sqrt(sq/n)),'seconds':round(time.time()-start,2)};history.append(entry);print(json.dumps(entry),flush=True)
 result={'status':'PERCEIVERCPI_UPSTREAM_DEV_NOT_PAPER_BENCHMARK','dataset':'BindingDB preliminary exact-looking Kd; assay and construct QC incomplete','train_pairs':len(train),'validation_pairs':len(valid),'protein_cluster_disjoint':True,'seed':2103,'epochs':10,'batch_size':32,'lr':0.0003,'protein_length':500,'note':'Upstream InteractionModel, short matched-split feasibility run, different training schedule from DeepDTA; not a fair full-paper superiority comparison','history':history,'train_only_mean_baseline_rmse':float(np.sqrt(np.mean((np.array([x['median_pKd'] for x in valid])-np.mean([x['median_pKd'] for x in train]))**2)))}
 (ROOT/'research/bindingdb_perceivercpi_reference_10epoch_dev.json').write_text(json.dumps(result,indent=2));print('SAVED',flush=True)
if __name__=='__main__':main()
