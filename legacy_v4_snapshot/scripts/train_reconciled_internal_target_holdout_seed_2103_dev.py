"""Train fresh sequence model on 7 DAVIS training targets; test 2 unseen training targets.
Exploratory internal target holdout, not an external independent benchmark.
No existing validation targets used for training or model selection.
"""
import sys,json,random,math,os,time,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from chemprop.features import mol2graph
from chemprop.models.model import InteractionModel
from dynacpi.sequence_model import SequenceFoundation
from tape import TAPETokenizer
from gpu_lease import gpu_lease

def main(kind):
 # Predeclared target-disjoint stress test, selected for numeric coverage; DEV only.
 holdout_targets={"CLK1","ROCK2"}
 torch.manual_seed(2103);np.random.seed(2103);random.seed(2103)
 source=root/"data"/"engineering_pilot_primary_reconciled_20261107";stage=root/"data"/"staging";cache=root/"data"/"esm2_t33_650M_layer33_w1000_o128_fp16"
 out=root/"research"/("reconciled_internal_target_holdout_"+kind+"_seed_2103_dev");out.mkdir(exist_ok=False)
 rows=json.loads((source/"rows.json").read_text())
 t=[r for r in rows if r["split"]=="train" and r["primary_label_kind"]=="numeric_Kd" and r["primary_protein"] not in holdout_targets];v=[r for r in rows if r["split"]=="train" and r["primary_label_kind"]=="numeric_Kd" and r["primary_protein"] in holdout_targets]
 assert len(t)==68 and len(v)==38 and {r["sequence"] for r in t}.isdisjoint({r["sequence"] for r in v})
 print(json.dumps({"experiment":"INTERNAL_TARGET_HOLDOUT_DEV","train_pairs":len(t),"heldout_pairs":len(v),"heldout_targets":sorted(holdout_targets)}),flush=True)
 mean=np.mean([r["y"] for r in t]);std=np.std([r["y"] for r in t])
 smiles=json.loads((stage/"compounds.json").read_text());seqs=json.loads((stage/"sequences.json").read_text())
 pack=np.load(stage/"morgan_radius2_2048_chiral.npz");fp=dict(zip(pack["ids"],pack["bits"]))
 seq_ids=set(r["sequence"] for r in rows);tok=TAPETokenizer(vocab="unirep")
 if kind=="perceiver":
  args=SimpleNamespace(dataset_type="regression",vocab_size=31,prot_hidden=128,sequence_length=500,prot_1d_out=64,kernel_size=7,prot_1dcnn_num=3,hidden_size=300,alpha=.5,dropout=.1,num_tasks=1,checkpoint_frzn=None,features_only=False,number_of_molecules=1,use_input_features=False,atom_descriptors=None,ffn_num_layers=2,ffn_hidden_size=300,activation="ReLU",overwrite_default_atom_features=False,overwrite_default_bond_features=False,device=torch.device("cuda"),mpn_shared=False,atom_messages=False,bias=False,depth=3,undirected=False,aggregation="mean",aggregation_norm=100)
  model=InteractionModel(args).cuda()
  enc={p:torch.tensor((list(tok.encode(list(seqs[p])))+[0]*500)[:500],dtype=torch.long) for p in seq_ids}
 else:
  model=SequenceFoundation().cuda()
  enc={}
  for p in seq_ids:
   obj=torch.load(cache/(p+".pt"),map_location="cpu",weights_only=True)
   assert obj["sequence_hash"]==p and obj["embeddings"].shape==(len(seqs[p]),1280)
   enc[p]=obj["embeddings"]
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
 def forward(rr):
  sm=[smiles[r["compound"]] for r in rr];f=torch.tensor(np.stack([fp[r["compound"]] for r in rr]),device="cuda",dtype=torch.float32)
  ee=[enc[r["sequence"]] for r in rr]
  if kind=="perceiver":return model([mol2graph(sm)],torch.stack(ee).cuda(),f).flatten()
  x=pad_sequence(ee,batch_first=True).cuda()
  mask=torch.arange(x.size(1),device="cuda")[None,:]<torch.tensor([len(e) for e in ee],device="cuda")[:,None]
  return model(sm,f,x,mask)
 log=[]
 for epoch in range(1,5):
  start=time.time();model.train();order=np.random.default_rng(2103+epoch).permutation(len(t));losses=[]
  for off in range(0,len(t),64):
   ids=order[off:off+64];opt.zero_grad(set_to_none=True)
   for j in range(0,len(ids),8):
    rr=[t[i] for i in ids[j:j+8]]
    y=torch.tensor([(r["y"]-mean)/std for r in rr],device="cuda",dtype=torch.float32)
    loss=(forward(rr)-y).square().sum()/len(ids)
    if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
    loss.backward();losses.append(float(loss.detach()))
   nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  model.eval();pred=[]
  with torch.inference_mode():
   for j in range(0,len(v),8):pred.extend((forward(v[j:j+8])*std+mean).tolist())
  y=np.array([r["y"] for r in v]);errs=(np.array(pred)-y)**2
  macro=np.mean([math.sqrt(np.mean([e for e,r in zip(errs,v) if r["sequence"]==p])) for p in sorted({r["sequence"] for r in v})])
  item={"epoch":epoch,"pooled_rmse":float(np.sqrt(errs.mean())),"target_macro_rmse":float(macro),"elapsed_s":round(time.time()-start,2),"peak_MiB":round(torch.cuda.max_memory_allocated()/2**20,1)}
  log.append(item);print(json.dumps(item),flush=True)
  (out/"history.json").write_text(json.dumps(log,indent=2))
  tmp=out/"checkpoint.tmp";torch.save({"state":model.state_dict(),"epoch":epoch,"label_mean":float(mean),"label_std":float(std),"row_sha256":hashlib.sha256((source/"rows.json").read_bytes()).hexdigest(),"kind":kind,"evaluation":"INTERNAL_TARGET_HOLDOUT_DEV_NOT_INDEPENDENT_EXTERNAL_TEST","heldout_targets":sorted(holdout_targets),"train_numeric_pairs":len(t),"heldout_numeric_pairs":len(v)},tmp);os.replace(tmp,out/"checkpoint.pt")
  (out/"val_predictions.json").write_text(json.dumps([dict(r,pred=yhat) for r,yhat in zip(v,pred)]))
if __name__=="__main__":
 if sys.argv[1] not in ("sequence",):raise ValueError("unknown model")
 with gpu_lease(root):main(sys.argv[1])


