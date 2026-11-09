"""Fresh DeepDTA CNN neural baseline on frozen public BALM cold-target DEV.

Only train/inner-dev labels, duplicate train averaging, train-only vocab
and normalization, fixed 10 epochs, no prospective test evaluation.
"""
import json,time,random
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from dynacpi.deepdta_reference_dev import DeepDTA
from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import collapse_training

OUT=ROOT/"research/balm_bindingdb_deepdta_10epoch_dev.json"
PROGRESS=ROOT/"research/balm_bindingdb_deepdta_training_progress_LOCAL.json"

def main():
 seed=2103
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 dataset=get_train_dev()
 train=collapse_training(dataset["train"])
 val=dataset["inner_dev"]
 assert len(train)==12315 and len(val)==3891
 assert {r["group"] for r in train}.isdisjoint({r["group"] for r in val})
 dchars=sorted(set("".join(r["smiles"] for r in train)))
 pchars=sorted(set("".join(r["protein"] for r in train)))
 dmap={ch:i+1 for i,ch in enumerate(dchars)}
 pmap={ch:i+1 for i,ch in enumerate(pchars)}
 assert len(dchars)<128 and len(pchars)<32
 groups=sorted({r["group"] for r in val})
 gmap={g:i for i,g in enumerate(groups)}
 mu=float(np.mean([r["label"] for r in train]))
 sd=float(np.std([r["label"] for r in train]));assert sd>0
 def convert(rows):
  D=np.zeros((len(rows),100),dtype=np.int64)
  P=np.zeros((len(rows),1000),dtype=np.int64)
  Y=np.asarray([r["label"] for r in rows],dtype=np.float32)
  G=np.full(len(rows),-1,dtype=np.int64)
  for i,r in enumerate(rows):
   for j,c in enumerate(r["smiles"][:100]):D[i,j]=dmap.get(c,0)
   for j,c in enumerate(r["protein"][:1000]):P[i,j]=pmap.get(c,0)
   if r["group"] in gmap:G[i]=gmap[r["group"]]
  return TensorDataset(torch.from_numpy(D),torch.from_numpy(P),torch.from_numpy(Y),
                       torch.from_numpy(G))
 train_dataset=convert(train)
 valid_dataset=convert(val)
 dev="cuda" if torch.cuda.is_available() else "cpu"
 assert dev=="cuda"
 model=DeepDTA().to(dev)
 optim=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
 loader=DataLoader(train_dataset,batch_size=128,shuffle=True,pin_memory=True)
 vloader=DataLoader(valid_dataset,batch_size=128,pin_memory=True)
 info={"train_rows_unique":len(train),"validation_rows":len(val),
       "train_protein_components":len({r["group"] for r in train}),
       "dev_protein_components":len(groups),
       "model":"DeepDTA-inspired dual CNN (not source paper full training)",
       "parameters":sum(p.numel() for p in model.parameters()),
       "device":dev,"seed":seed,"epochs":10,"batch_size":128,
       "lr":.0003,"optimizer":"AdamW",
       "weight_decay":.0001,"gradient_clipping_norm":1.,
       "max_drug_tokens":100,"max_protein_tokens":1000,
       "train_only_vocab":True,
       "train_only_target_standardization":True}
 print(json.dumps(info),flush=True)
 history=[]
 for epoch in range(1,11):
  start=time.time()
  model.train();losses=[]
  for d,p,y,_ in loader:
   d=d.to(dev);p=p.to(dev);y=y.to(dev)
   optim.zero_grad(set_to_none=True)
   output=model(d,p)
   loss=(output-(y-mu)/sd).square().mean()
   if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
   loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   optim.step()
   losses.append(float(loss.item()))
  model.eval();sse=0.;n=0
  bygroup=defaultdict(lambda:[0.,0])
  with torch.inference_mode():
   for d,p,y,g in vloader:
    pred=model(d.to(dev),p.to(dev)).cpu().numpy()*sd+mu
    true=y.numpy()
    err=(pred-true)**2
    sse+=float(np.sum(err));n+=len(y)
    for index,e in zip(g.numpy(),err):
     bygroup[int(index)][0]+=float(e)
     bygroup[int(index)][1]+=1
  entry={"epoch":epoch,"train_mse_normalized":float(np.mean(losses)),
         "pooled_dev_rmse":float(np.sqrt(sse/n)),
         "macro_group_dev_rmse":float(np.mean(
           [np.sqrt(s/c) for s,c in bygroup.values()])),
         "seconds":round(time.time()-start,2)}
  history.append(entry)
  PROGRESS.write_text(json.dumps({"info":info,"history":history,
                                  "active":epoch<10},indent=2))
  print(json.dumps(entry),flush=True)
 ydev=np.asarray([r["label"] for r in val])
 globalmean=float(np.sqrt(np.mean((ydev-mu)**2)))
 grouped=defaultdict(list)
 for r in val:grouped[r["group"]].append(r["label"])
 macro=float(np.mean([np.sqrt(np.mean((np.asarray(a)-mu)**2))
                      for a in grouped.values()]))
 report={"status":"EXPLORATORY_BALM_DEEPDTA_INSPIRED_10EPOCH_DEV_ONLY",
         **info,"history":history,
         "mean_baseline_pooled_rmse":globalmean,
         "mean_baseline_macro_rmse":macro,
         "warnings":["This short run does not reproduce the full original DeepDTA training schedule.",
                     "Protein input max 1000 vs Perceiver 500 and Ridge untruncated; different input contexts.",
                     "Only one seed; needs replication and matched training budget.",
                     "Reserved test not accessed or scored."]}
 OUT.write_text(json.dumps(report,indent=2))
 print("SAVED "+str(OUT),flush=True)

if __name__=="__main__":main()
