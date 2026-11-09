"""Matched multi-seed upstream PerceiverCPI baseline on the frozen BALM inner-dev split.

Identical data, model, optimizer and schedule to
train_balm_bindingdb_upstream_perceivercpi_10epoch_dev.py, except:
  (a) seed and epoch count are CLI arguments,
  (b) per-row inner-dev predictions are saved for EVERY epoch, so any epoch rule
      (final epoch, dev-best epoch) can be evaluated later without retraining,
  (c) the run executes behind the shared GPU lease.
Only TRAIN and inner-dev labels are loaded. The reserved test is never touched.
Usage: python -m scripts.train_balm_perceivercpi_multiseed_dev --seed 2601 --epochs 10
"""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from chemprop.models.model import InteractionModel
from chemprop.features import mol2graph
from tape import TAPETokenizer
from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import collapse_training
from scripts.gpu_lease import gpu_lease

OUTDIR=ROOT/"research/balm_perceiver_multiseed_dev_20261108"
RDLogger.DisableLog("rdApp.*")

def row_hash(rows):
    h=hashlib.sha256()
    for r in rows:
        h.update(f"{r['protein']}|{r['smiles']}|{r['label']}\n".encode())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--epochs",type=int,default=10)
    a=ap.parse_args()
    seed,epochs=a.seed,a.epochs
    OUTDIR.mkdir(parents=True,exist_ok=True)
    out=OUTDIR/f"perceiver_seed{seed}_ep{epochs}.json"
    if out.exists():raise FileExistsError(out)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    parts=get_train_dev()
    train=collapse_training(parts["train"])
    val=parts["inner_dev"]
    assert len(train)==12315 and len(val)==3891
    assert {r["group"] for r in train}.isdisjoint({r["group"] for r in val})
    tok=TAPETokenizer(vocab="unirep")
    needed={r["protein"] for r in train+val}
    protein={k:torch.tensor((list(tok.encode(list(k)))+[0]*500)[:500],dtype=torch.long) for k in needed}
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fp={}
    for sm in {r["smiles"] for r in train+val}:
        mol=Chem.MolFromSmiles(sm);assert mol is not None
        arr=np.zeros(2048,dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol),arr)
        fp[sm]=torch.tensor(arr,dtype=torch.float32)
    with gpu_lease(ROOT):
        device="cuda" if torch.cuda.is_available() else "cpu"
        assert device=="cuda"
        args=SimpleNamespace(dataset_type="regression",vocab_size=31,prot_hidden=128,
            sequence_length=500,prot_1d_out=64,kernel_size=7,prot_1dcnn_num=3,hidden_size=300,
            alpha=.5,dropout=.1,num_tasks=1,checkpoint_frzn=None,features_only=False,
            number_of_molecules=1,use_input_features=False,atom_descriptors=None,
            ffn_num_layers=2,ffn_hidden_size=300,activation="ReLU",
            overwrite_default_atom_features=False,overwrite_default_bond_features=False,
            device=torch.device(device),mpn_shared=False,atom_messages=False,bias=False,
            depth=3,undirected=False,aggregation="mean",aggregation_norm=100)
        model=InteractionModel(args).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        ytrain=np.asarray([r["label"] for r in train]);mu=float(ytrain.mean());sd=float(ytrain.std())
        def forward(batch):
            graph=mol2graph([r["smiles"] for r in batch])
            p=torch.stack([protein[r["protein"]] for r in batch]).to(device)
            f=torch.stack([fp[r["smiles"]] for r in batch]).to(device)
            return model([graph],p,f).flatten()
        ydev=np.asarray([r["label"] for r in val]);gdev=[r["group"] for r in val]
        history=[];preds_by_epoch=[]
        for epoch in range(1,epochs+1):
            t0=time.time();model.train();losses=[]
            order=np.random.default_rng(seed+epoch).permutation(len(train))
            for at in range(0,len(train),32):
                batch=[train[int(i)] for i in order[at:at+32]]
                y=torch.tensor([(r["label"]-mu)/sd for r in batch],dtype=torch.float32,device=device)
                opt.zero_grad(set_to_none=True)
                loss=(forward(batch)-y).square().mean()
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();losses.append(float(loss.item()))
            model.eval();pred=[]
            with torch.inference_mode():
                for at in range(0,len(val),32):
                    pred.extend((forward(val[at:at+32]).cpu().numpy()*sd+mu).tolist())
            pred=np.asarray(pred,dtype=np.float64);err=(pred-ydev)**2
            within=defaultdict(list)
            for g,e in zip(gdev,err):within[g].append(e)
            entry={"epoch":epoch,"train_mse_normalized":float(np.mean(losses)),
                   "pooled_dev_rmse":float(np.sqrt(err.mean())),
                   "macro_protein_group_dev_rmse":float(np.mean([np.sqrt(np.mean(v)) for v in within.values()])),
                   "seconds":round(time.time()-t0,2)}
            history.append(entry);preds_by_epoch.append([round(float(x),5) for x in pred])
            print(json.dumps(entry),flush=True)
    result={"status":"BALM_FROZEN_INNER_DEV_MATCHED_PERCEIVER_BASELINE_NO_TEST",
            "seed":seed,"epochs":epochs,"model":"PerceiverCPI InteractionModel (upstream, unchanged)",
            "train_rows":len(train),"dev_rows":len(val),"dev_row_sha256":row_hash(val),
            "history":history,"dev_predictions_by_epoch":preds_by_epoch,
            "note":"Dev rows are in get_train_dev()['inner_dev'] order. Epoch rule must be fixed before comparison."}
    out.write_text(json.dumps(result)+"\n",encoding="utf-8")
    print("SAVED",out,flush=True)

if __name__=="__main__":main()
