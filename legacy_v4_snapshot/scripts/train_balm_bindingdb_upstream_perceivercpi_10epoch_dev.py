"""Upstream PerceiverCPI InteractionModel on frozen public BALM development split.

Fixed 10-epoch schedule, one seed, no test labels; write progress every epoch.
"""
import json,hashlib,random,time
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
from train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import collapse_training

OUT=ROOT/"research/balm_bindingdb_perceivercpi_upstream_10epoch_dev.json"
PROGRESS=ROOT/"research/balm_bindingdb_perceivercpi_upstream_progress_LOCAL.json"
RDLogger.DisableLog("rdApp.*")

def main():
    seed=2103
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    parts=get_train_dev()
    train=collapse_training(parts["train"])
    val=parts["inner_dev"]
    assert len(train)==12315 and len(val)==3891
    assert {r["group"] for r in train}.isdisjoint({r["group"] for r in val})
    needed={r["protein"] for r in train+val}
    tok=TAPETokenizer(vocab="unirep")
    protein={k:torch.tensor((list(tok.encode(list(k)))+[0]*500)[:500],dtype=torch.long)
             for k in needed}
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    smiles={r["smiles"] for r in train+val}
    fp={}
    for sm in smiles:
        mol=Chem.MolFromSmiles(sm)
        assert mol is not None
        a=np.zeros(2048,dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol),a)
        fp[sm]=torch.tensor(a,dtype=torch.float32)
    device="cuda" if torch.cuda.is_available() else "cpu"
    assert device=="cuda"
    args=SimpleNamespace(
       dataset_type="regression",vocab_size=31,prot_hidden=128,
       sequence_length=500,prot_1d_out=64,kernel_size=7,prot_1dcnn_num=3,
       hidden_size=300,alpha=.5,dropout=.1,num_tasks=1,checkpoint_frzn=None,
       features_only=False,number_of_molecules=1,use_input_features=False,
       atom_descriptors=None,ffn_num_layers=2,ffn_hidden_size=300,
       activation="ReLU",overwrite_default_atom_features=False,
       overwrite_default_bond_features=False,device=torch.device(device),
       mpn_shared=False,atom_messages=False,bias=False,depth=3,
       undirected=False,aggregation="mean",aggregation_norm=100)
    model=InteractionModel(args).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    ytrain=np.asarray([r["label"] for r in train])
    mu=float(ytrain.mean());sd=float(ytrain.std());assert sd>0
    def forward(batch):
        graph=mol2graph([r["smiles"] for r in batch])
        p=torch.stack([protein[r["protein"]] for r in batch]).to(device)
        f=torch.stack([fp[r["smiles"]] for r in batch]).to(device)
        return model([graph],p,f).flatten()
    info={"train_rows":len(train),"validation_rows":len(val),
          "training_protein_groups":len({r["group"] for r in train}),
          "dev_protein_groups":len({r["group"] for r in val}),
          "upstream_model":"PerceiverCPI InteractionModel",
          "params":sum(p.numel() for p in model.parameters()),
          "seed":seed,"device":device,"batch":32,"lr":.0003,
          "train_only_label_standardization":True,"max_protein_tokens":500,
          "protein_tokenizer":"TAPE unirep, original architecture convention",
          "fixed_epochs":10}
    print(json.dumps(info),flush=True)
    history=[]
    ydev=np.asarray([r["label"] for r in val])
    trainmean_rmse=float(np.sqrt(np.mean((ydev-mu)**2)))
    group_values=defaultdict(list)
    for r in val:group_values[r["group"]].append(r["label"])
    baseline_macro=float(np.mean([np.sqrt(np.mean((np.asarray(v)-mu)**2))
                   for v in group_values.values()]))
    for epoch in range(1,11):
        start=time.time()
        model.train();losses=[]
        order=np.random.default_rng(seed+epoch).permutation(len(train))
        for at in range(0,len(train),32):
            batch=[train[int(i)] for i in order[at:at+32]]
            y=torch.tensor([(r["label"]-mu)/sd for r in batch],
                           dtype=torch.float32,device=device)
            opt.zero_grad(set_to_none=True)
            pred=forward(batch)
            loss=(pred-y).square().mean()
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            losses.append(float(loss.item()))
        model.eval();sq=0.;n=0
        within=defaultdict(lambda:[0.,0])
        with torch.inference_mode():
            for at in range(0,len(val),32):
                batch=val[at:at+32]
                pred=forward(batch).cpu().numpy()*sd+mu
                for row,p in zip(batch,pred):
                    e=float((float(p)-row["label"])**2)
                    sq+=e;n+=1
                    within[row["group"]][0]+=e
                    within[row["group"]][1]+=1
        entry={"epoch":epoch,"train_mse_normalized":float(np.mean(losses)),
               "pooled_dev_rmse":float(np.sqrt(sq/n)),
               "macro_protein_group_dev_rmse":float(np.mean(
                  [np.sqrt(v[0]/v[1]) for v in within.values()])),
               "seconds":round(time.time()-start,2)}
        history.append(entry)
        PROGRESS.write_text(json.dumps({"setup":info,"history":history,
                           "still_running":epoch<10},indent=2))
        print(json.dumps(entry),flush=True)
    result={"status":"EXPLORATORY_PUBLIC_BALM_COLD_TARGET_INNER_DEV_ONLY_NO_TEST",
            **info,"history":history,
            "train_mean_dev_rmse":trainmean_rmse,
            "train_mean_dev_macro_rmse":baseline_macro,
            "method_note":"Upstream PerceiverCPI InteractionModel with fixed ten-epoch training; not exact paper setup, different compute budget than ridge and no seeds beyond 2103.",
            "warnings":["No reserved-test labels or predictions accessed.",
                        "Protein trigram similarity graph split is not MMseqs2 identity clustering.",
                        "Assay provenance is absent from BALM source Arrow and may create label ambiguity.",
                        "10 epochs may underfit the Perceiver architecture."]}
    OUT.write_text(json.dumps(result,indent=2))
    print("SAVED "+str(OUT),flush=True)

if __name__=="__main__":main()
