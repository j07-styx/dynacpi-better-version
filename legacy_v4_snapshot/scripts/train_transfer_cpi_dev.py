"""Fresh, common-schedule TransferCPI/chemical/upstream PerceiverCPI comparison."""
import argparse,json,sys,time,hashlib,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.transfer_data import load_development
from dynacpi.provisional_transfer_training_lock import refuse_provisional_transfer_retraining
from dynacpi.transfer_batch import TransferCollator
from dynacpi.transfer_cpi import TransferCPI
from dynacpi.dual_transfer_cpi import DualTransferCPI
from gpu_lease import gpu_lease

def metrics(rows,pred):
    y=np.array([r["y"] for r in rows]);err=(pred-y)**2
    groups=sorted({r["component"] for r in rows})
    per={g:float(np.sqrt(np.mean(err[[r["component"]==g for r in rows]]))) for g in groups}
    return {"pooled_rmse":float(np.sqrt(err.mean())),"component_macro_rmse":float(np.mean(list(per.values()))),"component_rmse":per}
def main(args):
    # Retired provisional protocol: guard before any affinity-label read or CUDA work.
    refuse_provisional_transfer_retraining(ROOT)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    assert torch.cuda.is_available()
    rows,seq,split_hash=load_development()
    train=[r for r in rows if r["split"]=="train"];dev=[r for r in rows if r["split"]=="dev"]
    out=ROOT/"research/transfer_cpi_dev_20261108"/f"{args.model}_seed{args.seed}"
    out.mkdir(parents=True,exist_ok=False)
    mean=float(np.mean([r["y"] for r in train]));std=float(np.std([r["y"] for r in train]));assert std>0
    complete=json.loads((ROOT/"data/transfer_esm64_dev/COMPLETE.json").read_text())
    assert complete["split_sha256"]==split_hash
    cache=torch.load(ROOT/"data/transfer_compounds_dev.pt",map_location="cpu",weights_only=True)
    assert cache["meta"]["split_sha256"]==split_hash
    if args.model.endswith("_clm"):
        language=torch.load(ROOT/"data/transfer_chemberta_dev.pt",map_location="cpu",weights_only=True)
        assert language["meta"]["split_sha256"]==split_hash
        for cid,g in cache["graphs"].items():g["chemical_language"]=language["features"][cid]
    proteins={}
    for pid in seq:
        obj=torch.load(ROOT/"data/transfer_esm64_dev"/(pid+".pt"),map_location="cpu",weights_only=True)
        assert obj["sequence_hash"]==pid
        proteins[pid]=obj["tokens"]
    collate=TransferCollator(cache["graphs"],proteins,cache["property_mean"],cache["property_std"])
    if args.model=="perceiver":
        from chemprop.models.model import InteractionModel
        from chemprop.features import mol2graph
        from tape import TAPETokenizer
        from rdkit import Chem,DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        cfg=SimpleNamespace(dataset_type="regression",vocab_size=31,prot_hidden=128,sequence_length=500,prot_1d_out=64,kernel_size=7,prot_1dcnn_num=3,hidden_size=300,alpha=.5,dropout=.1,num_tasks=1,checkpoint_frzn=None,features_only=False,number_of_molecules=1,use_input_features=False,atom_descriptors=None,ffn_num_layers=2,ffn_hidden_size=300,activation="ReLU",overwrite_default_atom_features=False,overwrite_default_bond_features=False,device=torch.device("cuda"),mpn_shared=False,atom_messages=False,bias=False,depth=3,undirected=False,aggregation="mean",aggregation_norm=100)
        model=InteractionModel(cfg).cuda();tok=TAPETokenizer(vocab="unirep")
        encoded={pid:torch.tensor((list(tok.encode(list(s)))+[0]*500)[:500]) for pid,s in seq.items()}
        fps={};gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        for r in rows:
            if r["compound"] not in fps:
                a=np.zeros(2048,dtype=np.int8);DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(r["smiles"])),a);fps[r["compound"]]=torch.from_numpy(a).float()
        def forward(rr):
            pred=model([mol2graph([r["smiles"] for r in rr])],torch.stack([encoded[r["protein"]] for r in rr]).cuda(),torch.stack([fps[r["compound"]] for r in rr]).cuda()).flatten()
            return pred,torch.zeros_like(pred)
    else:
        model=(DualTransferCPI() if args.model.endswith("_clm") else TransferCPI()).cuda()
        def forward(rr):
            batch={k:v.cuda() for k,v in collate(rr).items()}
            return model(batch,chemical_only=args.model.startswith("chemical"))
    protocol={"status":"PROVISIONAL_FROZEN_INNER_DEV_ONLY","model":args.model,"seed":args.seed,
      "epochs":args.epochs,"batch_size":32,"optimizer":"AdamW","lr":.0003,"weight_decay":.0001,
      "normalization":"train mean/std","sampling":"uniform deterministic permutation each epoch",
      "selection":"minimum inner-dev component-macro RMSE; also report fixed final epoch",
      "train_pairs":len(train),"dev_pairs":len(dev),"split_sha256":split_hash,
      "parameters":sum(p.numel() for p in model.parameters()),"interaction_penalty":.01 if args.model.startswith("transfer") else 0,
      "baseline":metrics(dev,np.full(len(dev),mean)),
      "limitations":["Provisional assay/construct QC","Short training, not faithful paper schedule",
                     "Different representation/pretraining resources are intentional","No prospective test scoring"]}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    optim=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    history=[];best=float("inf")
    print(json.dumps({k:v for k,v in protocol.items() if k!="baseline"}),flush=True)
    for epoch in range(1,args.epochs+1):
        started=time.time();model.train();sumloss=0
        order=np.random.default_rng(args.seed+epoch).permutation(len(train))
        for off in range(0,len(train),32):
            rr=[train[int(i)] for i in order[off:off+32]]
            y=torch.tensor([(r["y"]-mean)/std for r in rr],device="cuda")
            optim.zero_grad(set_to_none=True);pred,residual=forward(rr)
            loss=(pred-y).square().mean()+(.01*residual.square().mean() if args.model.startswith("transfer") else 0)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optim.step()
            sumloss+=float(loss.detach())*len(rr)
        model.eval();predictions=[]
        with torch.inference_mode():
            for off in range(0,len(dev),32):
                pred,_=forward(dev[off:off+32]);predictions.extend((pred*std+mean).cpu().tolist())
        predictions=np.array(predictions);assert np.isfinite(predictions).all()
        score=metrics(dev,predictions)
        record={"epoch":epoch,"train_objective":sumloss/len(train),"seconds":round(time.time()-started,2),**score}
        history.append(record);(out/"history.json").write_text(json.dumps(history,indent=2))
        print(json.dumps({k:v for k,v in record.items() if k!="component_rmse"}),flush=True)
        state={"state":model.state_dict(),"epoch":epoch,"label_mean":mean,"label_std":std,"protocol":protocol}
        torch.save(state,out/"last.pt")
        payload=[{"component":r["component"],"protein":r["protein"],"compound":r["compound"],"y":r["y"],"pred":float(p)} for r,p in zip(dev,predictions)]
        (out/"last_predictions_LOCAL.json").write_text(json.dumps(payload))
        if score["component_macro_rmse"]<best:
            best=score["component_macro_rmse"];torch.save(state,out/"best.pt")
            (out/"best_predictions_LOCAL.json").write_text(json.dumps(payload))
    (out/"COMPLETE.json").write_text(json.dumps({"best_component_macro_rmse":best,"epochs":args.epochs,"status":"COMPLETE_NO_TEST_EVALUATION"},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",choices=["transfer","chemical","perceiver","transfer_clm","chemical_clm"],required=True);p.add_argument("--seed",type=int,default=2601);p.add_argument("--epochs",type=int,default=5);args=p.parse_args()
    # Reject the compromised provisional source split before leasing the GPU.
    refuse_provisional_transfer_retraining(ROOT)
    with gpu_lease(ROOT):main(args)
