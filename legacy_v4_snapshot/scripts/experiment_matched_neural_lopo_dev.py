"""81 matched neural fits; development training data only. CPU avoids active GPU jobs."""
import hashlib,json,time,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research/matched_neural_lopo_20261107"
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.chem=nn.Sequential(nn.Linear(2048,32),nn.ReLU())
        self.protein=nn.Sequential(nn.Linear(1280,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,c,p):
        return self.head(torch.cat([self.chem(c),self.protein(p)],1)).flatten()
def fit(c,p,y,train,test,seed,epochs=300):
    torch.manual_seed(seed);model=Head()
    mean=y[train].mean();scale=y[train].std(unbiased=False).clamp_min(.01)
    target=(y[train]-mean)/scale
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    start=None
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss=(model(c[train],p[train])-target).square().mean()
        assert torch.isfinite(loss)
        if start is None:start=float(loss.detach())
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
    model.eval()
    with torch.inference_mode():
        pred=model(c[test],p[test])*scale+mean
        final=float((model(c[train],p[train])-target).square().mean())
    return pred.numpy(),{"initial_standardized_mse":start,"final_standardized_mse":final}
def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    source=ROOT/"data/engineering_pilot_primary_reconciled_20261107/rows.json"
    rows=[r for r in json.loads(source.read_text()) if r["split"]=="train" and r["primary_label_kind"]=="numeric_Kd"]
    targets=sorted({r["sequence"] for r in rows})
    assert len(rows)==106 and len(targets)==9
    seeds=[71110,71111,71112]
    protocol={"scope":"Training-development LOPO, not independent benchmark","rows":106,"targets":9,
      "seeds":seeds,"epochs":300,"optimizer":"AdamW lr .001 weight_decay .01","device":"CPU two threads",
      "architecture":"Morgan 2048->32; mean ESM 1280->32; concatenate->32->1, ReLU",
      "modes":["chemical_only_zero_protein","real_protein","deranged_protein"],
      "matching":"Same initialization, parameters, optimizer and full-batch steps for all modes",
      "normalization":"Protein centering/scaling and label standardization use fold training rows only",
      "selection":"Fixed final epoch; no early stopping, hyperparameter search or validation scoring",
      "rows_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
      "limits":["Compact diagnostic head, not existing SequenceFoundation","Repeatedly inspected training-development pool","Identifier-disjoint, not homology-disjoint","Numeric-only ascertainment bias"]}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
    pack=np.load(ROOT/"data/staging/morgan_radius2_2048_chiral.npz")
    fp=dict(zip(pack["ids"],pack["bits"]))
    c=torch.tensor(np.stack([fp[r["compound"]] for r in rows]),dtype=torch.float32)
    embeddings=[]
    for t in targets:
        obj=torch.load(ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(t+".pt"),map_location="cpu",weights_only=True)
        assert obj["sequence_hash"]==t
        embeddings.append(obj["embeddings"].float().mean(0))
    x=torch.stack(embeddings)
    pi=np.array([targets.index(r["sequence"]) for r in rows])
    y=torch.tensor([r["numeric_pKd"] for r in rows],dtype=torch.float32)
    records=[];predictions=[];started=time.time()
    for hold in range(9):
        train=np.flatnonzero(pi!=hold);test=np.flatnonzero(pi==hold)
        assert set(pi[train]).isdisjoint(pi[test])
        for seed in seeds:
            rng=np.random.default_rng(seed);perm=rng.permutation(9)
            while np.any(perm==np.arange(9)):perm=rng.permutation(9)
            for mode in protocol["modes"]:
                p=x[pi] if mode!="deranged_protein" else x[perm][pi]
                mu=p[train].mean(0);scale=p[train].std(0,unbiased=False).clamp_min(.1)
                p=(p-mu)/scale
                if mode=="chemical_only_zero_protein":p=torch.zeros_like(p)
                pred,loss=fit(c,p,y,train,test,seed)
                assert np.isfinite(pred).all()
                rec={"target":rows[test[0]]["primary_protein"],"seed":seed,"mode":mode,
                     "n_query":len(test),"rmse":float(np.sqrt(np.mean((pred-y[test].numpy())**2))),**loss}
                records.append(rec)
                predictions.extend({"target":rec["target"],"compound":rows[i]["compound"],"seed":seed,"mode":mode,"y":float(y[i]),"pred":float(v)} for i,v in zip(test,pred))
        (OUT/"progress.json").write_text(json.dumps({"completed_folds":hold+1,"fits":len(records),"elapsed_seconds":time.time()-started}))
        print(json.dumps({"completed_folds":hold+1,"fits":len(records),"elapsed_seconds":round(time.time()-started,1)}),flush=True)
    (OUT/"fits.json").write_text(json.dumps(records,indent=2))
    (OUT/"predictions_LOCAL.json").write_text(json.dumps(predictions))
    means={mode:float(np.mean([r["rmse"] for r in records if r["mode"]==mode])) for mode in protocol["modes"]}
    (OUT/"summary.json").write_text(json.dumps({"mean_seed_target_rmse":means,"fits":len(records),"elapsed_seconds":time.time()-started},indent=2))
    print(json.dumps(means),flush=True)
if __name__=="__main__":main()
