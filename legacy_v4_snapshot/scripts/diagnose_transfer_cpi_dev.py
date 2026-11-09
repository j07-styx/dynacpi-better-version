"""Development-only perturbations; not a test-set or causal importance estimate."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from dynacpi.transfer_data import load_development
from dynacpi.transfer_batch import TransferCollator
from dynacpi.transfer_cpi import TransferCPI
from dynacpi.dual_transfer_cpi import DualTransferCPI
from train_transfer_cpi_dev import metrics
from gpu_lease import gpu_lease

def main():
    torch.set_num_threads(4)
    rows,seq,sha=load_development();dev=[r for r in rows if r["split"]=="dev"]
    cache=torch.load(ROOT/"data/transfer_compounds_dev.pt",weights_only=True)
    lang=torch.load(ROOT/"data/transfer_chemberta_dev.pt",weights_only=True)
    assert cache["meta"]["split_sha256"]==lang["meta"]["split_sha256"]==sha
    for cid,g in cache["graphs"].items():g["chemical_language"]=lang["features"][cid]
    proteins={p:torch.load(ROOT/"data/transfer_esm64_dev"/(p+".pt"),weights_only=True)["tokens"] for p in seq}
    collate=TransferCollator(cache["graphs"],proteins,cache["property_mean"],cache["property_std"])
    pids=sorted({r["protein"] for r in dev});rng=np.random.default_rng(2603)
    shuffled=list(rng.permutation(pids));mapping=dict(zip(pids,shuffled))
    report={"scope":"Inner-development diagnostic only; interventions are out-of-distribution, not causal attribution",
            "split_sha256":sha,"shuffle_seed":2603,"shuffled_protein_fixed_points":sum(k==v for k,v in mapping.items()),"models":{}}
    base=ROOT/"research/transfer_cpi_dev_20261108"
    for variant in ("transfer","transfer_clm"):
        out=base/(variant+"_seed2601");state=torch.load(out/"best.pt",weights_only=True,map_location="cpu")
        model=(DualTransferCPI() if variant.endswith("_clm") else TransferCPI()).cuda().eval();model.load_state_dict(state["state"])
        details={}
        for mode in ("original","protein_shuffle","interaction_removed"):
            predictions=[]
            with torch.inference_mode():
                for off in range(0,len(dev),32):
                    rr=dev[off:off+32]
                    if mode=="protein_shuffle":rr=[dict(r,protein=mapping[r["protein"]]) for r in rr]
                    batch={k:v.cuda() for k,v in collate(rr).items()}
                    pred,res=model(batch)
                    if mode=="interaction_removed":pred=pred-res
                    predictions.extend((pred*state["label_std"]+state["label_mean"]).cpu().tolist())
            pred=np.array(predictions);score=metrics(dev,pred);y=np.array([r["y"] for r in dev])
            score["mean_prediction"]=float(pred.mean());score["mean_label"]=float(y.mean())
            score["bias_by_component"]={g:float(np.mean((pred-y)[[r["component"]==g for r in dev]])) for g in score["component_rmse"]}
            if mode=="original":
                expected=np.array([r["pred"] for r in json.loads((out/"best_predictions_LOCAL.json").read_text())])
                error=float(np.max(np.abs(expected-pred)));assert error<2e-5,error
                score["saved_prediction_max_error"]=error
            details[mode]=score
        report["models"][variant]=details
    with (base/"perturbation_diagnostics.json").open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps({m:{k:{x:v[x] for x in ("pooled_rmse","component_macro_rmse","mean_prediction","mean_label")} for k,v in d.items()} for m,d in report["models"].items()},indent=2))
if __name__=="__main__":
    with gpu_lease(ROOT):main()
