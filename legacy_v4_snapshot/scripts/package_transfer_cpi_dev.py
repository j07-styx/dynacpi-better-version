"""Package selected model with required scaling and verify inference parity."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.transfer_data import load_development
from dynacpi.transfer_batch import TransferCollator
from dynacpi.transfer_cpi import TransferCPI
from dynacpi.dual_transfer_cpi import DualTransferCPI
def main(seed,variant="transfer"):
    torch.set_num_threads(2)
    out=ROOT/"research/transfer_cpi_dev_20261108"/f"{variant}_seed{seed}"
    state=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
    cache=torch.load(ROOT/"data/transfer_compounds_dev.pt",map_location="cpu",weights_only=True)
    rows,_,split_hash=load_development()
    assert state["protocol"]["split_sha256"]==cache["meta"]["split_sha256"]==split_hash
    state["property_mean"]=cache["property_mean"];state["property_std"]=cache["property_std"]
    state["feature_spec"]={"protein":"esm2_t33_650M_UR50D layer33","protein_tokens":"contiguous_nonoverlap_mean_max64",
      "window":1000,"overlap":128,"protein_weights_sha256":"ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
      "compound":"scripts.prepare_transfer_compounds_dev.featurize","schema":1,"status":"PROVISIONAL_DEVELOPMENT_MODEL"}
    if variant.endswith("_clm"):
        language=torch.load(ROOT/"data/transfer_chemberta_dev.pt",weights_only=True)
        assert language["meta"]["split_sha256"]==split_hash
        state["feature_spec"]["chemical_language"]=language["meta"]
        for cid,g in cache["graphs"].items():g["chemical_language"]=language["features"][cid]
    model=(DualTransferCPI() if variant.endswith("_clm") else TransferCPI()).eval();model.load_state_dict(state["state"])
    selected=[r for r in rows if r["split"]=="dev"][:32]
    proteins={r["protein"]:torch.load(ROOT/"data/transfer_esm64_dev"/(r["protein"]+".pt"),weights_only=True)["tokens"] for r in selected}
    collate=TransferCollator(cache["graphs"],proteins,state["property_mean"],state["property_std"])
    with torch.no_grad():pred=model(collate(selected))[0].numpy()*state["label_std"]+state["label_mean"]
    expected=np.array([r["pred"] for r in json.loads((out/"best_predictions_LOCAL.json").read_text())[:32]])
    error=float(np.max(np.abs(pred-expected)));assert error<2e-5
    dest=out/"inference_package.pt"
    if dest.exists():raise FileExistsError(dest)
    torch.save(state,dest)
    report={"seed":seed,"selected_epoch":state["epoch"],"parity_rows":len(selected),
       "cpu_vs_training_gpu_max_abs_error":error,"artifact_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
       "scope":"Development model; preprocessing included; requires matching frozen protein tokens"}
    with (out/"package_verification.json").open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=2601);p.add_argument("--model",choices=["transfer","transfer_clm"],default="transfer");a=p.parse_args();main(a.seed,a.model)
