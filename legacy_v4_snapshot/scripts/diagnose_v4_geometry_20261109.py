"""Read-only DynaCPI v4 GVP diagnostics on already-inspected DAVIS DEV.

Replays frozen 3x real and misaligned checkpoints without optimization.
Measures per-target direction of geometric correction, comparison to
the same trained network with scrambled residue mapping, and MSE components.
Never writes row-level affinity or predictions.
"""
import sys,json,math,hashlib,statistics
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
from scripts.train_v4_geometry_trainonly_20261109 import setup, SEEDS, OUTPUT, BATCH, sha_file
from dynacpi.geometry_branch import GeometryBranch
from gpu_lease import gpu_lease

DEST=OUTPUT/"DIAGNOSIS_20261109.json"

def predict(base,geom,row,graph,mean,std):
    tokens=row["tokens"]
    enhanced=geom.enhance(tokens,graph)
    estimates=[]
    for i in range(0,len(row["x"]),BATCH):
        x=row["x"][i:i+BATCH]
        valid=torch.ones((len(x),len(tokens)),dtype=torch.bool,device="cuda")
        pp=base.predict_tokens(x,enhanced[None,:,:].expand(len(x),-1,-1),valid)
        estimates.extend((pp.float().cpu().numpy()*std+mean).tolist())
    return np.asarray(estimates,dtype=np.float64)

def record_metrics(y,baseline,prediction):
    y=np.asarray(y,dtype=np.float64)
    baseline=np.asarray(baseline,dtype=np.float64)
    prediction=np.asarray(prediction,dtype=np.float64)
    d=prediction-baseline
    residual=y-baseline
    if d.std()>1e-9 and residual.std()>1e-9:
        r=float(np.corrcoef(d,residual)[0,1])
    else:
        r=None
    base_mse=float(np.mean((y-baseline)**2))
    geom_mse=float(np.mean((y-prediction)**2))
    delta_squared=float(np.mean(d*d))
    align_gain=float(-2*np.mean(d*residual))
    if not np.isclose(geom_mse-base_mse,delta_squared+align_gain,atol=1e-5):
        raise RuntimeError("MSE decomposition mismatch")
    return {"baseline_rmse":float(math.sqrt(base_mse)),"geometry_rmse":float(math.sqrt(geom_mse)),
       "geometry_minus_baseline_rmse":float(math.sqrt(geom_mse)-math.sqrt(base_mse)),
       "mean_prediction_delta":float(np.mean(d)),
       "mean_abs_prediction_delta":float(np.mean(np.abs(d))),
       "std_prediction_delta_within_protein":float(np.std(d)),
       "mean_baseline_residual_y_minus_prediction":float(np.mean(residual)),
       "correlation_delta_to_baseline_residual":r,
       "fraction_pairs_mse_improved":float(np.mean((y-prediction)**2<(y-baseline)**2)),
       "mse_penalty_delta_squared":delta_squared,
       "mse_alignment_term":align_gain,
       "error_change_mse":float(geom_mse-base_mse),
       "n_compounds":len(y),
       "n_unique_y":int(len(set(map(float,y))))}

def main():
    if DEST.exists():
        raise RuntimeError("Refusing to overwrite completed diagnostics file")
    t0=__import__("time").perf_counter()
    base,data,fit,inner,dev,mean,std,hashes=setup()
    output={"status":"ENGINEERING_DIAGNOSIS_PREVIOUSLY_INSPECTED_DAVIS_DEV_ONLY",
            "input_hashes":hashes,"seeds":SEEDS,
            "data_boundary":"No BindingDB labels or reserved prospective-test outcomes",
            "arms":[]}
    base.eval()
    for seed in SEEDS:
        result={"seed":seed,"proteins":{}}
        models={}
        for mode in ("real","misaligned"):
            ck_path=OUTPUT/(mode+"_seed"+str(seed))/"best_checkpoint.pt"
            ck=torch.load(ck_path,map_location="cpu",weights_only=False)
            if ck["seed"]!=seed or ck["mode"]!=mode or ck["sequence_checkpoint_sha256"]!=hashes["sequence_checkpoint_sha256"]:
                raise RuntimeError("Checkpoint identity mismatch")
            model=GeometryBranch().cuda().eval()
            model.load_state_dict(ck["state"],strict=True)
            models[mode]=model
        with torch.inference_mode():
            for pid in dev:
                row=data[pid]
                a=np.asarray(row["base_predictions"],dtype=np.float64)
                y=row["y"]
                r=predict(base,models["real"],row,row["graph"],mean,std)
                shuffled=predict(base,models["real"],row,row["misaligned_graph"],mean,std)
                controlled=predict(base,models["misaligned"],row,row["misaligned_graph"],mean,std)
                report={
                   "real":record_metrics(y,a,r),
                   "same_real_weights_but_misaligned_inference":record_metrics(y,a,shuffled),
                   "separately_trained_misaligned_control":record_metrics(y,a,controlled),
                   "same_weights_structure_alignment_mean_abs_difference":
                      float(np.mean(np.abs(r-shuffled))),
                   "same_weights_structure_alignment_max_abs_difference":
                      float(np.max(np.abs(r-shuffled))),
                   "correlation_real_vs_scrambled_delta":(
                       float(np.corrcoef(r-a,shuffled-a)[0,1])
                       if np.std(r-a)>1e-9 and np.std(shuffled-a)>1e-9 else None)
                }
                result["proteins"][row["name"]]=report
                print(json.dumps({"seed":seed,"protein":row["name"],
                    "real_delta_rmse":report["real"]["geometry_minus_baseline_rmse"],
                    "mean_real_shift":report["real"]["mean_prediction_delta"],
                    "within_target_shift_std":report["real"]["std_prediction_delta_within_protein"],
                    "same_weights_mapping_mae":report["same_weights_structure_alignment_mean_abs_difference"]}),flush=True)
        output["arms"].append(result)
        del models
        torch.cuda.empty_cache()
    output["runtime_seconds"]=round(__import__("time").perf_counter()-t0,2)
    DEST.write_text(json.dumps(output,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"DONE","output":str(DEST),"runtime_seconds":output["runtime_seconds"]}),flush=True)

if __name__=="__main__":
    with gpu_lease(ROOT):
        main()
