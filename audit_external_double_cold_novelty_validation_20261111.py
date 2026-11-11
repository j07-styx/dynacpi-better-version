"""Post-hoc double-cold VAL-only novelty and prior-error decomposition.

No new model selection, hyperparameter tuning, test-score access, or feature
training. Uses ONLY frozen TRAIN and VAL row indices; never indexes held-out
TEST labels, never writes TEST predictions. Designed after protein-cold tests
were observed, so this is descriptive only; NOT preregistered inference.
"""
from __future__ import annotations
import json,os
from pathlib import Path
import numpy as np,torch
from evaluate_external_kiba_metz_all_six_once_20261110 import (
    OUT,TRAINING_SEEDS,selected_predictions,infer_observed,metrics,sha
)
HERE=Path(__file__).resolve().parent
DEST=OUT/"DOUBLE_COLD_VALIDATION_NOVELTY_AUDIT_20261111"
REPORT=HERE/"KIBA_METZ_DOUBLE_COLD_VAL_NOVELTY_DIAGNOSTIC_2026-11-11.md"

def max_tanimoto(bits,valid_compounds,seen_compounds):
    # Ensure float32 accumulation: dot products of uint8 fingerprints overflow.
    mat=np.asarray(bits,dtype=np.float32)
    train=mat[seen_compounds]
    sums_train=train.sum(axis=1)[None,:]
    result={}
    for start in range(0,len(valid_compounds),32):
        c_batch=np.asarray(valid_compounds[start:start+32],dtype=int)
        a=mat[c_batch]
        both=a @ train.T
        denom=a.sum(axis=1)[:,None]+sums_train-both
        sim=both/np.maximum(denom,1)
        for idx,v in zip(c_batch,sim.max(axis=1)):
            result[int(idx)]=float(v)
    return result

def max_cosine(features,valid_proteins,seen_proteins):
    mat=np.asarray(features,dtype=np.float64)
    mat=mat/np.maximum(np.linalg.norm(mat,axis=1,keepdims=True),1e-12)
    sim=mat[valid_proteins] @ mat[seen_proteins].T
    return {int(k):float(v) for k,v in zip(valid_proteins,sim.max(axis=1))}

def err_summary(pred,y,pids):
    e=np.asarray(pred,dtype=float)-np.asarray(y,dtype=float)
    return {"macro_rmse":metrics(pred,y,pids)["macro_rmse"],
            "pooled_rmse":float(np.sqrt(np.mean(e*e))),
            "mean_absolute_error":float(np.mean(np.abs(e)))}

def bucket_report(pred,prior,y,pids,cids,novelty_map,mode):
    unique=np.asarray(sorted(novelty_map),dtype=int)
    v=np.asarray([novelty_map[int(k)] for k in unique],dtype=float)
    q=np.quantile(v,[1/3,2/3]).tolist()
    out={}
    ids=pids if mode=="protein" else cids
    a=np.asarray([novelty_map[int(k)] for k in ids],dtype=float)
    bins=[("low similarity",a<=q[0]),("mid similarity",(a>q[0])&(a<=q[1])),
          ("high similarity",a>q[1])]
    for name,mask in bins:
        if mask.sum()==0:
            out[name]={"n_pairs":0}
            continue
        d=err_summary(pred[mask],y[mask],pids[mask])
        b=err_summary(prior[mask],y[mask],pids[mask])
        out[name]={"n_pairs":int(mask.sum()),
                   "n_unique_proteins":int(len(np.unique(pids[mask]))),
                   "n_unique_compounds":int(len(np.unique(cids[mask]))),
                   "dyna":d,"chemical_prior":b,
                   "delta_macro_rmse_dyna_minus_prior":d["macro_rmse"]-b["macro_rmse"],
                   "delta_pooled_rmse_dyna_minus_prior":d["pooled_rmse"]-b["pooled_rmse"],
                   "pair_fraction_dyna_smaller_abs_error":float(np.mean(
                       np.abs(pred[mask]-y[mask])<np.abs(prior[mask]-y[mask])))}
    return {"cutoffs":q,"n_unique_entities":int(len(unique)),
            "min_nearest_similarity":float(v.min()),"max_nearest_similarity":float(v.max()),
            "bins":out}

def run(dataset):
    folder=OUT/dataset/"double_cold"
    split=folder/"SPLIT_LOCKED.json"
    frozen=json.loads((folder/"dynacpi_fixed_sparse_train"/"VALIDATION_FROZEN.json").read_text(encoding="utf-8"))
    spec=json.loads(split.read_text(encoding="utf-8"))
    if (frozen["status"]!="VALIDATION_LOCKED_TWO_SEEDS_NO_TEST"
        or frozen["frozen_split_sha256"]!=sha(split)
        or frozen["training_seeds"]!=list(TRAINING_SEEDS)):
        raise RuntimeError("Frozen 2-seed double-cold checkpoint invalid")
    # No test index access is permitted in this diagnostic.
    train=np.asarray(spec["pair_train_row_indices"],dtype=int)
    val=np.asarray(spec["pair_val_row_indices"],dtype=int)
    if np.intersect1d(train,val).size:
        raise RuntimeError("TRAIN and VALID overlap")
    with np.load(OUT/dataset/"source_features.npz",allow_pickle=False) as z:
        data={k:z[k] for k in ("protein_features","compound_morgan_bits",
          "protein_index","compound_index","observed_score","chemical_smiles")}
    selected=selected_predictions(data,spec,folder/"dynacpi_fixed_sparse_train")
    pids=data["protein_index"]
    cids=data["compound_index"]
    y=data["observed_score"][val].astype(np.float64)
    pred=np.mean([infer_observed(m,val,selected["pids"],selected["cids"],
        selected["prot"],selected["chem"],selected["prior"],selected["sd"])
        for m in selected["models"]],axis=0).astype(np.float64)
    prior=selected["prior"].detach().cpu().numpy()[cids[val]].astype(np.float64)
    observed=metrics(pred,y,pids[val])
    anchored=metrics(prior,y,pids[val])
    for group,actual,expected in (
        ("Dyna",observed,frozen["validation_two_seed"]),
        ("Prior",anchored,frozen["train_only_chemical_prior_validation"])):
        for metric in ("macro_rmse","pooled_rmse"):
            if abs(actual[metric]-expected[metric])>1e-5:
                raise RuntimeError(f"{dataset} {group} {metric} replay mismatch")
    valprots=np.unique(pids[val])
    trainprots=np.unique(pids[train])
    valchems=np.unique(cids[val])
    trainchems=np.unique(cids[train])
    if (np.intersect1d(valprots,trainprots).size>0
        or np.intersect1d(valchems,trainchems).size>0):
        raise RuntimeError("Double-cold validation leakage")
    molecule_similarity=max_tanimoto(
        data["compound_morgan_bits"],valchems,trainchems)
    protein_similarity=max_cosine(
        data["protein_features"],valprots,trainprots)
    return {
      "audit_type":"POST_HOC_TRAIN_AND_VALIDATION_ONLY_NEVER_OPEN_TEST",
      "dataset":dataset,"scenario":"double_cold",
      "source_features_sha256":sha(OUT/dataset/"source_features.npz"),
      "frozen_manifest_sha256":sha(split),
      "validation_frozen_checkpoint_seed_count":len(TRAINING_SEEDS),
      "n_training_pairs":int(len(train)),"n_validation_pairs":int(len(val)),
      "n_validation_proteins":int(len(valprots)),
      "n_validation_compounds":int(len(valchems)),
      "dyna_overall_val":observed,
      "chemical_prior_overall_val":anchored,
      "overall_dyna_minus_prior_macro":observed["macro_rmse"]-anchored["macro_rmse"],
      "overall_dyna_minus_prior_pooled":observed["pooled_rmse"]-anchored["pooled_rmse"],
      "protein_esm_cosine_vs_train":bucket_report(pred,prior,y,pids[val],cids[val],
          protein_similarity,"protein"),
      "morgan_tanimoto_vs_train":bucket_report(pred,prior,y,pids[val],cids[val],
          molecule_similarity,"chemical"),
      "limits":"Post-hoc VAL error diagnosis only. All checkpoint selections frozen prior to any external TEST. Nearest-embedding scores do not certify protein family distance, chemical scaffold bins are not chemical potency strata; small bins and correlated pairs. No model tuning allowed."
    }

def main():
    if DEST.exists() or REPORT.exists():
        raise RuntimeError("Refuse overwriting existing novelty audit")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    results=[run(dataset) for dataset in ("kiba","metz")]
    DEST.mkdir(parents=True,exist_ok=False)
    for item in results:
        (DEST/(item["dataset"]+"_double_cold_val_only.json")).write_text(
            json.dumps(item,indent=2),encoding="utf-8")
    lines=["# KIBA & Metz double-cold validation novelty diagnostic (11 November 2026)",
      "",
      "**Post-hoc validation-only experiment.** Preselected two-seed models and immutable",
      "train/validation indices were replayed exactly. No held-out double-cold test",
      "labels were indexed or scored. These findings must not be used to retrain,",
      "tune, select, or modify any pending frozen experiments.",
      "",
      "## Frozen overall validation comparison",
      "",
      "| Dataset | Pairs | Dyna macro | Prior macro | Dyna pooled | Prior pooled |",
      "|---|---:|---:|---:|---:|---:|"]
    for obj in results:
        d=obj["dyna_overall_val"];b=obj["chemical_prior_overall_val"]
        lines.append(f"| {obj['dataset'].upper()} | {obj['n_validation_pairs']:,} | "
          f"{d['macro_rmse']:.6f} | {b['macro_rmse']:.6f} | "
          f"{d['pooled_rmse']:.6f} | {b['pooled_rmse']:.6f} |")
    lines.extend(["","## Error by nearest-neighbor similarity (training set only)",
       "",
       "Scores computed per unique held-out-validation protein or molecule,",
       "then binned by thirds of *unique-entity* nearest-training similarity.",
       "Each diagnostic table reports pair-level pooled RMSE; negative Δ favors DynaCPI.",
       "The chemical-prior anchor was fitted only on training labels.",
       "Protein similarities are cosine of cached frozen ESM-2 vectors;",
       "chemical similarities are Tanimoto of binary 2048-bit Morgan fingerprints.",
       "",""])
    for obj in results:
        lines.append(f"### {obj['dataset'].upper()}")
        for key,label in (
            ("protein_esm_cosine_vs_train","Protein nearest-training ESM-2 cosine"),
            ("morgan_tanimoto_vs_train","Molecule nearest-training Morgan Tanimoto")):
            sect=obj[key]
            lines.extend([f"**{label}**; boundaries: "
                f"{sect['cutoffs'][0]:.4f}, {sect['cutoffs'][1]:.4f}; "
                f"{sect['n_unique_entities']} validation entities.",
                "",
                "| Validation similarity bin | Pairs | Dyna pooled RMSE | Prior pooled RMSE | Δ pooled | Dyna better pairs |",
                "|---|---:|---:|---:|---:|---:|"])
            for name,data in sect["bins"].items():
                if not data["n_pairs"]:
                    lines.append(f"| {name} | 0 | — | — | — | — |")
                else:
                    lines.append(f"| {name} | {data['n_pairs']:,} | "
                        f"{data['dyna']['pooled_rmse']:.4f} | "
                        f"{data['chemical_prior']['pooled_rmse']:.4f} | "
                        f"{data['delta_pooled_rmse_dyna_minus_prior']:+.4f} | "
                        f"{100*data['pair_fraction_dyna_smaller_abs_error']:.1f}% |")
            lines.append("")
    lines.extend(["## Caveats",
       "",
       "- Correlated protein–compound pair counts; subgroup sizes differ. No subgroup p-values, causal explanations, or universal out-of-domain claims.",
       "- The bins are diagnostics, not calibrated uncertainty estimates. Validation is used for model selection, so all findings are exploratory.",
       "- This is a DynaCPI-versus-chemical-prior *validation* audit; there is no held-out double-cold comparison to original PerceiverCPI here.",
       "- The ESM-2 nearest-neighbor measure does not prove protein family independence; similarly, Morgan distances do not guarantee identical chemical scaffold boundaries.",
       "- The public report intentionally excludes raw labels, predictions and compound/protein identifiers. Full aggregate diagnostics and SHA hashes are in the gitignored local output directory.",
       ""])
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"status":"VAL_ONLY_DOUBLE_COLD_NOVELTY_AUDIT_COMPLETE",
       "datasets":[{"dataset":r["dataset"],
                   "pairs":r["n_validation_pairs"],
                   "dyna_pooled":r["dyna_overall_val"]["pooled_rmse"],
                   "prior_pooled":r["chemical_prior_overall_val"]["pooled_rmse"]}
                  for r in results],
       "public_report":str(REPORT)},indent=2),flush=True)

if __name__=="__main__":
    main()
