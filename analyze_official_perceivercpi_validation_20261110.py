"""Validation-only, equal-metric comparison of the official PerceiverCPI interim checkpoint.
No 81-protein held-out DAVIS labels are read. Pooled RMSE and target-macro RMSE
are computed on the *same* original 59-protein validation set as DynaCPI v5.
Validation is used for checkpoint selection; do not claim test superiority.
"""
import ast
import csv
import json
from pathlib import Path
import numpy as np
from train_full_davis_v5 import metrics, RUN

ROOT=Path(__file__).resolve().parent/"matched_perceivercpi_20261110"
VAL=ROOT/"data_v2"/"val.csv"
PRED=ROOT/"validation_check_epoch3_seed2931"/"validation_predictions.csv"
OUT=ROOT/"validation_check_epoch3_seed2931"/"VALIDATION_INTERIM_METRICS.json"
SELECTED=RUN/"SELECTION_FROZEN.json"

def main():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite a previous validation comparison")
    if not PRED.exists():
        raise RuntimeError("Official validation predictions not yet complete")
    with VAL.open(newline="",encoding="utf-8") as f:
        truth=list(csv.DictReader(f))
    with PRED.open(newline="",encoding="utf-8") as f:
        pred=list(csv.DictReader(f))
    if len(truth)!=4012 or len(pred)!=4012:
        raise RuntimeError(f"Expected 4012 validation predictions, got {len(truth)}, {len(pred)}")
    for idx,(t,p) in enumerate(zip(truth,pred)):
        smile=p["smiles"]
        try:
            parsed=ast.literal_eval(smile)
            if isinstance(parsed,list) and len(parsed)==1:
                smile=parsed[0]
        except (ValueError,SyntaxError):
            pass
        if smile!=t["smiles"]:
            raise RuntimeError(f"SMILES row alignment failure at index {idx}: {smile} != {t['smiles']}")
    y=np.asarray([float(x["label"]) for x in truth],dtype=np.float64).reshape((59,68))
    p=np.asarray([float(x["label"]) for x in pred],dtype=np.float64).reshape((59,68))
    if not np.isfinite(p).all():raise RuntimeError("Nonfinite official validation predictions")
    official=metrics(p,y)
    vals=json.loads(SELECTED.read_text(encoding="utf-8"))
    dynacpi=vals["validation_scores"]["centered_esm_censored"]
    chemical=vals.get("chemical_prior_validation")
    result={
        "status":"VALIDATION_ONLY_NO_HELDOUT_TEST",
        "n_validation_protein_entries":59,
        "n_validation_pairs":4012,
        "checkpoint":"frozen interim official PerceiverCPI after epoch 3; 30 planned epochs not finished",
        "original_upstream_unchanged":True,
        "perceiver_interim":official,
        "dynacpi_v5_selected_on_same_validation":dynacpi,
        "relative_macro_rmse_difference_percent":float(100*(1-dynacpi["macro_rmse"]/official["macro_rmse"])),
        "relative_pooled_rmse_difference_percent":float(100*(1-dynacpi["pooled_rmse"]/official["pooled_rmse"])),
        "warning":"VALIDATION DATA USED FOR SELECTION OF BOTH MODELS, Perceiver still undertraining, results NOT independent test or evidence of generalizable superiority."
    }
    OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
