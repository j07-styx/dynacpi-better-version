"""Save per-row inner-dev predictions of the already-locked BALM ridge/interaction ensemble.

The model choice (selected_by_train_cv) is READ from the existing report; nothing is
re-selected and no hyperparameter changes. Fits on the collapsed TRAIN rows only.
Writes research/balm_ensemble_dev_predictions_20261108.json (refuses to overwrite).
The reserved test is never loaded.
Usage: python -m scripts.save_balm_ensemble_dev_predictions
"""
import json,hashlib
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from dynacpi.balm_public_data_dev import get_train_dev,ROOT
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import collapse_training
from scripts.train_balm_bindingdb_explicit_interaction_ensemble_dev import fit_predict

REPORT=ROOT/"research/balm_bindingdb_explicit_chemical_protein_interaction_ensemble_dev.json"
OUT=ROOT/"research/balm_ensemble_dev_predictions_20261108.json"

def row_hash(rows):
    h=hashlib.sha256()
    for r in rows:
        h.update(f"{r['protein']}|{r['smiles']}|{r['label']}\n".encode())
    return h.hexdigest()

def main():
    if OUT.exists():raise FileExistsError(OUT)
    selected=json.loads(REPORT.read_text(encoding="utf-8"))["selected_by_train_cv"]
    parts=get_train_dev()
    train=collapse_training(parts["train"]);dev=parts["inner_dev"]
    assert len(train)==12315 and len(dev)==3891
    assert {r["group"] for r in train}.isdisjoint({r["group"] for r in dev})
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    pred=fit_predict(train,dev,gen,{},{},{})
    result={"status":"BALM_FROZEN_INNER_DEV_ENSEMBLE_PREDICTIONS_NO_TEST",
            "selected_by_train_cv_from_existing_report":selected,
            "dev_rows":len(dev),"dev_row_sha256":row_hash(dev),
            "predictions":{k:[round(float(x),5) for x in v] for k,v in pred.items()
                           if k in ("ridge",selected)},
            "train_mean":float(np.mean([r["label"] for r in train]))}
    OUT.write_text(json.dumps(result)+"\n",encoding="utf-8")
    print("SAVED",OUT,"selected",selected,flush=True)

if __name__=="__main__":main()
