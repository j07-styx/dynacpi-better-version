"""Re-evaluate a stored DAVIS PerceiverCPI fold-zero checkpoint on the
original held-out rows and compare to its archived predictions.

Read-only legacy sources. Does not train a DynaCPI model, and the legacy split
is NOT a protein-family-held-out DynaCPI benchmark.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
import numpy as np
import torch
from tape import TAPETokenizer
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/"scripts"))
from chemprop.features import mol2graph
from chemprop.models.model import InteractionModel
from gpu_lease import gpu_lease

legacy=Path(r"C:\Users\J0124\OneDrive\Desktop\perceiver_cpi")
folder=legacy/"reproduction/davis_novel_pair_fold0"
ckpt=folder/"FINAL_model0_best.pt"
reference=folder/"FINAL_model0_test_preds.csv"
test=folder/"test.csv"
out=root/"research/legacy_checkpoint_parity.json"

def hashfile(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for buf in iter(lambda:f.read(4*1024*1024),b""):
            h.update(buf)
    return h.hexdigest()

def run():
    with test.open(encoding="utf-8-sig",newline="") as f:
        actual=list(csv.DictReader(f))
    with reference.open(encoding="utf-8-sig",newline="") as f:
        refs=list(csv.DictReader(f))
    assert len(refs)==len(actual)==1148
    for a,b in zip(actual,refs):
        assert a["smiles"]==b["smiles"] and a["sequence"]==b["sequence"]
    # Check full source equality before claiming this extension clone reuses the
    # original baseline implementation.
    paths=["chemprop/models/model.py","chemprop/models/mpn.py",
           "chemprop/models/CAB.py","chemprop/data/data.py",
           "chemprop/features/featurization.py","chemprop/train/predict.py"]
    source_checks={f:hashfile(legacy/f)==hashfile(root/f) for f in paths}
    if not all(source_checks.values()):
        raise RuntimeError("Original and isolated clone reference sources differ: "+str(source_checks))
    state=torch.load(ckpt,map_location="cpu",weights_only=False)
    args=state["args"]
    args.device=torch.device("cuda")
    model=InteractionModel(args).cuda().eval()
    incompat=model.load_state_dict(state["state_dict"],strict=True)
    assert not incompat.missing_keys and not incompat.unexpected_keys
    scaler=state["data_scaler"]
    mean=float(scaler["means"][0]) if scaler is not None else 0.0
    std=float(scaler["stds"][0]) if scaler is not None else 1.0
    tok=TAPETokenizer(vocab="unirep")
    predictions=[]
    # The original routine evaluated in sequential groups of 50. Preserve
    # those exact group boundaries because the upstream padded-neighbor mean
    # is batch-composition dependent.
    for off in range(0,len(actual),50):
        batch=actual[off:off+50]
        smiles=[x["smiles"] for x in batch]
        tensor=[]
        fp=[]
        for row in batch:
            codes=list(tok.encode(list(row["sequence"])))
            seq=(codes+[0]*500)[:500]
            tensor.append(seq)
            mol=Chem.MolFromSmiles(row["smiles"])
            if mol is None:raise ValueError("Invalid source molecule")
            fp.append(list(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)))
        x=torch.tensor(tensor,dtype=torch.long)
        fps=torch.tensor(fp,dtype=torch.float32)
        with torch.inference_mode():
            pp=(model([mol2graph(smiles)],x,fps).flatten().cpu().numpy()*std+mean)
        predictions.extend(pp.tolist())
    pred=np.asarray(predictions,dtype=float)
    original=np.asarray([float(r["label"]) for r in refs],dtype=float)
    diff=np.abs(pred-original)
    report={
        "status":"PASS" if float(diff.max())<1e-5 else "MISMATCH_INVESTIGATE",
        "scope":"All 1,148 archived first-fold DAVIS test compounds using pinned unmodified reference model and original inference batch boundaries; NOT DynaCPI",
        "checkpoint_sha256":hashfile(ckpt),
        "test_sha256":hashfile(test),
        "saved_predictions_sha256":hashfile(reference),
        "source_files_bit_identical":source_checks,
        "model_loaded_strict":True,
        "reference_batch_size":50,
        "rows_compared":len(actual),
        "max_abs_pKd_prediction_difference":float(diff.max()),
        "mean_abs_pKd_prediction_difference":float(diff.mean()),
        "reference_mean_all":float(original.mean()),
        "regression_label_mean":mean,
        "regression_label_std":std,
        "warning":"No PerceiverCPI publication-wide parity or trained DynaCPI improvement is established."}
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
    if report["status"]!="PASS":
        raise AssertionError("Stored baseline prediction parity mismatch")

if __name__=="__main__":
    with gpu_lease(root):run()
