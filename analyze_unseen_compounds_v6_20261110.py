"""Read-only OOD geometry for the 12 v6 unseen-compound predictions.
Uses exact frozen Morgan radius2 2048-bit features and all 68 DAVIS ligands.
Does not use or report external binding labels and cannot calculate test RMSE.
"""
import json
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from train_full_davis_v5 import PREP, tanimoto, OUT, sha

def main():
    src=OUT/"unseen_compound_inference_20261110.json"
    rows=json.loads(src.read_text(encoding="utf-8-sig"))
    if len(rows)!=12:raise RuntimeError("Expected 12 inference results")
    with np.load(PREP,allow_pickle=False) as data:
        b=data["bits"].astype(np.float64)
        original=data["smiles"].astype(str)
    fpgen=GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    per=[]
    for c in sorted({r["compound"] for r in rows}):
        matching=[r for r in rows if r["compound"]==c]
        assert len(matching)==2 and all(r["compound_in_original_davis"] is False for r in matching)
        smi=matching[0]["smiles"]
        m=Chem.MolFromSmiles(smi)
        if m is None:raise RuntimeError("Invalid SMILES "+smi)
        a=np.zeros(2048,dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(m),a)
        similarities=tanimoto(a.reshape(1,-1),b)[0]
        best=int(np.argmax(similarities))
        per.append({"compound":c,"smiles":smi,"max_tanimoto_to_68_training_compounds":float(similarities[best]),
            "mean_tanimoto_to_68_training_compounds":float(similarities.mean()),
            "closest_training_compound_smiles":str(original[best]),
            "closest_training_compound_index":best,
            "abl1_prediction":next(r["prediction_pKd"] for r in matching if r["protein"]=="ABL1"),
            "egfr_prediction":next(r["prediction_pKd"] for r in matching if r["protein"]=="EGFR"),
            "compound_in_training":False})
    result={"status":"PASS_12_INFERENCES_SIX_UNSEEN_LIGANDS",
        "n_predictions":12,"n_unseen_compounds":6,"n_target_proteins":2,
        "predictions_source_sha256":sha(src),
        "fingerprint":"RDKit Morgan radius2 2048-bit includeChirality=True",
        "protocol":"max and mean Tanimoto similarity to all 68 DAVIS training compounds",
        "no_ground_truth_affinity_for_new_pairs":True,
        "no_unseen_compound_rmse_available":True,
        "floor_clamped_pKd_to_5":True,
        "risk":"OOD chemical extrapolation; two seed dispersion is not calibrated; exact 5.0 predictions may be clipping",
        "compounds":per}
    dest=OUT/"unseen_compound_ood_analysis_20261110.json"
    if dest.exists():raise RuntimeError("Refusing overwrite")
    dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    for v in per:
        print(f"{v['compound']:<16} max_train_Tanimoto={v['max_tanimoto_to_68_training_compounds']:.4f} ABL1={v['abl1_prediction']:.3f} EGFR={v['egfr_prediction']:.3f}",flush=True)
    print(f"PASS_READ_ONLY_OOD_PROFILE {dest}",flush=True)

if __name__=="__main__":main()
