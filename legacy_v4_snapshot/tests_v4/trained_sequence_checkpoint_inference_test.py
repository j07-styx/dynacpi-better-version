"""Verify saved three-epoch DAVIS engineering checkpoint conversion to raw pKd.

The pilot uses a limited identifier-disjoint legacy DAVIS split with weak floor
and no independent kinase-family holdout. This proves checkpoint integration
only; it is NOT a scientific DynaCPI benchmark.
"""
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import hashlib
import json
import numpy as np
import sys
import torch

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/"scripts"))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.frozen_pair_predictor import FrozenPairPredictor
from gpu_lease import gpu_lease

def main():
    saved=root/"research/pilot_sequence/checkpoint.pt"
    data=torch.load(saved,map_location="cpu",weights_only=False)
    assert data["kind"]=="sequence" and data["epoch"]==3
    protocol=json.loads((root/"data/engineering_pilot/protocol.json").read_text(encoding="utf-8"))
    assert data["row_sha256"]==protocol["row_sha256"]
    lm=float(data["label_mean"])
    ls=float(data["label_std"])
    assert np.isfinite(lm) and np.isfinite(ls) and ls>0

    seq=SequenceFoundation().cuda().eval()
    incompat=seq.load_state_dict(data["state"],strict=True)
    assert not incompat.missing_keys and not incompat.unexpected_keys
    geom=GeometryBranch().cuda().eval()
    model=FrozenPairPredictor(seq,geom,device="cuda",label_mean=lm,label_std=ls)
    staging=json.loads((root/"data/staging/compounds.json").read_text(encoding="utf-8"))
    smiles=[next(iter(staging.values()))]
    mol=Chem.MolFromSmiles(smiles[0])
    canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    cid=hashlib.sha256(canonical.encode()).hexdigest()
    pdb_meta=json.loads((root/"data/structure_fixtures/P15056/metadata.json").read_text(encoding="utf-8"))[0]
    seqsha=hashlib.sha256(pdb_meta["uniprotSequence"].encode()).hexdigest()
    embs=torch.load(root/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(seqsha+".pt"),
                    map_location="cpu",weights_only=True)["embeddings"]
    report=model.predict([canonical],embs,None)[cid]
    assert report.geometry_available is False
    assert report.geometry_pKd==report.sequence_pKd

    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    bits=gen.GetFingerprintAsNumPy(mol).copy()
    fp=torch.tensor(bits,device="cuda",dtype=torch.float32)[None,:]
    with torch.inference_mode():
        standard=float(seq([canonical],fp,embs.float().cuda()[None,:,:],
             torch.ones((1,len(embs)),device="cuda",dtype=torch.bool)).item())
    expected=standard*ls+lm
    assert abs(report.sequence_pKd-expected)<1e-5

    result={
       "status":"PASS_TRAINED_ENGINEERING_CHECKPOINT_ONLY",
       "checkpoint_epoch":data["epoch"],
       "checkpoint_sha256":hashlib.sha256(saved.read_bytes()).hexdigest(),
       "training_labels_mean":lm,"training_labels_std":ls,
       "protein_length":len(embs),
       "raw_pKd_reconstruction_abs_difference":abs(report.sequence_pKd-expected),
       "strict_checkpoint_keys_loaded":True,
       "missing_3D_equals_sequence_exactly":True,
       "scope":"Legacy DAVIS engineering split, final 3-epoch checkpoint; not approved BindingDB study or held-out protein family result."
    }
    dest=root/"research/trained_sequence_checkpoint_inference_test.json"
    dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    with gpu_lease(root):main()
