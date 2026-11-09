"""Cold-start compound-protein similarity memory for affinity regression.

No model fitting on heldout labels. Chemical similarities use normalized
Morgan-bit cosine, protein similarities frozen ESM2 mean vector cosine.
Only the top k TRAIN pairs can contribute to any heldout prediction.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MemorySpec:
    name:str
    ligand_exponent:int
    protein_exponent:int
    neighbor_count:int


MEMORY_SPECS=(
    MemorySpec("chemical_only_k64",4,0,64),
    MemorySpec("joint_esm8_k64",2,8,64),
    MemorySpec("joint_esm16_k64",2,16,64),
    MemorySpec("joint_esm8_k256",2,8,256),
)


def neighbor_weighted_labels(chemical_cosine,protein_cosine,
                             training_labels,train_mean,spec,block_size=256):
    """Top-k similarity-weighted training label averages.

    Similarity matrix entries must be derived without heldout labels. Process
    rows in blocks so no dense 16m-pair products need persistent storage.
    """
    y=np.asarray(training_labels,dtype=np.float64)
    chem=np.asarray(chemical_cosine,dtype=np.float32)
    protein=np.asarray(protein_cosine,dtype=np.float32)
    if chem.shape!=protein.shape or chem.ndim!=2:
        raise ValueError("Invalid similarity matrices")
    if chem.shape[1]!=len(y) or not np.isfinite(chem).all() or not np.isfinite(protein).all():
        raise ValueError("Invalid similarity or labels")
    if not np.isfinite(y).all():
        raise ValueError("Non-finite training labels")
    k=min(spec.neighbor_count,len(y))
    if k<=0:
        raise ValueError("At least one training example required")
    out=np.full(chem.shape[0],float(train_mean),dtype=np.float64)
    for begin in range(0,len(out),block_size):
        end=min(begin+block_size,len(out))
        c=np.maximum(chem[begin:end],0.)
        scores=c**spec.ligand_exponent
        if spec.protein_exponent:
            p=np.maximum(protein[begin:end],0.)
            scores=scores*p**spec.protein_exponent
        idx=np.argpartition(-scores,kth=k-1,axis=1)[:,:k]
        w=np.take_along_axis(scores,idx,axis=1).astype(np.float64)
        neighbors=y[idx]
        total=w.sum(axis=1)
        good=total>1e-12
        out[begin:end][good]=(neighbors[good]*w[good]).sum(axis=1)/total[good]
    return out

