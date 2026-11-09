"""Read-only v4 sequence/geometry prediction interface.

The shared foundation and GVP are evaluated with frozen weights. Inputs are
real molecular SMILES and a single protein's cached ESM residue embedding,
plus optional correctly mapped 3D graph. Nothing here trains or establishes
that model parameters are scientifically fitted.

One compound per graph-forward is deliberately used until the upstream
padding-sensitive MPN behavior is independently resolved for larger batches.
"""
from dataclasses import dataclass
from typing import Dict,Iterable,Optional
import hashlib
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from dynacpi.frozen_adapter import FrozenMeans

class FrozenPairPredictor:
    def __init__(self, foundation, geometry, device="cuda", label_mean=0.0, label_std=1.0):
        self.label_mean=float(label_mean)
        self.label_std=float(label_std)
        if not np.isfinite(self.label_mean) or not np.isfinite(self.label_std) or self.label_std<=0:
            raise ValueError("Invalid frozen training-label scale")
        self.foundation=foundation
        self.geometry=geometry
        self.device=torch.device(device)
        self.foundation.to(self.device).eval()
        self.geometry.to(self.device).eval()
        for model in (self.foundation,self.geometry):
            for param in model.parameters():
                param.requires_grad_(False)
        self.gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)

    @torch.inference_mode()
    def predict(self,smiles:Iterable[str],residue_embeddings:torch.Tensor,
                structure_graph:Optional[dict]=None)->Dict[str,FrozenMeans]:
        self.foundation.eval()
        self.geometry.eval()
        embeddings=torch.as_tensor(residue_embeddings,device=self.device,dtype=torch.float32)
        if embeddings.ndim!=2 or embeddings.shape[1]!=1280 or len(embeddings)==0:
            raise ValueError("Expected one nonempty protein [L,1280] ESM residue tensor")
        if not torch.isfinite(embeddings).all():
            raise ValueError("Nonfinite frozen ESM embeddings")
        sequence_tokens=self.foundation.sequence(embeddings)
        geometry_tokens=self.geometry.enhance(sequence_tokens,structure_graph) if structure_graph is not None else sequence_tokens
        if geometry_tokens.shape!=sequence_tokens.shape:
            raise ValueError("Geometry tokens misaligned to sequence")
        valid=torch.ones((1,sequence_tokens.shape[0]),dtype=torch.bool,device=self.device)
        result={}
        for smi in smiles:
            mol=Chem.MolFromSmiles(smi)
            if mol is None:raise ValueError("Invalid molecular SMILES")
            canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
            cid=hashlib.sha256(canonical.encode()).hexdigest()
            if cid in result:raise ValueError("Duplicate canonical compound")
            fp=self.gen.GetFingerprintAsNumPy(mol)
            if not fp.any():raise ValueError("Empty Morgan fingerprint")
            features=torch.tensor(np.asarray(fp).copy(),device=self.device,dtype=torch.float32)[None,:]
            compound=self.foundation.compound([canonical],features)
            s=float(self.foundation.predict_tokens(compound,sequence_tokens[None,:,:],valid).item())*self.label_std+self.label_mean
            g=float(self.foundation.predict_tokens(compound,geometry_tokens[None,:,:],valid).item())*self.label_std+self.label_mean
            if structure_graph is None:g=s
            if not np.isfinite(s) or not np.isfinite(g):
                raise ValueError("Nonfinite frozen model prediction")
            result[cid]=FrozenMeans(sequence_pKd=s,geometry_pKd=g,
                                    geometry_available=(structure_graph is not None))
        if not result:raise ValueError("No compounds requested")
        return result
