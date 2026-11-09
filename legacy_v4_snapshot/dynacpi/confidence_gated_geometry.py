"""Learnable confidence-gated geometry residual, opt-in DynaCPI v4 experiment.

The legacy GVP checkpoint is unchanged. This module wraps it with a
per-residue gate conditioned on sequence and geometry residual, plus
global structure coverage. Missing structure remains exact identity.
"""
import torch
from torch import nn

class ConfidenceGatedGeometry(nn.Module):
    def __init__(self, geometry, width=256, hidden=128, gate_bias=-2.0):
        super().__init__()
        self.geometry=geometry
        self.gate=nn.Sequential(
            nn.LayerNorm(width*2+1),
            nn.Linear(width*2+1,hidden),
            nn.GELU(),
            nn.Linear(hidden,1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias,gate_bias)

    def enhance(self,tokens,graph):
        """Drop-in GeometryBranch-compatible interface for frozen inference."""
        return self.forward(tokens,graph)

    def forward(self,tokens,graph,return_gate=False):
        if tokens.ndim!=2 or tokens.shape[1]!=256:
            raise ValueError("Expected residue tokens [L,256]")
        if graph is None:
            if return_gate:
                return tokens,torch.zeros((len(tokens),1),device=tokens.device,dtype=tokens.dtype)
            return tokens
        idx=torch.as_tensor(graph["indices"],device=tokens.device,dtype=torch.long)
        if idx.ndim!=1 or idx.numel()==0 or idx.min()<0 or idx.max()>=len(tokens):
            raise ValueError("Invalid structure-to-sequence residue indices")
        if idx.unique().numel()!=idx.numel():
            raise ValueError("Duplicate structure residue indices")
        enhanced=self.geometry.enhance(tokens,graph)
        if enhanced.shape!=tokens.shape:raise ValueError("GVP residual alignment mismatch")
        delta=enhanced-tokens
        coverage=tokens.new_full((len(tokens),1),idx.numel()/len(tokens))
        inputs=torch.cat((tokens,delta,coverage),dim=-1)
        gates=torch.sigmoid(self.gate(inputs))
        mask=torch.zeros((len(tokens),1),device=tokens.device,dtype=tokens.dtype)
        mask.index_fill_(0,idx,1)
        gates=gates*mask
        result=tokens+gates*delta
        return (result,gates) if return_gate else result
