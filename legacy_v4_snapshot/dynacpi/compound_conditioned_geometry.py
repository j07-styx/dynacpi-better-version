"""Compound-conditioned residue gate for opt-in DynaCPI development.
Each compound receives its own structure contribution; uncovered residues remain unchanged.
"""
import torch
from torch import nn

class CompoundConditionedGeometry(nn.Module):
    def __init__(self,geometry,width=256,hidden=128,gate_bias=-2.0):
        super().__init__()
        if width <= 0 or hidden <= 0:
            raise ValueError("width and hidden must be positive")
        self.width = width
        self.geometry=geometry
        self.gate=nn.Sequential(
            nn.LayerNorm(width*3+1),
            nn.Linear(width*3+1,hidden),
            nn.GELU(),
            nn.Linear(hidden,1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias,gate_bias)

    def forward(self,tokens,graph,compounds,return_gate=False):
        if tokens.ndim!=2 or tokens.shape[-1]!=self.width or tokens.shape[0]==0:
            raise ValueError(f"Expected nonempty tokens [L,{self.width}]")
        if compounds.ndim!=2 or compounds.shape[-1]!=self.width:
            raise ValueError(f"Expected compound vectors [B,{self.width}]")
        if compounds.device != tokens.device or compounds.dtype != tokens.dtype:
            raise ValueError("Compound and token device/dtype must match")
        batch,length=compounds.shape[0],tokens.shape[0]
        original=tokens.unsqueeze(0).expand(batch,-1,-1)
        if graph is None:
            if return_gate:return original,tokens.new_zeros(batch,length,1)
            return original
        idx=torch.as_tensor(graph["indices"],device=tokens.device,dtype=torch.long)
        if idx.ndim!=1 or idx.numel()==0 or idx.min()<0 or idx.max()>=length or idx.unique().numel()!=idx.numel():
            raise ValueError("Invalid or duplicate structure residue indices")
        enhanced=self.geometry.enhance(tokens,graph)
        if enhanced.shape!=tokens.shape:raise ValueError("GVP alignment mismatch")
        if not torch.isfinite(enhanced).all():raise ValueError("Nonfinite GVP enhancement")
        delta=enhanced-tokens
        coverage=tokens.new_full((batch,length,1),idx.numel()/length)
        x=torch.cat((
            original,delta.unsqueeze(0).expand(batch,-1,-1),
            compounds.unsqueeze(1).expand(-1,length,-1),coverage
        ),dim=-1)
        gates=torch.sigmoid(self.gate(x))
        mask=tokens.new_zeros(length,1)
        mask.index_fill_(0,idx,1)
        gates=gates*mask.unsqueeze(0)
        out=original+gates*delta.unsqueeze(0)
        return (out,gates) if return_gate else out
