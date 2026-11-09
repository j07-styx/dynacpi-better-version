"""Experimental residue-level structural fusion; preserves compound representation.
Unlike compound-summary fusion, modifies residue tokens before protein readout.
"""
import torch
from torch import nn

class StructuralResidueFusion(nn.Module):
    def __init__(self, geometry, width=256, scale=0.1):
        super().__init__()
        self.geometry=geometry
        self.norm=nn.LayerNorm(width)
        self.project=nn.Linear(width,width,bias=False)
        self.scale=scale
        nn.init.normal_(self.project.weight,std=0.02)
    def forward(self,tokens,graph):
        if graph is None:return tokens
        enhanced=self.geometry.enhance(tokens,graph)
        residual=enhanced-tokens
        return tokens+self.scale*self.project(self.norm(residual))*(residual.abs().sum(dim=-1,keepdim=True)>0).to(tokens.dtype)
