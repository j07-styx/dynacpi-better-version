"""Compact bounded residual chemical-protein CNN with explicit prior shrinkage."""
import torch
from torch import nn
from dynacpi.deepdta_reference_dev import ConvEncoder
class ResidualDeepDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug=ConvEncoder(128,4)
        self.protein=ConvEncoder(32,8)
        self.head=nn.Sequential(nn.Linear(192,256),nn.ReLU(),nn.Dropout(.35),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.35),nn.Linear(128,1))
    def forward(self,drug,protein):
        raw=self.head(torch.cat([self.drug(drug),self.protein(protein)],dim=-1)).squeeze(-1)
        return 0.5*torch.tanh(raw)
