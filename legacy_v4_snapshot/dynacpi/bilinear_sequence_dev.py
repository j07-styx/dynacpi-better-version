"""Opt-in low-rank compound/protein bilinear interaction for sequence-only CPI.
Keeps the base encoder, adds a residual interaction term to normalized pKd.
"""
import torch
from torch import nn
from dynacpi.sequence_model import SequenceFoundation

class BilinearSequenceFoundation(SequenceFoundation):
    def __init__(self, device="cuda", width=256, rank=32):
        super().__init__(device=device,width=width)
        self.compound_projection=nn.Linear(width,rank,bias=False)
        self.protein_projection=nn.Linear(width,rank,bias=False)
        self.interaction_head=nn.Linear(rank,1,bias=False)
        self.interaction_norm=nn.LayerNorm(rank)
        nn.init.normal_(self.compound_projection.weight,std=0.02)
        nn.init.normal_(self.protein_projection.weight,std=0.02)
        nn.init.normal_(self.interaction_head.weight,std=0.01)

    def predict_tokens(self,c,tokens,valid):
        if not torch.all(valid.any(dim=1)):
            raise ValueError("Empty construct")
        residue=self.readout(c.unsqueeze(1),tokens,tokens,
                             key_padding_mask=~valid,need_weights=False)[0][:,0]
        mean=(tokens*valid.unsqueeze(-1)).sum(dim=1)/valid.sum(dim=1,keepdim=True)
        protein=self.norm(mean+residue)
        baseline=self.head(self.channel(c,protein,protein)).squeeze(-1)
        compound=torch.tanh(self.compound_projection(c))
        target=torch.tanh(self.protein_projection(protein))
        interaction=self.interaction_head(self.interaction_norm(compound*target)).squeeze(-1)
        return baseline+0.1*interaction
