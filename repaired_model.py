"""DynaCPI v4.2 two-bypass candidate; isolated from the original model.

Same supervised architecture tested in causal_diagnosis_20261109/Repair:
- direct LayerNorm(fingerprint projection + molecular graph embedding)
- direct MLP([compound, protein, compound*protein, |compound-protein|])
This model changes neither the original SequenceFoundation class nor its weights.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from chemprop.features import mol2graph

from dynacpi.sequence_model import SequenceFoundation


class RepairedSequenceFoundation(SequenceFoundation):
    def __init__(self, width=256, device="cuda"):
        if width != 256:
            raise ValueError("Only the fixed/previously tested width of 256 is supported.")
        super().__init__(device=device, width=width)
        self.chemical_norm = nn.LayerNorm(width)
        self.pair_head = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def compound(self, smiles, fingerprints):
        graph = self.mpn([mol2graph(list(smiles))])
        return self.chemical_norm(self.fp(fingerprints) + graph)

    def predict_tokens(self, c, tokens, valid):
        if not torch.all(valid.any(dim=1)):
            raise ValueError("Empty protein sequence.")
        residue = self.readout(
            c.unsqueeze(1), tokens, tokens,
            key_padding_mask=~valid, need_weights=False,
        )[0][:, 0]
        mean = (tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        protein = self.norm(mean + residue)
        combined = torch.cat((c, protein, c * protein, (c - protein).abs()), dim=-1)
        return self.pair_head(combined).squeeze(-1)
