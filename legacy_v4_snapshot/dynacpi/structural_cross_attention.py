"""Opt-in compound-query structural attention, retaining residue-specific information.
Development-only: no claim of benchmark improvement.
"""
import torch
from torch import nn


class StructuralCrossAttention(nn.Module):
    """Generate a compound-specific summary of geometry residuals.

    The compound query attends to covered residue corrections, rather than
    averaging centered corrections (which cancels at uniform weights).
    """
    def __init__(self, geometry, width=256, heads=4, scale=0.1):
        super().__init__()
        if width <= 0 or width % heads or scale <= 0:
            raise ValueError('Invalid attention width, heads or scale')
        self.geometry = geometry
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.project = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.project.weight, std=0.001)
        self.scale = scale

    def forward(self, tokens, graph, compounds):
        if tokens.ndim != 2 or tokens.shape[-1] != self.project.in_features:
            raise ValueError('Invalid token shape')
        if compounds.ndim != 2 or compounds.shape[-1] != tokens.shape[-1]:
            raise ValueError('Invalid compound shape')
        if tokens.device != compounds.device or tokens.dtype != compounds.dtype:
            raise ValueError('Compound and token device/dtype mismatch')
        if graph is None:
            return compounds
        idx = torch.as_tensor(graph['indices'], device=tokens.device, dtype=torch.long)
        if idx.ndim != 1 or not idx.numel() or int(idx.min()) < 0 or int(idx.max()) >= len(tokens) or idx.unique().numel() != idx.numel():
            raise ValueError('Invalid residue indices')
        enhanced = self.geometry.enhance(tokens, graph)
        if enhanced.shape != tokens.shape or not torch.isfinite(enhanced).all():
            raise ValueError('Invalid geometry enhancement')
        residual = (enhanced - tokens).index_select(0, idx)
        key = tokens.index_select(0, idx).unsqueeze(0).expand(len(compounds), -1, -1)
        value = residual.unsqueeze(0).expand(len(compounds), -1, -1)
        summary = self.attention(compounds.unsqueeze(1), key, value, need_weights=False)[0][:, 0]
        result = compounds + self.scale * self.project(summary)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite structural summary')
        return result
