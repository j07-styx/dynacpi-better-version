"""Development fusion: bounded direct affinity correction with geometry controls."""
import math
import torch
from torch import nn

class BoundedStructuralFusion(nn.Module):
    def __init__(self, width=256, hidden=64, cap=0.25, confidence=True):
        super().__init__()
        if cap <= 0: raise ValueError("cap must be positive")
        self.cap, self.confidence = cap, confidence
        self.seq = nn.Linear(width, hidden, bias=False)
        self.query = nn.Linear(width, hidden, bias=False)
        # Invariant scalar node features and vector magnitudes; no orientation dependence.
        self.geometry = nn.Sequential(nn.Linear(17, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Sequential(nn.Linear(hidden*3, hidden), nn.SiLU(), nn.Dropout(.1), nn.Linear(hidden, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens, graph, compounds, mode="real", return_attention=False):
        if mode not in ("real", "sequence", "disabled"):
            raise ValueError(mode)
        if graph is None or mode == "disabled":
            zero = compounds[:, 0]*0
            return (zero, None) if return_attention else zero
        idx = graph["indices"].to(tokens.device)
        sequence = self.seq(tokens.index_select(0, idx))
        node_s = graph["node_s"].to(tokens.device)
        node_v = graph["node_v"].to(tokens.device)
        if mode == "sequence":
            structural = torch.zeros_like(sequence)
        else:
            structural = self.geometry(torch.cat((node_s, node_v.norm(dim=-1)), dim=-1))
        q = self.query(compounds)
        weights = torch.softmax(q @ self.norm(sequence+structural).T / math.sqrt(q.shape[-1]), dim=-1)
        if self.confidence and mode != "sequence":
            confidence = torch.where(node_s[:, 13] > .5, node_s[:, 12].clamp(0, 1), torch.ones_like(node_s[:, 12]))
            structural = structural*confidence[:, None]
        summary = weights @ self.norm(sequence+structural)
        correction = self.cap*torch.tanh(self.output(torch.cat((q, summary, q*summary), dim=-1)).squeeze(-1))
        return (correction, weights) if return_attention else correction
