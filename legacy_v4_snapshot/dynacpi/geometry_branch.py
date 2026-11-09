import torch
from torch import nn
from gvp import GVP,GVPConvLayer,LayerNorm

class GeometryBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes=nn.Sequential(LayerNorm((14,3)),GVP((14,3),(128,16)))
        self.edges=nn.Sequential(LayerNorm((37,1)),GVP((37,1),(32,1)))
        self.layers=nn.ModuleList(GVPConvLayer((128,16),(32,1),drop_rate=.1) for _ in range(3))
        self.scalars=GVP((128,16),(128,0),activations=(None,None))
        self.final=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,256))
        nn.init.zeros_(self.final[-1].weight);nn.init.zeros_(self.final[-1].bias)

    def enhance(self,tokens,graph):
        """One construct, [L,256]; missing geometry gives exact identity."""
        if graph is None:return tokens
        g={k:v.to(tokens.device) for k,v in graph.items()}
        h=self.nodes((g["node_s"],g["node_v"]));e=self.edges((g["edge_s"],g["edge_v"]))
        for layer in self.layers:h=layer(h,g["edges"],e)
        d=self.final(self.scalars(h))
        residual=torch.zeros_like(tokens).index_copy(0,g["indices"],d)
        return tokens+residual
