import math
from types import SimpleNamespace
import torch
from torch import nn
from chemprop.features import mol2graph
from dynacpi.mpn import MPN

class ChannelAttention(nn.Module):
    """One-head pooled channel attention, matching the upstream CAB operation."""
    def __init__(self, width, dropout):
        super().__init__()
        self.q,self.k,self.v,self.out=[nn.Linear(width,width) for _ in range(4)]
        self.drop=nn.Dropout(dropout)
        self.scale=math.sqrt(width)

    def forward(self,query,key,value):
        q=self.q(query).unsqueeze(-1)
        k=self.k(key).unsqueeze(-2)
        v=self.v(value).unsqueeze(-1)
        weights=self.drop(torch.softmax(q@k/self.scale,dim=-1))
        return self.drop(self.out((weights@v).squeeze(-1)))

class SequenceFoundation(nn.Module):
    def __init__(self,device="cuda",width=256):
        super().__init__()
        args=SimpleNamespace(atom_messages=False,hidden_size=width,bias=False,depth=3,dropout=.1,undirected=False,
            device=torch.device(device),aggregation="mean",aggregation_norm=100,activation="ReLU",atom_descriptors=None,
            overwrite_default_atom_features=False,overwrite_default_bond_features=False,features_only=False,
            use_input_features=False,mpn_shared=False,number_of_molecules=1)
        self.mpn=MPN(args)
        self.fp=nn.Sequential(nn.Linear(2048,width),nn.ReLU(),nn.Dropout(.1))
        self.channel=ChannelAttention(width,.1)
        self.sequence=nn.Sequential(nn.Linear(1280,width),nn.LayerNorm(width))
        self.readout=nn.MultiheadAttention(width,4,dropout=.1,batch_first=True)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Dropout(.1),nn.Linear(width,1))

    def compound(self,smiles,fingerprints):
        graph=self.mpn([mol2graph(list(smiles))])
        low=graph.amin(dim=-1,keepdim=True)
        graph=(graph-low)/(graph.amax(dim=-1,keepdim=True)-low).clamp_min(1e-6)
        mixed=graph+self.channel(self.fp(fingerprints),graph,graph)
        return self.channel(mixed,mixed,mixed)

    def predict_tokens(self,c,tokens,valid):
        if not torch.all(valid.any(dim=1)):raise ValueError("Empty construct")
        residue=self.readout(c.unsqueeze(1),tokens,tokens,key_padding_mask=~valid,need_weights=False)[0][:,0]
        mean=(tokens*valid.unsqueeze(-1)).sum(dim=1)/valid.sum(dim=1,keepdim=True)
        protein=self.norm(mean+residue)
        return self.head(self.channel(c,protein,protein)).squeeze(-1)

    def forward(self,smiles,fingerprints,embeddings,valid):
        return self.predict_tokens(self.compound(smiles,fingerprints),self.sequence(embeddings.float()),valid)
