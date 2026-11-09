"""TransferCPI: frozen protein tokens + bond-aware atom messages + local cross-attention.

Original implementation inspired by multimodal interaction modeling; not a
GenSPARC reproduction, structure-aware model, or validated superiority claim.
"""
import torch
from torch import nn

def masked_mean(x,mask):
    return (x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

class BondBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.bond=nn.ModuleList([nn.Linear(width,width,bias=False) for _ in range(4)])
        self.update=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width)
    def forward(self,x,adj,mask):
        msg=sum(torch.bmm(adj[:,i],layer(x)) for i,layer in enumerate(self.bond))
        degree=adj.sum((1,3)).clamp_min(1).unsqueeze(-1)
        return self.norm(x+self.update(torch.cat([x,msg/degree],-1)))*mask.unsqueeze(-1)

class TransferCPI(nn.Module):
    def __init__(self,width=96,protein_dim=1280,fp_dim=1024,property_dim=12):
        super().__init__()
        self.atom=nn.Embedding(119,width,padding_idx=0)
        self.atom_features=nn.Linear(10,width,bias=False)
        self.graph=nn.ModuleList([BondBlock(width) for _ in range(3)])
        self.protein=nn.Sequential(nn.LayerNorm(protein_dim),nn.Linear(protein_dim,width),nn.GELU(),nn.LayerNorm(width))
        self.drug_to_protein=nn.MultiheadAttention(width,4,dropout=.1,batch_first=True)
        self.protein_to_drug=nn.MultiheadAttention(width,4,dropout=.1,batch_first=True)
        self.drug_norm=nn.LayerNorm(width);self.protein_norm=nn.LayerNorm(width)
        self.properties=nn.Sequential(nn.Linear(property_dim,width),nn.GELU())
        self.fingerprint=nn.Sequential(nn.Linear(fp_dim,width),nn.GELU(),nn.Dropout(.2))
        self.chemical_head=nn.Sequential(nn.Linear(width*3,width),nn.GELU(),nn.Dropout(.2),nn.Linear(width,1))
        self.interaction_head=nn.Sequential(nn.Linear(width*4,width),nn.GELU(),nn.Dropout(.2),nn.Linear(width,1))
        # Small nonzero initialization allows gradients through both branches immediately.
        nn.init.normal_(self.interaction_head[-1].weight,std=.001)
        nn.init.zeros_(self.interaction_head[-1].bias)
    def forward(self,batch,chemical_only=False):
        z,features,adj,mask=batch["z"],batch["atom_features"],batch["adj"],batch["atom_mask"]
        pmask=batch["protein_mask"]
        if not mask.any(1).all() or not pmask.any(1).all():raise ValueError("Empty molecule or protein")
        d=(self.atom(z)+self.atom_features(features))*mask.unsqueeze(-1)
        for block in self.graph:d=block(d,adj,mask)
        dg=masked_mean(d,mask)
        chem=self.chemical_head(torch.cat([dg,self.properties(batch["properties"]),self.fingerprint(batch["fp"])],-1)).flatten()
        if chemical_only:return chem,torch.zeros_like(chem)
        p=self.protein(batch["protein"])*pmask.unsqueeze(-1)
        dp,_=self.drug_to_protein(d,p,p,key_padding_mask=~pmask,need_weights=False)
        pd,_=self.protein_to_drug(p,d,d,key_padding_mask=~mask,need_weights=False)
        dlocal=masked_mean(self.drug_norm(d+dp),mask)
        plocal=masked_mean(self.protein_norm(p+pd),pmask)
        interaction=self.interaction_head(torch.cat([dlocal,plocal,dlocal*plocal,(dlocal-plocal).abs()],-1)).flatten()
        return chem+interaction,interaction
