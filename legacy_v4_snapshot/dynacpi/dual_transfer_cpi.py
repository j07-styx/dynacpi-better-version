"""Dual pretrained encoders: adds frozen chemical language features to TransferCPI."""
import torch
from torch import nn
from dynacpi.transfer_cpi import TransferCPI,masked_mean
class DualTransferCPI(TransferCPI):
    def __init__(self,width=96):
        super().__init__(width=width)
        self.chemical_language=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,width),nn.GELU())
        self.chemical_head=nn.Sequential(nn.Linear(width*4,width),nn.GELU(),nn.Dropout(.2),nn.Linear(width,1))
    def encode_pair(self,batch,chemical_only=False):
        """Return chemical score and joint features without changing parameter names."""
        z,features,adj,mask=batch["z"],batch["atom_features"],batch["adj"],batch["atom_mask"]
        pmask=batch["protein_mask"]
        if not mask.any(1).all() or not pmask.any(1).all():raise ValueError("Empty molecule or protein")
        language=self.chemical_language(batch["chemical_language"])
        d=(self.atom(z)+self.atom_features(features))*mask.unsqueeze(-1)
        for block in self.graph:d=block(d,adj,mask)
        dg=masked_mean(d,mask)
        chem=self.chemical_head(torch.cat([dg,self.properties(batch["properties"]),self.fingerprint(batch["fp"]),language],-1)).flatten()
        if chemical_only:return chem,None
        d=(d+language.unsqueeze(1))*mask.unsqueeze(-1)
        p=self.protein(batch["protein"])*pmask.unsqueeze(-1)
        dp,_=self.drug_to_protein(d,p,p,key_padding_mask=~pmask,need_weights=False)
        pd,_=self.protein_to_drug(p,d,d,key_padding_mask=~mask,need_weights=False)
        dlocal=masked_mean(self.drug_norm(d+dp),mask)
        plocal=masked_mean(self.protein_norm(p+pd),pmask)
        return chem,torch.cat([dlocal,plocal,dlocal*plocal,(dlocal-plocal).abs()],-1)
    def forward(self,batch,chemical_only=False):
        chem,pair=self.encode_pair(batch,chemical_only=chemical_only)
        if chemical_only:return chem,torch.zeros_like(chem)
        interaction=self.interaction_head(pair).flatten()
        return chem+interaction,interaction
