"""Trainable dual-pretrained correction to frozen Perceiver predictions.

Untrained architecture, not an accuracy claim. Frozen ESM/ChemBERTa inputs use
the existing feature contract; actual fitting needs a separate approved split.
Historical failed TransferCPI weights are NOT loaded by default.
"""
import torch
from torch import nn
from dynacpi.dual_transfer_cpi import DualTransferCPI
from dynacpi.perceiver_anchored_residual import PerceiverAnchoredResidual

class PerceiverAnchoredTransfer(nn.Module):
    def __init__(self,width=96,hidden=64,cap=.5):
        super().__init__()
        self.encoder=DualTransferCPI(width=width)
        # The legacy standalone interaction readout is not used by this model.
        self.encoder.interaction_head.requires_grad_(False)
        self.correction=PerceiverAnchoredResidual(width*4+1,hidden=hidden,cap=cap)

    def forward(self,batch,perceiver_pkd,availability=None,return_delta=False):
        chemical,pair=self.encoder.encode_pair(batch)
        features=torch.cat([pair,chemical.unsqueeze(-1)],dim=-1)
        return self.correction(perceiver_pkd,features,availability,return_delta)

    def parameter_counts(self):
        return {"total":sum(p.numel() for p in self.parameters()),
                "trainable":sum(p.numel() for p in self.parameters() if p.requires_grad)}
