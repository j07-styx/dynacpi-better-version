"""DeepDTA architecture smoke test: two 3-layer CNN encoders, paper-size FC head.
Research reproduction component, not yet a benchmark reproduction.
"""
import torch
from torch import nn

class ConvEncoder(nn.Module):
    def __init__(self, vocab, kernel):
        super().__init__()
        self.embed=nn.Embedding(vocab,128,padding_idx=0)
        self.conv=nn.Sequential(nn.Conv1d(128,32,kernel),nn.ReLU(),nn.Conv1d(32,64,kernel),nn.ReLU(),nn.Conv1d(64,96,kernel),nn.ReLU(),nn.AdaptiveMaxPool1d(1))
    def forward(self,x):return self.conv(self.embed(x).transpose(1,2)).squeeze(-1)

class DeepDTA(nn.Module):
    def __init__(self,drug_vocab=128,protein_vocab=32):
        super().__init__()
        self.drug=ConvEncoder(drug_vocab,4)
        self.protein=ConvEncoder(protein_vocab,8)
        self.head=nn.Sequential(nn.Linear(192,1024),nn.ReLU(),nn.Dropout(.1),nn.Linear(1024,1024),nn.ReLU(),nn.Dropout(.1),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,1))
    def forward(self,drug,protein):return self.head(torch.cat([self.drug(drug),self.protein(protein)],dim=-1)).squeeze(-1)

if __name__=='__main__':
    model=DeepDTA().to('cuda' if torch.cuda.is_available() else 'cpu')
    device=next(model.parameters()).device
    d=torch.randint(1,80,(3,100),device=device)
    p=torch.randint(1,25,(3,1000),device=device)
    y=model(d,p)
    assert y.shape==(3,) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert model.protein.conv[0].weight.grad.abs().sum()>0
    assert model.drug.conv[0].weight.grad.abs().sum()>0
    print('PASS DeepDTA 2 CNN branches, 32/64/96 filters, 1024/1024/512 head; gradient flow on',device)
