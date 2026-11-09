"""Check mean cancellation and attention-path sensitivity of centered residuals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from dynacpi.centered_geometry import CenteredGeometry
from dynacpi.compound_conditioned_geometry import CompoundConditionedGeometry

class FixedGeometry(nn.Module):
    def enhance(self, tokens, graph):
        out = tokens.clone()
        out[graph['indices']] += graph['offsets']
        return out

torch.manual_seed(221)
width = 8
length = 6
batch = 3
idx = torch.tensor([0, 2, 3, 5])
graph = {'indices':idx, 'offsets':torch.randn(len(idx),width)}
tokens = torch.randn(length,width)
compounds = torch.randn(batch,width)
model = CompoundConditionedGeometry(CenteredGeometry(FixedGeometry()),width=width,hidden=12,gate_bias=-3.5)
model.eval()
with torch.no_grad():
    altered, gates = model(tokens,graph,compounds,return_gate=True)
    original = tokens.unsqueeze(0).expand_as(altered)
    correction = altered-original
    assert torch.allclose(correction.sum(1),torch.zeros(batch,width),atol=2e-6)
    assert torch.allclose(altered.mean(1),original.mean(1),atol=2e-6)
    assert torch.allclose(gates[:,idx],torch.sigmoid(torch.tensor(-3.5)).expand(batch,len(idx),1))
    assert torch.equal(altered[:,[1,4]],original[:,[1,4]])
    # A residue-sensitive attention readout can still detect a zero-mean correction.
    attn = nn.MultiheadAttention(width,2,batch_first=True,dropout=0)
    query = compounds.unsqueeze(1)
    old_attn = attn(query,original,original,need_weights=False)[0]
    new_attn = attn(query,altered,altered,need_weights=False)[0]
    attn_shift = (new_attn-old_attn).abs().max().item()
    assert attn_shift>1e-7, 'Attention should respond to nonuniform zero-mean changes'
print('PASS centered geometry: mean-pool cancellation, constant initial gate, and attention readout sensitivity')
