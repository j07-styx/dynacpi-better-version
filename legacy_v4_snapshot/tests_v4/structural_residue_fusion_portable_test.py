"""Portable regression tests for residue-level fusion, without local datasets."""
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynacpi.structural_residue_fusion_dev import StructuralResidueFusion


class MockGeometry(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.delta = nn.Parameter(torch.linspace(-0.2, 0.2, width))

    def enhance(self, tokens, graph):
        result = tokens.clone()
        idx = torch.as_tensor(graph['indices'], dtype=torch.long, device=tokens.device)
        return result.index_add(0, idx, self.delta.expand(idx.numel(), -1))


def main():
    torch.manual_seed(2103)
    model = StructuralResidueFusion(MockGeometry())
    tokens = torch.randn(8, 256, requires_grad=True)
    graph = {'indices': torch.tensor([1, 3, 6])}
    assert torch.equal(model(tokens, None), tokens), 'missing graph must be exact identity'
    output = model(tokens, graph)
    assert output.shape == tokens.shape
    assert torch.isfinite(output).all()
    untouched = torch.tensor([0, 2, 4, 5, 7])
    assert torch.equal(output[untouched], tokens[untouched]), 'uncovered residues changed'
    assert (output[graph['indices']] - tokens[graph['indices']]).abs().sum() > 0
    loss = output.square().mean()
    loss.backward()
    assert model.project.weight.grad is not None
    assert torch.isfinite(model.project.weight.grad).all()
    assert model.project.weight.grad.norm() > 0
    assert model.geometry.delta.grad is not None
    assert torch.isfinite(model.geometry.delta.grad).all()
    assert model.geometry.delta.grad.norm() > 0
    assert torch.isfinite(tokens.grad).all()
    print('PASS portable residue fusion: identity, sparse support, shape, finiteness, gradients')


if __name__ == '__main__':
    main()
