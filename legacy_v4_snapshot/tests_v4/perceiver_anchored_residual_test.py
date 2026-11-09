"""Synthetic mathematical invariants for baseline-anchored DynaCPI head."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.perceiver_anchored_residual import PerceiverAnchoredResidual


class BaselineAnchoredResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2601)
        self.module=PerceiverAnchoredResidual(8,hidden=12,cap=0.35)
        self.base=torch.tensor([5.1,7.4,6.5,8.0],requires_grad=True)
        self.features=torch.randn(4,8)

    def test_baseline_identity_exact_at_initialization(self):
        out,delta=self.module(self.base,self.features,return_delta=True)
        self.assertTrue(torch.equal(out,self.base))
        self.assertTrue(torch.count_nonzero(delta)==0)

    def test_zero_availability_exact_fallback_after_nonzero_training_step(self):
        with torch.no_grad():
            self.module.head[-1].weight.fill_(0.6)
            self.module.head[-1].bias.fill_(0.3)
        out,delta=self.module(self.base,self.features,
                              availability=torch.zeros(4),return_delta=True)
        self.assertTrue(torch.equal(out,self.base))
        self.assertTrue(torch.count_nonzero(delta)==0)

    def test_bounded_correction(self):
        with torch.no_grad():
            self.module.head[-1].bias.fill_(100.0)
        output,delta=self.module(self.base,self.features,return_delta=True)
        self.assertLessEqual(float(delta.abs().max()),0.35)
        self.assertTrue(torch.allclose(output,self.base+delta))

    def test_partial_availability_scales_delta(self):
        with torch.no_grad():
            self.module.head[-1].bias.fill_(0.4)
        _,full=self.module(self.base,self.features,return_delta=True)
        mask=torch.tensor([0.,0.5,1.,0.2])
        _,partial=self.module(self.base,self.features,availability=mask,return_delta=True)
        self.assertTrue(torch.allclose(partial,full*mask))

    def test_frozen_base_and_trainable_last_layer(self):
        pred=self.module(self.base,self.features)
        loss=pred.sum()
        loss.backward()
        self.assertIsNone(self.base.grad)
        self.assertIsNotNone(self.module.head[-1].weight.grad)
        self.assertGreater(float(self.module.head[-1].weight.grad.abs().sum()),0)

    def test_invalid_dimensions_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError,"dimension mismatch"):
            self.module(self.base,self.features[:3])
        invalid=self.features.clone()
        invalid[0,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"Non-finite"):
            self.module(self.base,invalid)
        with self.assertRaisesRegex(ValueError,"Availability"):
            self.module(self.base,self.features,availability=torch.tensor([1.,-1.,0.,1.]))

    def test_invalid_hyperparameters(self):
        with self.assertRaises(ValueError):
            PerceiverAnchoredResidual(0)
        with self.assertRaises(ValueError):
            PerceiverAnchoredResidual(8,cap=-1)

if __name__=="__main__":
    unittest.main(verbosity=2)
