"""Integrated architecture behavior with artificial features and labels only."""
import sys,unittest,io
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from dynacpi.perceiver_anchored_transfer import PerceiverAnchoredTransfer
from dynacpi.transfer_batch import TransferCollator
from prepare_transfer_compounds_dev import featurize

class AnchoredTransferTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2613)
        graphs={"a":featurize("CCO"),"b":featurize("c1ccccc1")}
        for g in graphs.values():g["chemical_language"]=torch.randn(384)
        self.collate=TransferCollator(graphs,{"p":torch.randn(7,1280),"q":torch.randn(9,1280)},torch.zeros(12),torch.ones(12))
        self.rows=[dict(compound="a",protein="p",y=0.),dict(compound="b",protein="q",y=0.)]
        self.batch=self.collate(self.rows);self.base=torch.tensor([6.,7.],requires_grad=True)
        self.model=PerceiverAnchoredTransfer().eval()
    def activate(self):
        with torch.no_grad():self.model.correction.head[-1].weight.normal_(std=.02)
    def test_exact_baseline_initialization_and_no_baseline_gradient(self):
        pred,delta=self.model(self.batch,self.base,return_delta=True)
        self.assertTrue(torch.equal(pred,self.base))
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        pred.square().mean().backward();self.assertIsNone(self.base.grad)
        self.assertGreater(self.model.correction.head[-1].weight.grad.abs().sum(),0)
    def test_encoder_receives_gradients_after_first_update(self):
        opt=torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad],lr=.01)
        target=torch.tensor([6.2,6.8])
        loss=(self.model(self.batch,self.base)-target).square().mean()
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        (self.model(self.batch,self.base)-target).square().mean().backward()
        for name in ("protein.1.weight","chemical_language.1.weight","atom.weight","drug_to_protein.in_proj_weight"):
            g=dict(self.model.encoder.named_parameters())[name].grad
            self.assertIsNotNone(g,name);self.assertTrue(torch.isfinite(g).all(),name)
            self.assertGreater(g.abs().sum(),0,name)
        self.assertTrue(all(p.grad is None for p in self.model.encoder.interaction_head.parameters()))
    def test_cap_and_unavailable_exact_fallback(self):
        with torch.no_grad():self.model.correction.head[-1].bias.fill_(100.)
        pred,delta=self.model(self.batch,self.base,torch.tensor([0.,1.]),True)
        self.assertEqual(pred[0],self.base[0]);self.assertTrue((delta.abs()<=.5).all())
    def test_batch_padding_and_order_invariance_after_activation(self):
        self.activate()
        with torch.no_grad():
            pred=self.model(self.batch,self.base)
            one=self.model(self.collate(self.rows[:1]),self.base[:1])
            reverse=self.model(self.collate(self.rows[::-1]),self.base.flip(0))
            changed={k:v.clone() for k,v in self.batch.items()}
            changed["protein"][~changed["protein_mask"]]=99.
            changed["atom_features"][~changed["atom_mask"]]=99.
            self.assertTrue(torch.allclose(pred[:1],one,atol=2e-6))
            self.assertTrue(torch.allclose(pred,reverse.flip(0),atol=2e-6))
            self.assertTrue(torch.allclose(pred,self.model(changed,self.base),atol=2e-6))
    def test_state_roundtrip(self):
        self.activate();buf=io.BytesIO();torch.save(self.model.state_dict(),buf);buf.seek(0)
        other=PerceiverAnchoredTransfer().eval();other.load_state_dict(torch.load(buf,weights_only=True))
        with torch.no_grad():self.assertTrue(torch.equal(self.model(self.batch,self.base),other(self.batch,self.base)))
    def test_wrong_baseline_shape_fails(self):
        with self.assertRaises(ValueError):self.model(self.batch,self.base[:,None])
if __name__=="__main__":unittest.main(verbosity=2)
