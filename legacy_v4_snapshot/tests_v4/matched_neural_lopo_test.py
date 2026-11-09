"""Held-out labels must have no influence on training or predictions."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from experiment_matched_neural_lopo_dev import fit
torch.set_num_threads(2)
torch.manual_seed(17)
c=torch.randn(8,2048);p=torch.randn(8,1280);y=torch.arange(8,dtype=torch.float32)
train=np.arange(6);test=np.array([6,7])
a,loss=fit(c,p,y,train,test,19,epochs=20)
changed=y.clone();changed[test]=99999
b,_=fit(c,p,changed,train,test,19,epochs=20)
assert np.array_equal(a,b),"Held-out-label leakage"
assert np.isfinite(a).all()
assert loss["final_standardized_mse"]<loss["initial_standardized_mse"]
print("PASS held-out-label independence, deterministic fits, finite predictions and train-loss decrease")
