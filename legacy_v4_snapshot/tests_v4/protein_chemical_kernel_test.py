import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from experiment_protein_chemical_kernel_dev import protein_kernel
x=np.array([[0.,0.],[1.,0.],[0.,2.],[4.,4.]])
k=protein_kernel(x,[0,1,2])
assert np.allclose(k,k.T) and np.allclose(np.diag(k),1)
assert np.linalg.eigvalsh(k).min()>-1e-10
altered=x.copy();altered[3]=100
assert np.allclose(k[:3,:3],protein_kernel(altered,[0,1,2])[:3,:3])
assert np.allclose(k,protein_kernel(x*3,[0,1,2]))
r=json.loads((ROOT/"research/protein_chemical_kernel_20261107/results.json").read_text())
assert len(r["folds"])==9 and sum(f["query_count"] for f in r["folds"])==106
assert all(len(f["shuffled_hybrid_rmse"])==20 for f in r["folds"])
assert all(np.isfinite(v) for f in r["folds"] for v in list(f["rmse"].values())+f["shuffled_hybrid_rmse"])
for model in ("chemical","protein","product","half_chemical_half_product"):
    assert np.isclose(r["macro_rmse"][model],np.mean([f["rmse"][model] for f in r["folds"]]))
print("PASS PSD kernel, train-only bandwidth, scale invariance, 9 folds/106 queries and aggregate consistency")
