import sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.tanimoto_regressor import TanimotoRegressor,tanimoto
x=np.eye(2);y=np.array([1.,3.])
m=TanimotoRegressor().fit(x,y)
assert np.allclose(m.predict(x),[1.5,2.5])
assert np.allclose(tanimoto([[1,1,0]],[[1,0,1]]),[[1/3]])
assert np.allclose(tanimoto([[0,0]],[[0,0]]),1)
assert np.allclose(TanimotoRegressor().fit(x,y+7).predict(x),m.predict(x)+7)
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/"model.json";m.save(p,{"scope":"synthetic test"})
 assert np.array_equal(TanimotoRegressor.load(p).predict(x),m.predict(x))
 try:m.save(p,{})
 except FileExistsError:pass
 else:raise AssertionError("Overwrote artifact")
for bad in ([[2,0]],[[float("nan"),0]]):
 try:m.predict(bad)
 except ValueError:pass
 else:raise AssertionError("Invalid features accepted")
print("PASS analytic ridge, Tanimoto, label translation, safe roundtrip, overwrite protection and malformed inputs")
