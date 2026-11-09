"""Small deterministic chemical baseline for paired CPI experiments.

This model ignores protein identity. Artifacts must not be represented as
protein-specific binding predictions or source-approved scientific models.
"""
import json
from pathlib import Path
import numpy as np

def binary_matrix(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not x.shape[0] or not x.shape[1]:
        raise ValueError("Expected nonempty 2D fingerprints")
    if not np.isfinite(x).all() or not np.isin(x,[0,1]).all():
        raise ValueError("Fingerprints must contain only binary values")
    return x

def tanimoto(a,b):
    a=binary_matrix(a);b=binary_matrix(b)
    if a.shape[1]!=b.shape[1]:raise ValueError("Fingerprint width mismatch")
    intersection=a@b.T
    union=a.sum(1)[:,None]+b.sum(1)[None,:]-intersection
    return np.divide(intersection,union,out=np.ones_like(intersection),where=union!=0)

class TanimotoRegressor:
    def __init__(self,alpha=1.0):
        if not np.isfinite(alpha) or alpha<=0:raise ValueError("alpha must be positive")
        self.alpha=float(alpha)
    def fit(self,x,y):
        x=binary_matrix(x);y=np.asarray(y,dtype=float)
        if y.shape!=(len(x),) or not np.isfinite(y).all():raise ValueError("Invalid labels")
        self.x=x.copy();self.mean=float(y.mean())
        self.weights=np.linalg.solve(tanimoto(x,x)+self.alpha*np.eye(len(x)),y-self.mean)
        return self
    def predict(self,x):
        if not hasattr(self,"weights"):raise RuntimeError("Model is not fitted")
        return self.mean+tanimoto(x,self.x)@self.weights
    def save(self,path,metadata):
        if not hasattr(self,"weights"):raise RuntimeError("Model is not fitted")
        if not isinstance(metadata,dict):raise ValueError("Metadata required")
        payload={"schema":1,"model":"protein_agnostic_tanimoto_ridge","alpha":self.alpha,
                 "mean":self.mean,"fingerprints":self.x.astype(int).tolist(),
                 "weights":self.weights.tolist(),"metadata":metadata}
        with Path(path).open("x") as f:json.dump(payload,f)
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if d["schema"]!=1 or d["model"]!="protein_agnostic_tanimoto_ridge":raise ValueError("Unknown artifact")
        model=cls(d["alpha"]);model.x=binary_matrix(d["fingerprints"])
        model.mean=float(d["mean"]);model.weights=np.asarray(d["weights"],dtype=float)
        if model.weights.shape!=(len(model.x),) or not np.isfinite(model.weights).all() or not np.isfinite(model.mean):
            raise ValueError("Invalid model state")
        return model
