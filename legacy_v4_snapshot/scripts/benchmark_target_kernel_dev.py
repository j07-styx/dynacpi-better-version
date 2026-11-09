"""Training-only leave-one-target-out selection of simple protein-transfer models."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_bounded_fusion_study_dev import metrics
OUT=ROOT/"research/target_kernel_20261107"
OUT.mkdir(exist_ok=False)
pilot=ROOT/"data/engineering_pilot"
manifest=json.loads((pilot/"limited_alpha_3d_graph_manifest.json").read_text())
ids={r["sequence_sha256"] for r in manifest["records"]}
rows=[r for r in json.loads((pilot/"rows.json").read_text()) if r["sequence"] in ids]
train_ids=sorted({r["sequence"] for r in rows if r["split"]=="train"})
val_ids=sorted({r["sequence"] for r in rows if r["split"]=="val"})
compounds=sorted({r["compound"] for r in rows})
lookup={(r["sequence"],r["compound"]):r for r in rows}
assert len(lookup)==len(rows)==len(compounds)*(len(train_ids)+len(val_ids))
Y=np.array([[lookup[(p,c)]["y"] for c in compounds] for p in train_ids])
features={}
for p in train_ids+val_ids:
    obj=torch.load(ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(p+".pt"),map_location="cpu",weights_only=True)
    assert obj["sequence_hash"]==p
    features[p]=obj["embeddings"].float().mean(0).numpy().astype(float)
X=np.stack([features[p] for p in train_ids])
V=np.stack([features[p] for p in val_ids])
def predict(x,y,q,kind,alpha):
    center=x.mean(0)
    a=x-center;b=q-center
    scale=np.sqrt(np.mean(np.sum(a*a,axis=1)))
    a=a/max(scale,1e-12);b=b/max(scale,1e-12)
    if kind=="linear":
        k=a@a.T;cross=b@a.T
    else:
        d=np.maximum(0,(a*a).sum(1)[:,None]+(a*a).sum(1)[None,:]-2*a@a.T)
        e=np.maximum(0,(b*b).sum(1)[:,None]+(a*a).sum(1)[None,:]-2*b@a.T)
        k=np.exp(-d/2);cross=np.exp(-e/2)
    mu=y.mean(0)
    return mu+cross@np.linalg.solve(k+alpha*np.eye(len(k)),y-mu)
candidates=[(kind,alpha) for kind in ("linear","rbf") for alpha in (.001,.01,.1,1.,10.,100.)]
cv=[]
for kind,alpha in candidates:
    errors=[]
    for hold in range(len(X)):
        keep=np.arange(len(X))!=hold
        pred=predict(X[keep],Y[keep],X[hold:hold+1],kind,alpha)[0]
        errors.append(float(np.sqrt(np.mean((pred-Y[hold])**2))))
    cv.append({"kind":kind,"alpha":alpha,"train_LOPO_macro_rmse":float(np.mean(errors))})
best=min(cv,key=lambda r:r["train_LOPO_macro_rmse"])
# Commit selection to disk before any validation affinity is read for scoring.
protocol={"training_targets":len(train_ids),"validation_targets":len(val_ids),"selection":best,
          "candidates":cv,"selection_uses_validation_labels":False,
          "rows_sha256":hashlib.sha256((pilot/"rows.json").read_bytes()).hexdigest()}
(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
pred=predict(X,Y,V,best["kind"],best["alpha"])
validation=[lookup[(p,c)] for p in val_ids for c in compounds]
mean=np.broadcast_to(Y.mean(0),pred.shape)
result={"selected_kernel":metrics(validation,pred.ravel()),"compound_mean":metrics(validation,mean.ravel()),
        "selection":best,"scope":"Existing repeatedly used development targets; NOT independent holdout"}
(OUT/"result.json").write_text(json.dumps(result,indent=2))
(OUT/"predictions_LOCAL.json").write_text(json.dumps([dict(r,pred=float(p)) for r,p in zip(validation,pred.ravel())]))
print(json.dumps(result,indent=2))
