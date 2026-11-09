"""Train-only global/compound-mean controls evaluated on inspected DAVIS DEV.

Matched chemistry panel (54 known compounds). No fitting to DEV, no models
selected from DEV, and no claim of novel-compound generalization.
"""
import json, math
from pathlib import Path
import numpy as np
from train_repaired_full import HERE, OLD, OUT, dump, sha_file
rows=json.loads((OLD/"data/engineering_pilot/rows.json").read_text(encoding="utf-8"))
tr=[r for r in rows if r["split"]=="train"]
dev=[r for r in rows if r["split"]=="val"]
compounds=sorted({r["compound"] for r in rows})
trainids=sorted({r["sequence"] for r in tr})
devids=sorted({r["sequence"] for r in dev})
assert len(tr)==1296 and len(dev)==432 and len(compounds)==54
def block(records, protein):
    mapping={r["compound"]:float(r["y"]) for r in records if r["sequence"]==protein}
    assert set(mapping)==set(compounds)
    return np.asarray([mapping[c] for c in compounds],dtype=np.float64)
train=np.stack([block(tr,p) for p in trainids])
val=np.stack([block(dev,p) for p in devids])
compound=train.mean(axis=0)
global_prediction=np.full(54,float(train.mean()))
names={}
for r in json.loads((OLD/"data/engineering_pilot/limited_alpha_3d_graph_manifest.json").read_text(encoding="utf-8"))["records"]:
    names[r["sequence_sha256"]]=r["source_name"]
data={}
for ix,p in enumerate(devids):
    y=val[ix]
    data[names.get(p,p[:12])]={
      "compound_mean_rmse":float(np.sqrt(np.mean((compound-y)**2))),
      "global_mean_rmse":float(np.sqrt(np.mean((global_prediction-y)**2))),
      "target_sd":float(y.std()),"floor_fraction":float(np.mean(y==5.0)), "n":54
    }
result={"status":"TRAIN_ONLY_SIMPLE_BASELINES_ON_REUSED_8_PROTEIN_DEV",
        "train_proteins":24,"dev_proteins":8,"shared_compounds":54,
        "global_mean_macro_rmse":float(np.mean([v["global_mean_rmse"] for v in data.values()])),
        "compound_mean_macro_rmse":float(np.mean([v["compound_mean_rmse"] for v in data.values()])),
        "source_rows_sha256":sha_file(OLD/"data/engineering_pilot/rows.json"),
        "per_protein":data,
        "limitations":["DEV previously inspected; known compounds across train and validation",
                       "Simple mean baselines see every compound during training, unlike a cold compound split"]}
dump(OUT/"MATCHED_SIMPLE_BASELINES.json",result)
print(json.dumps({"global_mean_macro":result["global_mean_macro_rmse"],
    "compound_mean_macro":result["compound_mean_macro_rmse"],
    "per_protein":data}),flush=True)
