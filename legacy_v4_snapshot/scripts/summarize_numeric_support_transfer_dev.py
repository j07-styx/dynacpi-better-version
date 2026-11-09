"""Target-block bootstrap; repeated episodes are not independent proteins."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"research/numeric_support_transfer_20261107"
rows=json.loads((out/"results.json").read_text())
rng=np.random.default_rng(71108);summary=[]
for r in rows:
    for a,b in (("pooled_chemical_kernel","compound_mean"),("support_residual_kernel","pooled_chemical_kernel"),("support_residual_kernel","shuffled_support_kernel")):
        differences=np.array([t["mean_rmse"][a]-t["mean_rmse"][b] for t in r["per_target"]])
        boot=differences[rng.integers(0,len(differences),size=(10000,len(differences)))].mean(1)
        summary.append({"k":r["k"],"comparison":a+" minus "+b,"mean_delta_rmse":float(differences.mean()),
                        "target_bootstrap_95_percentile_interval":np.quantile(boot,[.025,.975]).tolist(),
                        "independent_blocks":len(differences)})
path=out/"uncertainty.json"
with path.open("x") as f:json.dump({"scope":"Descriptive uncertainty across seven target identifiers, not independent protein families; no multiplicity correction","comparisons":summary},f,indent=2)
print(json.dumps(summary,indent=2))
