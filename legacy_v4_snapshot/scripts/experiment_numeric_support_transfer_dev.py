"""Fixed training-only leave-one-protein-out support-transfer experiment."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research/numeric_support_transfer_20261107"
def kernel_predict(ktrain,y,kquery,alpha=1.0):
    mean=float(np.mean(y))
    return mean+kquery@np.linalg.solve(ktrain+alpha*np.eye(len(y)),y-mean)
def main():
    OUT.mkdir(exist_ok=False)
    path=ROOT/"data/engineering_pilot_primary_reconciled_20261107/rows.json"
    # Validation values are never selected or passed to model/scoring.
    rows=[r for r in json.loads(path.read_text()) if r["split"]=="train" and r["primary_label_kind"]=="numeric_Kd"]
    assert len(rows)==106
    targets=sorted({r["sequence"] for r in rows})
    protocol={"scope":"Training-only numeric-label LOPO engineering experiment, not independent benchmark",
      "training_rows":len(rows),"training_targets":len(targets),"support_sizes":[1,3,5],
      "draws":100,"seed":71107,"kernel":"chiral Morgan radius 2, 2048-bit Tanimoto",
      "ridge_alpha":1.0,"offset_shrinkage":1.0,"selection":"All held-out targets with >=8 numeric records",
      "rows_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
      "validation_scored":False,"hyperparameter_search":False,
      "limits":["Small reused development training pool","Numeric-only ascertainment bias",
                "Targets are identifier-disjoint, not homology-disjoint","Not evidence of superiority over neural baseline"]}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
    compounds=json.loads((ROOT/"data/staging/compounds.json").read_text())
    ids=sorted({r["compound"] for r in rows});index={c:i for i,c in enumerate(ids)}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fps=[gen.GetFingerprint(Chem.MolFromSmiles(compounds[c])) for c in ids]
    K=np.array([DataStructs.BulkTanimotoSimilarity(fp,fps) for fp in fps])
    assert np.allclose(K,K.T) and np.allclose(np.diag(K),1)
    result=[];perdraw=[]
    for target in targets:
        held=[r for r in rows if r["sequence"]==target]
        if len(held)<8:continue
        train=[r for r in rows if r["sequence"]!=target]
        assert not any(r["sequence"]==target for r in train)
        ti=np.array([index[r["compound"]] for r in train]);hi=np.array([index[r["compound"]] for r in held])
        y=np.array([r["numeric_pKd"] for r in train]);hy=np.array([r["numeric_pKd"] for r in held])
        pred=kernel_predict(K[np.ix_(ti,ti)],y,K[np.ix_(hi,ti)])
        cm=defaultdict(list)
        for r in train:cm[r["compound"]].append(r["numeric_pKd"])
        compound_mean=np.array([np.mean(cm[r["compound"]]) if cm[r["compound"]] else np.mean(y) for r in held])
        rng=np.random.default_rng(71107+int(target[:8],16))
        for draw in range(100):
            order=rng.permutation(len(held))
            for n in (1,3,5):
                support=order[:n];query=order[n:]
                assert set(support).isdisjoint(query)
                residual=hy[support]-pred[support]
                ks=K[np.ix_(hi[support],hi[support])];kq=K[np.ix_(hi[query],hi[support])]
                correction=kq@np.linalg.solve(ks+np.eye(n),residual)
                # Permute measured labels across the same support compounds, retaining their distribution.
                fake=hy[support][rng.permutation(n)]-pred[support]
                shuffled=kq@np.linalg.solve(ks+np.eye(n),fake)
                guesses={"compound_mean":compound_mean[query],"pooled_chemical_kernel":pred[query],
                  "support_mean":np.full(len(query),hy[support].mean()),
                  "support_offset":pred[query]+residual.sum()/(n+1),
                  "support_residual_kernel":pred[query]+correction,
                  "shuffled_support_kernel":pred[query]+shuffled}
                errors={name:float(np.sqrt(np.mean((v-hy[query])**2))) for name,v in guesses.items()}
                perdraw.append({"target":held[0]["primary_protein"],"k":n,"draw":draw,"query_count":len(query),"rmse":errors})
    for n in (1,3,5):
        subset=[r for r in perdraw if r["k"]==n];names=list(subset[0]["rmse"])
        pertarget=[]
        for target in sorted({r["target"] for r in subset}):
            selected=[r for r in subset if r["target"]==target]
            pertarget.append({"target":target,"mean_rmse":{m:float(np.mean([r["rmse"][m] for r in selected])) for m in names}})
        means={m:float(np.mean([r["mean_rmse"][m] for r in pertarget])) for m in names}
        result.append({"k":n,"targets":len(pertarget),"episodes":len(subset),"macro_mean_episode_rmse":means,
                       "per_target":pertarget,
                       "residual_beats_pooled_targets":sum(r["mean_rmse"]["support_residual_kernel"]<r["mean_rmse"]["pooled_chemical_kernel"] for r in pertarget),
                       "residual_beats_shuffled_targets":sum(r["mean_rmse"]["support_residual_kernel"]<r["mean_rmse"]["shuffled_support_kernel"] for r in pertarget)})
    (OUT/"results.json").write_text(json.dumps(result,indent=2))
    (OUT/"episode_metrics_LOCAL.json").write_text(json.dumps(perdraw))
    print(json.dumps([{k:v for k,v in r.items() if k!="per_target"} for r in result],indent=2),flush=True)
if __name__=="__main__":main()
