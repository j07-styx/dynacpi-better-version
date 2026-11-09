"""Fixed numeric training-only protein contribution experiment."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from experiment_numeric_support_transfer_dev import kernel_predict
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research/protein_chemical_kernel_20261107"
def protein_kernel(x,train_ids):
    distances=np.maximum(0,((x[:,None,:]-x[None,:,:])**2).sum(2))
    block=distances[np.ix_(train_ids,train_ids)]
    bandwidth=float(np.median(block[block>0]))
    assert bandwidth>0
    return np.exp(-distances/(2*bandwidth))
def main():
    OUT.mkdir(exist_ok=False)
    path=ROOT/"data/engineering_pilot_primary_reconciled_20261107/rows.json"
    rows=[r for r in json.loads(path.read_text()) if r["split"]=="train" and r["primary_label_kind"]=="numeric_Kd"]
    assert len(rows)==106
    targets=sorted({r["sequence"] for r in rows});compounds=sorted({r["compound"] for r in rows})
    protocol={"scope":"Training-only LOPO diagnostic; no validation scoring or architecture selection",
       "rows":106,"targets":9,"alpha":1.0,"protein_kernel":"RBF of mean ESM embeddings; bandwidth median squared distance among training target identifiers",
       "chemical_kernel":"Morgan radius2 2048bit chiral Tanimoto",
       "models":["chemical","protein","product","half_chemical_half_product"],
       "negative_control":"20 deterministic derangements of protein embedding assignments, same mapping within each run",
       "seed":71109,"rows_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"hyperparameter_search":False}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
    x=[]
    for target in targets:
        obj=torch.load(ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(target+".pt"),map_location="cpu",weights_only=True)
        assert obj["sequence_hash"]==target
        x.append(obj["embeddings"].float().mean(0).numpy().astype(float))
    x=np.stack(x);assert np.isfinite(x).all()
    smiles=json.loads((ROOT/"data/staging/compounds.json").read_text())
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fps=[gen.GetFingerprint(Chem.MolFromSmiles(smiles[c])) for c in compounds]
    chemical=np.array([DataStructs.BulkTanimotoSimilarity(fp,fps) for fp in fps])
    ci=np.array([compounds.index(r["compound"]) for r in rows]);pi=np.array([targets.index(r["sequence"]) for r in rows])
    y=np.array([r["numeric_pKd"] for r in rows]);kc=chemical[np.ix_(ci,ci)]
    rng=np.random.default_rng(71109);permutations=[]
    while len(permutations)<20:
        p=rng.permutation(9)
        if np.all(p!=np.arange(9)) and p.tolist() not in permutations:permutations.append(p.tolist())
    folds=[]
    for hold in range(9):
        train=np.flatnonzero(pi!=hold);test=np.flatnonzero(pi==hold)
        assert set(pi[train]).isdisjoint(set(pi[test]))
        train_targets=[i for i in range(9) if i!=hold]
        kp=protein_kernel(x,train_targets)[np.ix_(pi,pi)]
        matrices={"chemical":kc,"protein":kp,"product":kc*kp,"half_chemical_half_product":kc*(.5+.5*kp)}
        scores={}
        for name,k in matrices.items():
            pred=kernel_predict(k[np.ix_(train,train)],y[train],k[np.ix_(test,train)])
            scores[name]=float(np.sqrt(np.mean((pred-y[test])**2)))
        shuffled=[]
        for p in permutations:
            fake=protein_kernel(x[p],train_targets)[np.ix_(pi,pi)]
            k=kc*(.5+.5*fake)
            pred=kernel_predict(k[np.ix_(train,train)],y[train],k[np.ix_(test,train)])
            shuffled.append(float(np.sqrt(np.mean((pred-y[test])**2))))
        folds.append({"target":rows[test[0]]["primary_protein"],"query_count":len(test),"rmse":scores,"shuffled_hybrid_rmse":shuffled})
    means={m:float(np.mean([f["rmse"][m] for f in folds])) for m in matrices}
    means["shuffled_hybrid_mean"]=float(np.mean([f["shuffled_hybrid_rmse"] for f in folds]))
    differences=np.array([f["rmse"]["half_chemical_half_product"]-f["rmse"]["chemical"] for f in folds])
    boot=differences[rng.integers(0,9,size=(10000,9))].mean(1)
    result={"macro_rmse":means,"folds":folds,
      "hybrid_minus_chemical_target_bootstrap_95_interval":np.quantile(boot,[.025,.975]).tolist(),
      "hybrid_beats_chemical_targets":int(sum(differences<0)),
      "limits":["Nine target identifiers, not independent protein families","Numeric-only ascertainment bias",
                "Repeatedly used training-development pool","Not comparable to prior support experiment's seven-target query subsets"]}
    (OUT/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="folds"},indent=2),flush=True)
if __name__=="__main__":main()
