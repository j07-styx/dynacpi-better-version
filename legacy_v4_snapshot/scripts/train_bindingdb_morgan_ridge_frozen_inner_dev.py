"""Train-only Morgan fingerprint ridge baseline on provisional frozen BindingDB inner dev.

This is a genuinely fitted chemical generalization model, not compound ID lookup.
It does not read reserved test pairs or use dev labels during fitting.
"""
import json,math,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
ROOT=Path(__file__).resolve().parents[1];A=ROOT/"data/audit"
OUT=ROOT/"research/bindingdb_morgan_ridge_frozen_inner_dev_seedless.json"
def read(p):
    with p.open(encoding="utf8") as f:
        for line in f:yield json.loads(line)
def fingerprint_matrix(smiles,gen):
    ri=[];ci=[];invalid=0
    for i,s in enumerate(smiles):
        mol=Chem.MolFromSmiles(s)
        if mol is None:
            invalid+=1;continue
        fp=gen.GetFingerprint(mol)
        ci.extend(fp.GetOnBits());ri.extend([i]*fp.GetNumOnBits())
    return sparse.csr_matrix((np.ones(len(ci),dtype=np.float32),(ri,ci)),shape=(len(smiles),2048),dtype=np.float32),invalid
def main():
    if OUT.exists():raise FileExistsError(OUT)
    manifest=json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    groups={}
    for g in read(A/"bindingdb_full_kd_protein_components.jsonl"):
        for seq in g["members"]:groups[seq]=g["component_id"]
    tasks={(x["sequence_sha256"],x["assay_key"]) for x in read(A/"bindingdb_full_kd_task_summary.jsonl") if x["unique_canonical_compounds"]>=15}
    train=[];valid=[];train_components=set(manifest["train_components"]);dev_components=set(manifest["inner_dev_components"]);test_components=set(manifest["test_components"])
    assert not train_components&dev_components and not train_components&test_components and not dev_components&test_components
    for r in read(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if (r["protein_seq_sha256"],r["assay_key"]) not in tasks or r["range_pKd"]>.1:continue
        component=groups[r["protein_seq_sha256"]]
        if component in train_components:train.append(r)
        elif component in dev_components:valid.append(r)
    assert len(train)==manifest["train_pairs"] and len(valid)==manifest["inner_dev_pairs"]
    t0=time.time()
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    tr,itr=fingerprint_matrix([r["smiles_isomeric"] for r in train],gen)
    va,iva=fingerprint_matrix([r["smiles_isomeric"] for r in valid],gen)
    assert itr==0 and iva==0,(itr,iva)
    y=np.array([r["median_pKd"] for r in train],dtype=np.float64)
    v=np.array([r["median_pKd"] for r in valid],dtype=np.float64)
    mu=float(y.mean());std=float(y.std());assert std>0
    # Ridge on standardized training targets with a free (unpenalized) intercept
    # by appending a constant column with large scaling.
    scale=100.0
    augmented=sparse.hstack([tr,sparse.csr_matrix(np.full((len(train),1),scale,dtype=np.float32))],format="csr")
    # Fixed alpha=100, declared before evaluating the development set.
    alpha=100.0
    solution=lsqr(augmented,(y-mu)/std,damp=math.sqrt(alpha),atol=1e-6,btol=1e-6,iter_lim=300)
    weights=solution[0][:-1];bias=solution[0][-1]*scale
    pred=(va@weights+bias)*std+mu
    sq=(pred-v)**2
    comp_err={}
    for i,r in enumerate(valid):
        component=groups[r["protein_seq_sha256"]]
        comp_err.setdefault(component,[]).append(float(sq[i]))
    result={
      "status":"PROVISIONAL_INNER_DEV_ONLY_NO_TEST",
      "architecture":"Morgan radius2 chirality2048 sparse ridge with free intercept",
      "alpha_fixed_predeclared":alpha,
      "train_pairs":len(train),"inner_dev_pairs":len(valid),
      "train_mean_rmse":float(np.sqrt(np.mean((v-mu)**2))),
      "morgan_ridge_rmse":float(np.sqrt(np.mean(sq))),
      "morgan_ridge_component_macro_rmse":float(np.mean([np.sqrt(np.mean(a)) for a in comp_err.values()])),
      "inner_dev_components":len(comp_err),
      "lsqr_iterations":solution[2],"lsqr_stop_code":solution[1],
      "seconds":round(time.time()-t0,2),
      "warnings":["Provisional assay/construct QC, no scientific cohort approval.",
                  "Chemical-only baseline; held-out proteins are not represented explicitly.",
                  "No reserved test pairs evaluated or fitted; development set has been inspected repeatedly."]
    }
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()

