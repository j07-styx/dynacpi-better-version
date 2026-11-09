"""Label-blind chemical leakage audit of the reconciled DAVIS DEV split.

Computes Bemis-Murcko scaffold overlap and nearest train Tanimoto for
validation compounds. This is an audit, never a new split or benchmark.
"""
import json, hashlib
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
ROOT=Path(__file__).resolve().parents[1]
ROWS=ROOT/"data/engineering_pilot_primary_reconciled_20261107/rows.json"
COMPOUNDS=ROOT/"data/staging/compounds.json"
OUT=ROOT/"research/reconciled_chemical_overlap_audit_dev.json"

def audit(rows, smiles):
    train={r["compound"] for r in rows if r["split"]=="train"}
    val={r["compound"] for r in rows if r["split"]=="val"}
    if not train or not val:raise ValueError("Empty split")
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    compounds={}
    for cid in train|val:
        mol=Chem.MolFromSmiles(smiles[cid])
        if mol is None:raise ValueError("Invalid source compound")
        canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
        if hashlib.sha256(canonical.encode()).hexdigest()!=cid:
            raise ValueError("Compound hash identity mismatch")
        compounds[cid]=(gen.GetFingerprint(mol),MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    trfp=[compounds[c][0] for c in sorted(train)]
    trsc={compounds[c][1] for c in train}
    nearest={}
    for cid in sorted(val):
        nearest[cid]=max(DataStructs.BulkTanimotoSimilarity(compounds[cid][0],trfp))
    sim=np.array(list(nearest.values()))
    return {
      "status":"CHEMICAL_OVERLAP_AUDIT_DEV_NOT_COLD_COMPOUND_BENCHMARK",
      "train_unique_compounds":len(train),
      "validation_unique_compounds":len(val),
      "exact_compounds_shared":len(train&val),
      "validation_compounds_with_scaffold_in_train":sum(compounds[c][1] in trsc for c in val),
      "validation_compounds_with_nearest_train_tanimoto_at_least_0_8":int((sim>=.8).sum()),
      "validation_compounds_with_nearest_train_tanimoto_at_least_0_9":int((sim>=.9).sum()),
      "nearest_train_tanimoto_summary":{"min":float(sim.min()),"median":float(np.median(sim)),"max":float(sim.max())},
      "note":"All calculations use split membership and canonical structures only, never affinities. No independent test set created."
    }

def main():
    rows=json.loads(ROWS.read_text(encoding="utf-8"))
    smiles=json.loads(COMPOUNDS.read_text(encoding="utf-8"))
    result=audit(rows,smiles)
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()


