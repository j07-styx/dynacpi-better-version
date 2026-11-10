"""Label-blind chemistry novelty audit of the PREEXISTING frozen DAVIS splits.
No affinity labels accessed, no model selection, no split modifications.
"""
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from train_full_davis_v5 import tanimoto,sha,PREP
from prepare_preregistered_davis_generalization_20261110 import OUT
REPORT=OUT/"CHEM_NOVELTY_DIAGNOSTIC.json"

def main():
    if REPORT.exists():raise RuntimeError("Refusing overwrite")
    summary=json.loads((OUT/"SUITE_LOCKED.json").read_text(encoding="utf-8"))
    with np.load(PREP,allow_pickle=False) as z:
        fingerprints=z["bits"].astype(np.float64)
        smiles=z["smiles"].astype(str)
    assert fingerprints.shape==(68,2048)
    scaffold=[
       MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(str(s)))
       for s in smiles]
    record={"status":"LABEL_BLIND_NO_TEST_METRICS","suite_source_sha256":sha(OUT/"SUITE_LOCKED.json"),
            "n_ligands":68,"n_exact_murcko_scaffolds":len(set(scaffold)),"scenarios":[]}
    for scenario in summary["scenarios"]:
        name=scenario["scenario"]
        spec_path=OUT/name/"SPLIT_LOCKED.json"
        spec=json.loads(spec_path.read_text(encoding="utf-8"))
        if sha(spec_path)!=scenario["split_manifest_sha256"]:
            raise RuntimeError("Frozen scenario changed")
        tr=np.array(spec["indices"]["compound_train"],dtype=int)
        va=np.array(spec["indices"]["compound_val"],dtype=int)
        te=np.array(spec["indices"]["compound_test"],dtype=int)
        all_tr_scaff={scaffold[i] for i in tr}
        report={"scenario":name,"category":spec["category"],"n_training_compounds":len(tr),
                "n_validation_compounds":len(va),"n_test_compounds":len(te),
                "test_labels_read":False}
        if spec["category"] in ("compound","double"):
            assert not set(tr)&set(va) and not set(tr)&set(te)
            assert not all_tr_scaff & {scaffold[i] for i in va}
            assert not all_tr_scaff & {scaffold[i] for i in te}
            for label,ix in (("validation",va),("test",te)):
                mat=tanimoto(fingerprints[ix],fingerprints[tr])
                max_sim=mat.max(axis=1)
                report[label+"_nearest_training_morgan_tanimoto"]={
                    "min":float(np.min(max_sim)),"median":float(np.median(max_sim)),
                    "mean":float(np.mean(max_sim)),"max":float(np.max(max_sim)),
                    "fraction_nearest_ge0_5":float(np.mean(max_sim>=.5)),
                    "fraction_nearest_ge0_7":float(np.mean(max_sim>=.7)),
                    "n_ligands":int(len(ix))}
        record["scenarios"].append(report)
    REPORT.write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record,indent=2),flush=True)
if __name__=="__main__":main()
