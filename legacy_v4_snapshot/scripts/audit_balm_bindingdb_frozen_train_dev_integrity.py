"""Training and inner-development data integrity for public BALM BindingDB.

No reserved test Y values are used or inspected.
"""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from rdkit import Chem,RDLogger
from dynacpi.balm_public_data_dev import get_train_dev,ROOT
RDLogger.DisableLog("rdApp.*")
OUT=ROOT/"research/balm_bindingdb_frozen_train_dev_integrity_audit.json"

def main():
    parts=get_train_dev()
    data={}
    fingerprint_compounds={}
    seen_by_partition={}
    for name,rows in parts.items():
        y=np.asarray([r["label"] for r in rows],dtype=np.float64)
        assert np.isfinite(y).all()
        components=Counter(r["group"] for r in rows)
        unique_proteins={r["protein"] for r in rows}
        unique_smiles={r["smiles"] for r in rows}
        for sm in unique_smiles:
            if sm not in fingerprint_compounds:
                mol=Chem.MolFromSmiles(sm)
                fingerprint_compounds[sm]=Chem.MolToSmiles(mol,isomericSmiles=True) if mol is not None else None
        invalid=sum(fingerprint_compounds[sm] is None for sm in unique_smiles)
        canonical_compounds={fingerprint_compounds[sm] for sm in unique_smiles if fingerprint_compounds[sm] is not None}
        perinput=defaultdict(list)
        for row in rows:
            canonical=fingerprint_compounds[row["smiles"]]
            perinput[row["protein"],canonical].append(row["label"])
        repeats=[v for v in perinput.values() if len(v)>1]
        contradictory=[v for v in repeats if max(v)-min(v)>.5]
        floor_sq=sum(float(((np.asarray(v)-np.mean(v))**2).sum()) for v in perinput.values())
        report={
            "rows":len(rows),
            "unique_protein_sequences":len(unique_proteins),
            "unique_target_ids":len({r["target_id"] for r in rows}),
            "unique_protein_components":len(components),
            "largest_component_pair_share":max(components.values())/len(rows),
            "unique_raw_smiles":len(unique_smiles),
            "unique_canonical_compounds":len(canonical_compounds),
            "invalid_smiles":invalid,
            "unique_sequence_compound_inputs":len(perinput),
            "repeated_identical_input_groups":len(repeats),
            "repeated_identical_input_rows":sum(len(v) for v in repeats),
            "identical_input_groups_spread_gt_0_5":len(contradictory),
            "identical_input_groups_spread_gt_1_0":sum(max(v)-min(v)>1 for v in repeats),
            "optimistic_within_input_oracle_rmse_floor":float(np.sqrt(floor_sq/len(rows))),
            "label_mean":float(y.mean()),
            "label_std":float(y.std()),
            "label_min":float(y.min()),
            "label_max":float(y.max()),
            "label_q05_median_q95":np.quantile(y,[.05,.5,.95]).tolist()
        }
        data[name]=report
        seen_by_partition[name]={"seq":unique_proteins,"smiles":canonical_compounds}
    tr=parts["train"];dv=parts["inner_dev"]
    mu=np.mean([r["label"] for r in tr]);bygroup=defaultdict(list)
    for r in dv:bygroup[r["group"]].append(r["label"])
    v=np.asarray([r["label"] for r in dv])
    data["integrity"]={
       "train_dev_exact_protein_sequences_shared":len(seen_by_partition["train"]["seq"]&seen_by_partition["inner_dev"]["seq"]),
       "train_dev_exact_canonical_compounds_shared":len(seen_by_partition["train"]["smiles"]&seen_by_partition["inner_dev"]["smiles"]),
       "train_mean_inner_dev_pooled_rmse":float(np.sqrt(np.mean((v-mu)**2))),
       "train_mean_inner_dev_macro_group_rmse":float(np.mean([
          np.sqrt(np.mean((np.asarray(yy)-mu)**2)) for yy in bygroup.values()])),
       "note":"Source has no source-DOI or assay protocol columns, so assay harmonization is not verified. Frozen reserved test labels never accessed."
    }
    report={"status":"TRAIN_AND_INNER_DEV_QC_NO_TEST_EVALUATION",
            "dataset":"BALM public BindingDB_filtered, CC-BY-4.0",
            "train":data["train"],"inner_dev":data["inner_dev"],
            "integrity":data["integrity"]}
    OUT.write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
