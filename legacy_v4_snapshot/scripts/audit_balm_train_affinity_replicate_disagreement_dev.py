"""BALM frozen TRAIN-only within-exact-input affinity disagreement audit.

No development/test labels. Analyze exact protein and canonical isomeric SMILES
with no assumptions that repeated labels have same assay conditions.
"""
import hashlib,json,math
from collections import defaultdict,Counter

import numpy as np
from rdkit import Chem
from dynacpi.balm_public_data_dev import ROOT,get_train_only

OUT=ROOT/"research/balm_train_only_affinity_replicate_disagreement_audit_20261108.json"
THRESHOLDS=(.1,.3,.5,1.,2.,3.)

def summary(values):
    x=np.array(list(values),dtype=np.float64)
    return {
        "count":len(x),"min":float(np.min(x)),"median":float(np.median(x)),
        "mean":float(np.mean(x)),"p90":float(np.quantile(x,.9)),"max":float(np.max(x))
    }

def main():
    if OUT.exists():raise FileExistsError(OUT)
    rows=get_train_only()
    assert len(rows)==15538
    smi={}
    for raw in {r["smiles"] for r in rows}:
        mol=Chem.MolFromSmiles(raw)
        assert mol is not None
        smi[raw]=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    grouped=defaultdict(list)
    for r in rows:
        grouped[(hashlib.sha256(r["protein"].encode()).hexdigest(),smi[r["smiles"]])].append(r)
    repeated={key:parts for key,parts in grouped.items() if len(parts)>=2}
    label_ranges={key:float(np.ptp([r["label"] for r in parts])) for key,parts in repeated.items()}
    # Delta pKd here is a logarithmic-unit difference, not a direct Kd ratio.
    conflict={}
    for threshold in THRESHOLDS:
        bad={key for key,spread in label_ranges.items() if spread>=threshold}
        conflict[str(threshold)]={
            "exact_input_groups":len(bad),
            "original_rows_in_groups":sum(len(repeated[key]) for key in bad),
            "share_of_15538_original_rows":sum(len(repeated[key]) for key in bad)/len(rows),
            "share_of_repeated_groups":len(bad)/len(repeated) if repeated else 0.,
        }
    repeated_label_pairs=[p for p in repeated.values()]
    within_labels=np.asarray([r["label"] for r in rows],dtype=np.float64)
    inconsistent_counts=Counter()
    for key,records in repeated.items():
        spread=label_ranges[key]
        if spread>=1.:
            inconsistent_counts[len(records)]+=1
    report={
        "status":"BALM_FROZEN_TRAIN_ONLY_DUPLICATE_AFFINITY_DISAGREEMENT_AUDIT_NO_DEV_OR_TEST",
        "raw_training_rows":len(rows),
        "unique_exact_protein_canonical_isomeric_smiles":len(grouped),
        "repeated_exact_input_groups":len(repeated),
        "rows_in_repeated_groups":sum(len(v) for v in repeated.values()),
        "maximum_repeat_count":max(len(v) for v in grouped.values()),
        "unique_smiles":len(smi),
        "label_distribution":summary(within_labels),
        "repeat_count_distribution":summary([len(v) for v in repeated.values()]),
        "label_range_across_repeat_group":summary(label_ranges.values()) if repeated else None,
        "conflict_thresholds_log_affinity_units":conflict,
        "repeat_groups_range_ge_1_by_repeat_count":dict(sorted(inconsistent_counts.items())),
        "caveats":[
            "Repeated input does not guarantee the same assay method, construct, buffer or measurement provenance.",
            "BALM source lacks per-record assay DOI and conditions, so disagreements cannot automatically be called experimental errors.",
            "Input canonicalization uses RDKit isomeric SMILES; stereoisomers remain distinct.",
            "Only frozen TRAIN labels accessed, not original inner development or prospective reserved test.",
            "No labels excluded or relabeled by this audit."
        ],
    }
    OUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":main()

