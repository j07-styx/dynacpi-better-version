"""Export the frozen DynaCPI v5 train+val DAVIS split for ORIGINAL PerceiverCPI.

This DOES NOT read test affinity labels or export the held-out test manifest.
The upstream CLI requires a separate_test_path; val_proxy_test.csv is an exact
COPY of validation and MUST NOT be reported as held-out performance.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from prepare_full_davis_v5 import DATA, SPLIT, SRC, Y_SHA256, sha

OUT=Path(__file__).resolve().parent/"matched_perceivercpi_20261110"/"data_v2"
def write_csv(dest,rows):
    with dest.open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["smiles","sequence","label"])  # upstream get_data expects singular (README is outdated)
        writer.writerows(rows)

def main():
    manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
    assert sha(DATA)==manifest["prepared_features_sha256"]
    assert sha(SRC/"Y")==manifest["raw_Y_sha256"]==Y_SHA256
    with np.load(DATA,allow_pickle=False) as z:
        train=z["train"].astype(int); val=z["val"].astype(int)
        smiles=z["smiles"].astype(str)
        y=z["y"]
    original_proteins=list(json.loads((SRC/"proteins.txt").read_text(encoding="utf-8")).values())
    assert len(original_proteins)==442 and len(smiles)==68 and y.shape==(442,68)
    assert len(train)==302 and len(val)==59
    assert not set(train)&set(val)
    rows_train=[(str(smiles[j]),original_proteins[int(i)],repr(float(y[i,j])))
                for i in train for j in range(68)]
    rows_val=[(str(smiles[j]),original_proteins[int(i)],repr(float(y[i,j])))
              for i in val for j in range(68)]
    assert (len(rows_train),len(rows_val))==(20536,4012)
    if OUT.exists():raise RuntimeError("Refusing to replace existing split export")
    OUT.mkdir(parents=True)
    write_csv(OUT/"train.csv",rows_train)
    write_csv(OUT/"val.csv",rows_val)
    write_csv(OUT/"val_proxy_test.csv",rows_val)
    # Smoke uses only rows from the prescribed training/validation sets.
    write_csv(OUT/"smoke_train.csv",rows_train[:4*68])
    write_csv(OUT/"smoke_val.csv",rows_val[:2*68])
    write_csv(OUT/"smoke_val_proxy_test.csv",rows_val[:2*68])
    checks={
      "dataset":"original public DeepDTA DAVIS pKd=9-log10(Kd[nM])",
      "source_Y_sha256":Y_SHA256,
      "prepared_features_sha256":sha(DATA),
      "split_manifest_sha256":sha(SPLIT),
      "exact_frozen_v5_protein_indices":True,
      "n_training_proteins":302,"n_validation_proteins":59,
      "n_training_pairs":len(rows_train),"n_validation_pairs":len(rows_val),
      "n_compounds":68,
      "test_labels_not_exported":True,
      "upstream_required_test_path_is_validation_clone":True,
      "official_PerceiverCPI_repo":"https://github.com/dmis-lab/PerceiverCPI",
      "csv_sha256":{name:sha(OUT/name) for name in [
         "train.csv","val.csv","val_proxy_test.csv","smoke_train.csv","smoke_val.csv","smoke_val_proxy_test.csv"]}
    }
    (OUT/"SPLIT_EXPORT.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2),flush=True)

if __name__=="__main__":
    main()
