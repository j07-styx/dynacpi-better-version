"""Generate exact sequence FASTA for preliminary BindingDB full-Kd clustering.

Output does not certify constructs or define a train/test split.
"""
from pathlib import Path
import json, hashlib

root=Path(__file__).resolve().parents[1]
source=root/"data/audit/bindingdb_full_kd_candidates.jsonl"
dest=root/"data/audit/bindingdb_full_kd_sequences.fasta"
seqs={}
for line in source.open(encoding="utf-8"):
    d=json.loads(line)
    seq=d["sequence"]
    h=hashlib.sha256(seq.encode()).hexdigest()
    if h in seqs and seqs[h]!=seq:
        raise ValueError("SHA256 collision or inconsistent sequence")
    seqs[h]=seq
with dest.open("w",encoding="utf-8") as f:
    for h,s in sorted(seqs.items()):
        f.write(f">{h}\n{s}\n")
print(json.dumps({"fasta":str(dest),"unique_sequences":len(seqs),
                  "max_residues":max(map(len,seqs.values())),
                  "min_residues":min(map(len,seqs.values()))}),flush=True)
