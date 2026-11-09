"""Stage sequence-only inputs for preliminary K=5 candidate task proteins.

No affinity labels are used. This DOES NOT select the final scientific cohort.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
tasks={}
for line in (audit/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl").open(encoding="utf-8"):
    x=json.loads(line)
    if x["n_unique_compounds"]>=15:
        tasks[x["protein_seq_sha256"]]=tasks.get(x["protein_seq_sha256"],0)+1
sequences={}
for line in (audit/"bindingdb_full_kd_candidates.jsonl").open(encoding="utf-8"):
    x=json.loads(line)
    s=x["sequence"]
    h=hashlib.sha256(s.encode()).hexdigest()
    if h in tasks:
        if h in sequences and sequences[h]!=s:raise ValueError("Sequence hash collision")
        sequences[h]=s
if set(sequences)!=set(tasks):
    raise ValueError("Eligible task sequences missing from source")
outdir=root/"data/bindingdb_kd15_preliminary"
outdir.mkdir(parents=True,exist_ok=True)
(outdir/"sequences.json").write_text(json.dumps(sequences),encoding="utf-8")
with (outdir/"sequences.fasta").open("w",encoding="utf-8") as f:
    for h,s in sorted(sequences.items()):
        f.write(f">{h}\n{s}\n")
legacy=json.loads((root/"data/staging/sequences.json").read_text(encoding="utf-8"))
lengths=list(map(len,sequences.values()))
result={
    "status":"PRELIMINARY_SEQUENCE_CACHE_INPUT_ONLY",
    "source_tasks_ge15":sum(tasks.values()),
    "unique_protein_sequences":len(sequences),
    "already_in_legacy_esm_sequence_set":len(set(sequences)&set(legacy)),
    "total_amino_acids":sum(lengths),
    "min_sequence_length":min(lengths),
    "max_sequence_length":max(lengths),
    "note":"Source candidates have not passed all assay, protein-construct and protein-cluster QC; caching sequence features is not final cohort selection or model training."}
(outdir/"manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
