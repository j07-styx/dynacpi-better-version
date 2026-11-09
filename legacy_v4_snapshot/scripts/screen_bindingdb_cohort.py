"""Preliminary cohort screen with RDKit chemical identity and deduplicated task sizes.

Important: THIS DOES NOT establish assay equivalence, exact endpoint provenance,
construct validity, sequence-family separation, or a final benchmark split.
"""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import statistics
import sys
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
root = Path(__file__).resolve().parents[1]
full = "--full" in sys.argv
stem = "bindingdb_full_kd" if full else "bindingdb_articles_kd"
source = root / "data/audit" / (stem + "_candidates.jsonl")
if not source.exists():
    raise FileNotFoundError(source)
summary_file = root / "data/audit" / (stem + "_task_summary.jsonl")
out_path = root / "data/audit" / (stem + "_rdkit_screen.json")
tasks = defaultdict(lambda: defaultdict(list))
stats = Counter()
for ln in source.open(encoding="utf-8"):
    d = json.loads(ln)
    stats["source_candidate_records"] += 1
    mol = Chem.MolFromSmiles(d["smiles_raw"])
    if mol is None:
        stats["invalid_rdkit_smiles"] += 1
        continue
    compound = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if compound != d["smiles_raw"]:
        stats["canonicalization_changed_string"] += 1
    if not 0 < len(d["sequence"]) < 10000:
        stats["unexpected_sequence_length"] += 1
        continue
    if not 0 < float(d["pKd"]) < 20:
        stats["pKd_outside_initial_plausibility_range"] += 1
        continue
    seq_sha = hashlib.sha256(d["sequence"].encode()).hexdigest()
    task_key = (seq_sha, d["assay_key"])
    tasks[task_key][compound].append((d["pKd"],d["reactant_set_id"]))
    stats["rdkit_valid_records"] += 1

eligible_15 = []
eligible_30 = []
duplicate_compound_groups = 0
conflicting_replicate_groups = 0
with summary_file.open("w",encoding="utf-8") as f:
    for (sha,assay),compounds in sorted(tasks.items()):
        n = len(compounds)
        replicate_records = sum(len(v) for v in compounds.values())
        conflicts = 0
        for values in compounds.values():
            if len(values)>1:
                duplicate_compound_groups+=1
                if max(v[0] for v in values) - min(v[0] for v in values) > 0.1:
                    conflicts+=1
                    conflicting_replicate_groups+=1
        result = {"sequence_sha256":sha,"assay_key":assay,
                  "unique_canonical_compounds":n,"raw_records":replicate_records,
                  "within_group_duplicate_compounds_gt_0_1_pKd":conflicts}
        f.write(json.dumps(result)+"\n")
        if n >= 15:
            eligible_15.append(result)
        if n >= 30:
            eligible_30.append(result)

counts_15 = set(x["sequence_sha256"] for x in eligible_15)
counts_30 = set(x["sequence_sha256"] for x in eligible_30)
report = {
    "scope": "Full November 2026 BindingDB ZIP" if full else "November 2026 Articles subset ZIP",
    "status": "PRELIMINARY RDKit identity screening only; not a verified scientific cohort",
    "records": dict(stats),
    "task_count":len(tasks),
    "groups_ge15":len(eligible_15),"groups_ge30":len(eligible_30),
    "sequence_hashes_ge15":len(counts_15),"sequence_hashes_ge30":len(counts_30),
    "within_task_duplicate_compound_groups":duplicate_compound_groups,
    "within_task_duplicate_compound_groups_disagreement_gt0_1_pKd":conflicting_replicate_groups,
    "top_20_tasks":sorted(eligible_15,key=lambda x:x["unique_canonical_compounds"],reverse=True)[:20],
    "warning":"Assay ID only is insufficient to certify experimental comparability. Protein similarities and actual constructs still need validation; do not build test split from this screening alone."
}
out_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="top_20_tasks"},indent=2),flush=True)
