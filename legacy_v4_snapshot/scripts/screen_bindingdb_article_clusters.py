"""RDKit identity and preliminary protein-cluster screen for the Articles subset.
Not a final split; same-construct/assay and sequence-only homology checks remain.
"""
from pathlib import Path
from collections import defaultdict, Counter
from Bio.Align import PairwiseAligner
from rdkit import Chem, RDLogger
import hashlib
import json

RDLogger.DisableLog("rdApp.*")
root = Path(__file__).resolve().parents[1]
src = root / "data/audit/bindingdb_articles_kd_candidates.jsonl"
groups = defaultdict(set)
rows = 0
invalid = 0
identity_changes = 0
with src.open(encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        rows += 1
        mol = Chem.MolFromSmiles(d["smiles_raw"])
        if mol is None:
            invalid += 1
            continue
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if smi != d["smiles_raw"]:
            identity_changes += 1
        groups[(d["sequence"], d["assay_key"])].add(smi)

counts = {15: [], 30: []}
for cutoff in (15, 30):
    counts[cutoff] = [(seq, assay, len(cs)) for (seq, assay), cs in groups.items() if len(cs) >= cutoff]
 
def connected_components(rows):
    seqs = sorted(set(seq for seq, _, _ in rows))
    n = len(seqs)
    parent = list(range(n))
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.
    aligner.mismatch_score = -1.
    aligner.open_gap_score = -10.
    aligner.extend_gap_score = -0.5
    def rootof(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = seqs[i], seqs[j]
            hit = aligner.align(a,b)[0]
            matched = 0
            paired = 0
            covered_a = 0
            covered_b = 0
            for (starta, enda), (startb, endb) in zip(hit.aligned[0],hit.aligned[1]):
                span = enda-starta
                if span != endb-startb:
                    raise ValueError("Aligned block mismatch")
                paired += span
                matched += sum(x==y for x,y in zip(a[starta:enda], b[startb:endb]))
                covered_a += span
                covered_b += span
            if paired and matched/paired >= .30 and min(covered_a,covered_b)/min(len(a),len(b)) >= .80:
                pi, pj = rootof(i), rootof(j)
                if pi != pj:
                    parent[pi] = pj
                edges += 1
    comp = defaultdict(list)
    for i, seq in enumerate(seqs):
        comp[rootof(i)].append(seq)
    sizes = sorted((len(x) for x in comp.values()), reverse=True)
    return {"eligible_unique_sequences":n, "connected_components":len(comp),
            "similarity_edges":edges, "component_sizes":sizes,
            "alignment_policy": "Biopython local; match=2 mismatch=-1 open=-10 extend=-0.5; >=30% identity over aligned positions; aligned coverage >=80% shorter. PRELIMINARY only, domain homology unverified."}

result = {
    "scope":"BindingDB Articles subset; not the full 202611 export",
    "source_rows": rows, "rdkit_invalid_smiles_rows": invalid,
    "canonicalization_changes":identity_changes,
    "canonical_policy":"RDKit canonical isomeric SMILES without parent/salt normalization; preliminary stereochemistry-preserving string identity",
    "candidate_groups_total":len(groups),
    "candidate_groups_ge15":len(counts[15]),
    "candidate_groups_ge30":len(counts[30]),
    "protein_screen_ge15":connected_components(counts[15]),
    "protein_screen_ge30":connected_components(counts[30]),
    "warnings":[
        "No authoritative target construct identity, assay condition equivalence or exact Kd semantic review yet",
        "Protein clustering only on preliminary eligible subset can undercount edges through excluded sequences",
        "No chemical scaffold / assay / train-validation-test manifests; cannot claim split feasibility",
    ],
}
dest=root/"data/audit/bindingdb_articles_kd_cluster_screen.json"
dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
