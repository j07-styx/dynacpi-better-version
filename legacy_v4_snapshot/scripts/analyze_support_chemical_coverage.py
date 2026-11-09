"""Assess whether identity-selected support compounds are chemically relevant.

No affinity labels are accessed. Estimates query-to-support similarity for the
quarantined candidate cohorts; conclusions are descriptive, not predictive.
"""
from collections import defaultdict,Counter
from pathlib import Path
import json
import numpy as np
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
episodes={r["task_id"]:r for r in
    (json.loads(x) for x in (audit/"bindingdb_full_kd_candidate_episodes_PRE_SPLIT.jsonl").open(encoding="utf-8"))}
compounds=defaultdict(dict)
for line in (audit/"bindingdb_full_kd_compound_median_candidates.jsonl").open(encoding="utf-8"):
    r=json.loads(line)
    if r["task_id"] in episodes:
        compounds[r["task_id"]][r["compound_sha256"]]=r["smiles_isomeric"]

generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
feature_cache={}
def features(smi):
    if smi in feature_cache:
        return feature_cache[smi]
    mol=Chem.MolFromSmiles(smi)
    if mol is None:raise ValueError("Invalid previously accepted compound")
    fp=generator.GetFingerprint(mol)
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=True)
    feature_cache[smi]=(fp,scaffold)
    return fp,scaffold

results={}
for k,cohort in ((5,"all"),(5,"ge30"),(20,"ge30")):
    per_query=[]
    weight=[]
    scaffold_match=0
    n=0
    tasks_used=set()
    for tid,episode in episodes.items():
        if cohort=="ge30" and len(episode["candidate_support_ids"])<20:
            continue
        cr=compounds[tid]
        for draw in episode["support_draws"]:
            key=str(k)
            if key not in draw["support_prefixes"]:
                continue
            support=[features(cr[c]) for c in draw["support_prefixes"][key]]
            sfp=[x[0] for x in support]
            scaffolds={x[1] for x in support if x[1]}
            tasks_used.add(tid)
            for cid in episode["query_ids"]:
                qfp,qscaf=features(cr[cid])
                sims=DataStructs.BulkTanimotoSimilarity(qfp,sfp)
                per_query.append(max(sims))
                weight.append(sum(v**4 for v in sims))
                scaffold_match+=bool(qscaf and qscaf in scaffolds)
                n+=1
    maximum=np.array(per_query,dtype=np.float32)
    sums=np.array(weight,dtype=np.float32)
    results["K"+str(k)+"_"+cohort]={
        "candidate_tasks":len(tasks_used),"query_draw_pairs":n,
        "unique_query_per_task":10,"draws_per_task":10,
        "median_nearest_support_tanimoto":float(np.median(maximum)),
        "mean_nearest_support_tanimoto":float(maximum.mean()),
        "fraction_nearest_support_similarity_ge_0_3":float((maximum>=0.3).mean()),
        "fraction_nearest_support_similarity_ge_0_5":float((maximum>=0.5).mean()),
        "fraction_nearest_support_similarity_ge_0_7":float((maximum>=0.7).mean()),
        "fraction_same_nonempty_murcko_scaffold":scaffold_match/n,
        "median_kernel_weight_sum":float(np.median(sums)),
        "fraction_kernel_weight_sum_lt_0_01":float((sums<0.01).mean()),
        "fraction_kernel_weight_sum_lt_0_001":float((sums<0.001).mean()),
    }
report={
    "scope":"PRELIMINARY BindingDB exact-looking Kd candidate episode chemical coverage",
    "status":"DATA-ONLY ANALYSIS, no model predictions or affinity labels used",
    "fingerprint_policy":"RDKit Morgan radius=2, 2048 bits, chirality enabled; kernel Tanimoto^4",
    "results":results,
    "distinct_compound_feature_cache_size":len(feature_cache),
    "warning":"Coverage can change after assay/construct and chemical-standardization QC; high Tanimoto or scaffold identity does not guarantee similar affinity; low similarity does not prove model failure."
}
(audit/"bindingdb_full_kd_support_chemical_coverage.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
