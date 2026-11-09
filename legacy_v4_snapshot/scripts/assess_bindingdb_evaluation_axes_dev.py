"""Assess candidate split viability without labels, training, approvals or split freeze."""
import collections,hashlib,json,sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.evaluation_axes import plan_axes,summarize
a=ROOT/"data/audit"
out=ROOT/"research/bindingdb_evaluation_axes_20261107.json"
if out.exists():raise FileExistsError(out)
hard={"multiple_target_name_annotations","multiple_or_missing_uniprot_ids",
"within_assay_ph_variation","within_assay_temp_variation","replicates_disagree_over_0_1_pKd",
"some_multicomponent_smiles","assay_text_unavailable"}
tasks={}
with (a/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl").open() as f:
    for line in f:
        r=json.loads(line)
        if r["n_unique_compounds"]>=15 and not set(r["quality_review_flags"])&hard:
            tasks[r["task_id"]]={k:r[k] for k in ("protein_component_id","n_unique_compounds")}
scaffolds={};rows=[]
with (a/"bindingdb_full_kd_compound_median_candidates.jsonl").open() as f:
    for line in f:
        r=json.loads(line)
        if r["task_id"] not in tasks:continue
        # Deliberately whitelist identities; no affinity/range values enter split logic.
        cid=r["compound_sha256"];sm=r["smiles_isomeric"]
        if hashlib.sha256(sm.encode()).hexdigest()!=cid:raise ValueError("Chemical hash mismatch")
        if cid not in scaffolds:
            mol=Chem.MolFromSmiles(sm)
            if mol is None:raise ValueError("Invalid SMILES")
            core=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
            scaffolds[cid]=hashlib.sha256(core.encode()).hexdigest() if core else "ACYCLIC"
        rows.append({"task_id":r["task_id"],"protein_component_id":tasks[r["task_id"]]["protein_component_id"],
                    "compound_id":cid,"scaffold_id":scaffolds[cid]})
counts=collections.Counter(r["task_id"] for r in rows)
assert all(counts[t]==m["n_unique_compounds"] for t,m in tasks.items())
assignments=plan_axes(rows,1105)
report={"status":"UNAPPROVED_METADATA_FEASIBILITY_ONLY","seed":1105,"tasks":len(tasks),
    "protein_components":len({r["protein_component_id"] for r in rows}),
    "compounds":len(scaffolds),"scaffolds":len(set(scaffolds.values())),"pairs":len(rows),
    "acyclic_compounds":sum(v=="ACYCLIC" for v in scaffolds.values()),
    "axes":summarize(rows,assignments),
    "limits":["No source approval or test membership is written.",
             "Protein components are preliminary; homology thresholds and domains remain subject to review.",
             "Scaffold-disjoint does not bound all chemical similarity; acyclic compounds kept together.",
             "The 15-compound threshold is 5 support plus 10 query; counts are viability, not assay approval.",
             "No seed search; single predeclared seed 1105.",
             "Candidate JSON includes affinity fields, but values are neither used for assignments nor reported."]}
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
