"""Build scientifically QUARANTINED exact-Kd candidate tasks and disjoint episode identities.

Inputs: full November 2026 BindingDB screened reactant IDs, source metadata,
assay description and protein components. NO inference claims or final splits.
Selection depends only on query-compound identity, not on its affinity value.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
source={
    x["BindingDB Reactant_set_id"]:x
    for x in (json.loads(ln) for ln in (audit/"bindingdb_full_kd_source_provenance.jsonl").open(encoding="utf-8"))
}
assays={(x["sequence_sha256"],x["assay_key"]):x for x in
        (json.loads(ln) for ln in (audit/"bindingdb_full_kd_assay_contexts.jsonl").open(encoding="utf-8"))}
component={}
for ln in (audit/"bindingdb_full_kd_protein_components.jsonl").open(encoding="utf-8"):
    rec=json.loads(ln)
    for sha in rec["members"]:
        if sha in component:raise AssertionError("Duplicate protein component membership")
        component[sha]=rec["component_id"]
tasks=defaultdict(lambda:defaultdict(list))
meta=defaultdict(lambda:defaultdict(set))
stats=Counter()
for ln in (audit/"bindingdb_full_kd_candidates.jsonl").open(encoding="utf-8"):
    rec=json.loads(ln)
    stats["source_candidates"]+=1
    rid=rec["reactant_set_id"]
    source_row=source.get(rid)
    if source_row is None:
        stats["no_provenance"]+=1;continue
    seqsha=hashlib.sha256(rec["sequence"].encode()).hexdigest()
    if source_row["protein_construct_sequence_sha256"]!=seqsha:
        stats["source_sequence_hash_mismatch"]+=1;continue
    raw_smi=rec["smiles_raw"]
    mol=Chem.MolFromSmiles(raw_smi)
    if mol is None:
        stats["invalid_rdkit_molecule"]+=1;continue
    y=float(rec["pKd"])
    if not math.isfinite(y) or not 0<y<20:
        stats["nonfinite_or_extreme_pkd"]+=1;continue
    kd=float(source_row["Kd (nM)"].strip())
    if not math.isfinite(kd) or kd<=0 or abs(9-math.log10(kd)-y)>1e-9:
        stats["source_kd_mismatch"]+=1;continue
    smi=Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)
    frag_count=len(Chem.GetMolFrags(mol))
    comp_sha=hashlib.sha256(smi.encode()).hexdigest()
    key=(seqsha,rec["assay_key"])
    entry={"reactant_id":rid,"y":y,"source":source_row["Curation/DataSource"],
           "doi":source_row["Article DOI"],"fragments":frag_count}
    tasks[key][(comp_sha,smi)].append(entry)
    m=meta[key]
    m["names"].add(source_row["Target Name"])
    x=source_row["UniProt (SwissProt) Primary ID of Target Chain 1"] or source_row["UniProt (TrEMBL) Primary ID of Target Chain 1"]
    if x:m["uniprot"].add(x)
    if source_row["pH"]:m["ph"].add(source_row["pH"])
    if source_row["Temp (C)"]:m["temp"].add(source_row["Temp (C)"])
    stats["records_after_technical_checks"]+=1

if not tasks:raise RuntimeError("No technically valid candidates")
rows_path=audit/"bindingdb_full_kd_compound_median_candidates.jsonl"
episodes_path=audit/"bindingdb_full_kd_candidate_episodes_PRE_SPLIT.jsonl"
tasks_path=audit/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl"
with rows_path.open("w",encoding="utf-8") as rowf,episodes_path.open("w",encoding="utf-8") as epf,tasks_path.open("w",encoding="utf-8") as taskf:
    for (seqsha,assay),compounds in sorted(tasks.items()):
        task_id=hashlib.sha256((seqsha+"|"+assay).encode()).hexdigest()
        ass=assays.get((seqsha,assay))
        flags=[]
        if not ass or not ass["name"] or not ass["description"]:
            flags.append("assay_text_unavailable")
        if len(meta[(seqsha,assay)]["names"])!=1:
            flags.append("multiple_target_name_annotations")
        if len(meta[(seqsha,assay)]["uniprot"])!=1:
            flags.append("multiple_or_missing_uniprot_ids")
        if len(meta[(seqsha,assay)]["ph"])>1:
            flags.append("within_assay_ph_variation")
        if len(meta[(seqsha,assay)]["temp"])>1:
            flags.append("within_assay_temp_variation")
        if not meta[(seqsha,assay)]["ph"]:
            flags.append("ph_not_reported")
        if not meta[(seqsha,assay)]["temp"]:
            flags.append("temperature_not_reported")
        if seqsha not in component:
            raise RuntimeError("Protein sequence not clustered: "+seqsha)
        crows=[]
        duplicate_compound_count=0
        duplicate_spread_over01=0
        multipart_compounds=0
        for (compound_id,smi),records in sorted(compounds.items()):
            ys=[x["y"] for x in records]
            spread=max(ys)-min(ys)
            if len(records)>1:
                duplicate_compound_count+=1
                if spread>0.1:duplicate_spread_over01+=1
            if max(x["fragments"] for x in records)>1:
                multipart_compounds+=1
            item={"task_id":task_id,"protein_seq_sha256":seqsha,"assay_key":assay,
                  "compound_sha256":compound_id,"smiles_isomeric":smi,
                  "median_pKd":statistics.median(ys),
                  "replicates":len(ys),"range_pKd":spread,
                  "reactant_ids":[x["reactant_id"] for x in records]}
            rowf.write(json.dumps(item)+"\n")
            crows.append(item)
        if duplicate_spread_over01:
            flags.append("replicates_disagree_over_0_1_pKd")
        if multipart_compounds:
            flags.append("some_multicomponent_smiles")
        if len(crows)>=15:
            stats["tasks_ge15"]+=1
        if len(crows)>=30:
            stats["tasks_ge30"]+=1
        stats["technical_tasks"]+=1
        # SHA-based deterministic order is independent of actual affinity values.
        ranked=sorted(crows,key=lambda x:hashlib.sha256(("query_selection_v4|"+task_id+"|"+x["compound_sha256"]).encode()).hexdigest())
        query_ids=[r["compound_sha256"] for r in ranked[:10]] if len(ranked)>=15 else []
        support_ids=[r["compound_sha256"] for r in ranked[10:]] if len(ranked)>=15 else []
        if query_ids and (len(support_ids)<5 or len(set(query_ids)&set(support_ids))):
            raise AssertionError("Insufficient or overlapping query/support identities")
        t={"task_id":task_id,"protein_seq_sha256":seqsha,
           "protein_component_id":component[seqsha],
           "assay_key":assay,"n_unique_compounds":len(crows),
           "n_duplicate_compounds":duplicate_compound_count,
           "n_discordant_replicate_groups":duplicate_spread_over01,
           "n_multicomponent_smiles":multipart_compounds,
           "quality_review_flags":sorted(set(flags)),
           "assay_description_sha256":hashlib.sha256((ass["description"] if ass else "").encode()).hexdigest(),
           "status":"PRE_SPLIT_TECHNICAL_CANDIDATE_NOT_ASSAY_VERIFIED"}
        taskf.write(json.dumps(t)+"\n")
        if query_ids:
            draws=[]
            for draw in range(10):
                ranked_support=sorted(support_ids,key=lambda x:hashlib.sha256(
                    ("support_draw_v4|"+task_id+"|"+str(draw)+"|"+x).encode()).hexdigest())
                draws.append({"draw_index":draw,
                    "support_prefixes":{str(k):ranked_support[:k] for k in (0,1,2,5,10,20)
                                        if k<=len(ranked_support)}})
            distinct_k5=len({tuple(d["support_prefixes"]["5"]) for d in draws})
            if distinct_k5<10:stats["tasks_with_fewer_than_10_distinct_k5_draws"]+=1
            epf.write(json.dumps({"task_id":task_id,"query_ids":query_ids,
                 "candidate_support_ids":support_ids,
                 "support_draws":draws,
                 "distinct_k5_support_draws":distinct_k5,
                 "status":"DO_NOT_USE_FOR_FINAL_BENCHMARK_UNTIL_ASSAY_QC_AND_SPLIT_LOCK"})+"\n")
            if duplicate_spread_over01:stats["eligible_tasks_with_discordant_replicates"]+=1
            if "multiple_target_name_annotations" in flags:stats["eligible_tasks_with_multiple_target_names"]+=1
            if "multiple_or_missing_uniprot_ids" in flags:stats["eligible_tasks_with_uniprot_ambiguity"]+=1
            if not meta[(seqsha,assay)]["ph"]:stats["eligible_tasks_missing_ph"]+=1
            if not meta[(seqsha,assay)]["temp"]:stats["eligible_tasks_missing_temp"]+=1

report={"status":"TECHNICAL_ONLY_QUARANTINED_NOT_EXPERIMENTALLY_VERIFIED",
 "input_source":"BindingDB full 202611 and MMseqs2 preliminary components",
 "counts":dict(stats),
 "component_threshold":">=30% identity over >=80% shorter length; MMseqs2 local search",
 "episode_policy":"10 query compounds by task+compound hash; 10 predetermined nested support draws per task by compound hash, disjoint under canonical isomeric SMILES; no split assigned",
 "task_manifest":tasks_path.name,"episode_manifest":episodes_path.name,"compound_medians":rows_path.name,
 "warning":"Do not train/validate or claim assay provenance from these files until construct/assay QC and group splits are locked. Source conditions and chemical standardization can alter eligibility."}
(audit/"bindingdb_full_kd_pre_split_technical_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
