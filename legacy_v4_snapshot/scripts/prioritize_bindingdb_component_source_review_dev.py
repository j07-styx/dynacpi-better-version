"""Label-blind, component-balanced source review *triage*, not assay approval.

This script consumes the scientific PRE_SPLIT BindingDB technical candidate
TASK metadata and original source bibliographic/construct fields, but NEVER
uses any candidate numeric affinity or query/support outcome to rank leads.

Review priority is metadata quality only. Every lead remains UNREVIEWED and
must pass independent source, construct, chemistry, assay and split review.
No original BALM frozen development/prospective test is accessed.
"""
import csv
import hashlib
import json
import re
import html
from collections import Counter,defaultdict
from pathlib import Path

from dynacpi.balm_public_data_dev import ROOT

AUDIT=ROOT/"data/audit"
REPORT=ROOT/"research/bindingdb_component_balanced_source_review_priority_20261108.json"
MARKDOWN=ROOT/"research/BINDINGDB_COMPONENT_BALANCED_SOURCE_REVIEW_TRIAGE_20261108.md"
WORKLIST=ROOT/"research/bindingdb_component_balanced_source_review_priority_20261108.csv"
SLOTS=40
LITERATURE_SOURCES={"ChEMBL","Curated from the literature by BindingDB"}
PHYSICAL_METHOD_PATTERNS={
    "SPR":(r"\bsurface plasmon resonance\b",r"\bSPR\b",r"\bbiacore\b"),
    "ITC":(r"\bisothermal titration calorimetry\b",r"\bITC\b"),
    "BLI":(r"\bbiolayer interferometry\b",r"\bbio-layer interferometry\b",
           r"\bBLI\b",r"\boctet\b"),
}
ASSAY_RISK_PATTERNS={
    "competition_or_displacement":(
        r"\bcompetition\b",r"\bcompetitive\b",r"\bdisplacement\b"),
    "indirect_activity_or_probe":(
        r"\bkinobead",r"\bpull.down\b",r"\bIC50\b",r"\bKI\b",
        r"\binhibition\b",r"\benzyme activity\b",
        r"\bfluorescen",r"\bpolarization\b",r"\banisotropy\b",
        r"\bFITC\b"),
    "kinetics_not_equilibrium_kd":(
        r"\bdissociation rate constant\b",r"\bassociation rate constant\b",
        r"\boff.rate\b",r"\bSPR response\b",r"\bkinetic rate\b"),
    "construct_requires_review":(
        r"\bunknown origin\b",r"\bchimeric\b",r"\bmutant\b",
        r"\bprotein complex\b",r"\bheterodimer\b"),
}

def load_jsonl(path):
    with path.open(encoding="utf8") as src:
        for ln in src:
            if ln.strip():
                yield json.loads(ln)

def source_description_flags(description):
    """Word boundaries avoid false ITC matches within FITC probe names."""
    text=html.unescape(description or "")
    methods=tuple(sorted(k for k,patterns in PHYSICAL_METHOD_PATTERNS.items()
                         if any(re.search(pattern,text,re.IGNORECASE)
                                for pattern in patterns)))
    risk=tuple(sorted(k for k,patterns in ASSAY_RISK_PATTERNS.items()
                      if any(re.search(pattern,text,re.IGNORECASE)
                             for pattern in patterns)))
    return methods,risk
def canonical_source(s):
    return (s or "").strip()

def metadata_indicators(task,metadata,assay):
    """Never accesses Kd, pKd, label, median, model prediction or test score."""
    assert metadata["count"]>=1
    dois=metadata["doi"]
    pmids=metadata["pmid"]
    sources=metadata["source"]
    hashes=metadata["sequence_sha256"]
    target=metadata["target"]
    uniprot=metadata["uniprot"]
    patents=metadata["patent"]
    ph=metadata["ph"]
    temp=metadata["temp"]
    methods,risks=source_description_flags(assay.get("description",""))
    source_match=(hashes=={task["protein_seq_sha256"]})
    unique_reference=(len(uniprot)==1)
    unique_target=(len(target)==1)
    single_source=(len(sources)==1)
    doi_traced=(len(dois)==1)
    pmid_traced=(len(pmids)==1)
    paper_source=(not patents and single_source and next(iter(sources)) in LITERATURE_SOURCES)
    assay_text=bool(assay.get("name","").strip() and assay.get("description","").strip())
    # Metadata-only scores: these are NOT measurements of scientific quality.
    score=(
        5*doi_traced+3*pmid_traced+3*source_match+2*unique_reference
        +2*unique_target+2*paper_source+single_source+2*bool(methods)
        +2*assay_text+int(len(ph)==1)+int(len(temp)==1)
        -4*len(risks)-2*bool(patents)-2*bool(len(ph)>1)
        -2*bool(len(temp)>1)
    )
    flags=[]
    for condition,flag in (
        (not doi_traced,"doi_absent_or_multiple"),
        (not pmid_traced,"pmid_absent_or_multiple"),
        (not source_match,"source_construct_sequence_mismatch"),
        (not unique_reference,"uniprot_accession_missing_or_mixed"),
        (not unique_target,"target_annotation_missing_or_mixed"),
        (not single_source,"multiple_original_sources"),
        (not paper_source,"not_single_nonpatent_literature_source"),
        (not assay_text,"assay_text_missing"),
        (not methods,"no_explicit_spr_itc_bli_keyword"),
        (bool(risks),"assay_risk_keyword_needs_method_review"),
        (len(ph)!=1,"ph_missing_or_inconsistent"),
        (len(temp)!=1,"temperature_missing_or_inconsistent"),
    ):
        if condition:flags.append(flag)
    return {
        "metadata_priority_score":int(score),
        "has_one_article_doi":doi_traced,
        "has_one_pmid":pmid_traced,
        "source_construct_hash_matched":source_match,
        "unique_uniprot_accession":unique_reference,
        "unique_target_name":unique_target,
        "single_source":single_source,
        "nonpatent_literature_source":paper_source,
        "assay_text_present":assay_text,
        "explicit_physical_binding_method_keywords":list(methods),
        "potential_indirect_or_competition_keywords":list(risks),
        "ph_values_reported":len(ph),
        "temp_values_reported":len(temp),
        "metadata_flags":flags,
    }

def gather():
    tasks={}
    counts=Counter()
    for task in load_jsonl(AUDIT/"bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl"):
        counts["all_pre_split_technical_tasks"]+=1
        if int(task["n_unique_compounds"])<15:
            continue
        k=(task["protein_seq_sha256"],task["assay_key"])
        if k in tasks:raise ValueError("Duplicate eligible protein/assay pair")
        tasks[k]=task
    assert len(tasks)==1146
    counts["eligible_tasks_at_least_15_compounds"]=len(tasks)
    counts["distinct_eligible_protein_components"]=len({x["protein_component_id"] for x in tasks.values()})
    assay_context={}
    for assay in load_jsonl(AUDIT/"bindingdb_full_kd_assay_contexts.jsonl"):
        key=(assay["sequence_sha256"],assay["assay_key"])
        if key in tasks:
            if key in assay_context:raise ValueError("Duplicate assay context")
            assay_context[key]=assay
    # Exact IDs from the previously screened PRE_SPLIT technical compound
    # inventory. Access task_id/reactant_ids ONLY (never median_pKd or y).
    valid_by_task=defaultdict(set)
    for compound in load_jsonl(AUDIT/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        task_id=compound["task_id"]
        valid_by_task[task_id].update(str(x) for x in compound["reactant_ids"])
    eligible_rids={}
    for candidate in load_jsonl(AUDIT/"bindingdb_full_kd_candidates.jsonl"):
        key=(hashlib.sha256(candidate["sequence"].encode()).hexdigest(),
             candidate["assay_key"])
        if key in tasks:
            rid=str(candidate["reactant_set_id"])
            if rid not in valid_by_task[tasks[key]["task_id"]]:
                continue  # prior technical screen excluded this raw row
            if rid in eligible_rids:raise ValueError("Duplicate reactant set ID")
            eligible_rids[rid]=key
    assert len(eligible_rids)==43629,len(eligible_rids)
    counts["eligible_source_reactant_ids"]=len(eligible_rids)
    metadata=defaultdict(lambda:{
        "count":0,
        "source":set(),"doi":set(),"pmid":set(),"patent":set(),
        "sequence_sha256":set(),"target":set(),"uniprot":set(),
        "ph":set(),"temp":set()
    })
    seen=set()
    for row in load_jsonl(AUDIT/"bindingdb_full_kd_source_provenance.jsonl"):
        rid=str(row["BindingDB Reactant_set_id"])
        key=eligible_rids.get(rid)
        if key is None:continue
        if rid in seen:raise ValueError("Duplicate source record")
        seen.add(rid)
        m=metadata[key]
        m["count"]+=1
        m["sequence_sha256"].add(row["protein_construct_sequence_sha256"])
        for dest,src in (
            ("source","Curation/DataSource"),("doi","Article DOI"),
            ("pmid","PMID"),("patent","Patent Number"),
            ("target","Target Name"),("ph","pH"),("temp","Temp (C)")
        ):
            value=canonical_source(row.get(src,""))
            if value:m[dest].add(value)
        for src in (
            "UniProt (SwissProt) Primary ID of Target Chain 1",
            "UniProt (TrEMBL) Primary ID of Target Chain 1"
        ):
            value=canonical_source(row.get(src,""))
            if value:m["uniprot"].add(value)
    if seen!=set(eligible_rids):
        raise ValueError("Source lineage incomplete: missing "
                         +str(len(set(eligible_rids)-seen))+" reactant IDs")
    counts["reactant_ids_with_exact_source_row"]=len(seen)
    if len(assay_context)!=len(tasks):
        raise ValueError("Eligible assay contexts missing")
    return tasks,metadata,assay_context,counts

def choose_review_leads(items,max_slots=SLOTS):
    """Sort by provenance evidence, then chemistry *count*, not affinity."""
    ordered=sorted(items,key=lambda x:(
        -x["metadata_priority_score"],-x["n_unique_compounds"],x["task_id"]))
    selected=[];seen=set()
    for item in ordered:
        comp=item["protein_component_id"]
        if comp in seen:continue
        seen.add(comp)
        selected.append(item)
        if len(selected)>=max_slots:break
    return selected

def main():
    if any(p.exists() for p in (REPORT,MARKDOWN,WORKLIST)):
        raise FileExistsError("Review artifact already exists; refuse overwrite")
    tasks,source,assays,counts=gather()
    with (AUDIT/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv").open(
        encoding="utf-8-sig",newline="") as f:
        old_review=list(csv.DictReader(f))
    old_review_task={r["task_id"] for r in old_review}
    assert len(old_review_task)==30
    items=[]
    indicators=Counter()
    for key,task in tasks.items():
        s=source[key]
        a=assays[key]
        checks=metadata_indicators(task,s,a)
        for field,value in checks.items():
            if field in (
                "has_one_article_doi","has_one_pmid",
                "source_construct_hash_matched","unique_uniprot_accession",
                "unique_target_name","nonpatent_literature_source",
                "assay_text_present"
            ) and value:
                indicators[field]+=1
        methods=checks["explicit_physical_binding_method_keywords"]
        indicators["tasks_with_any_SPR_ITC_BLI_keyword"]+=bool(methods)
        indicators["tasks_with_any_assay_risk_keyword"]+=bool(
            checks["potential_indirect_or_competition_keywords"])
        indicators["tasks_with_reported_ph"]+=bool(s["ph"])
        indicators["tasks_with_reported_temperature"]+=bool(s["temp"])
        items.append({
            "task_id":task["task_id"],
            "protein_component_id":task["protein_component_id"],
            "protein_sha256":task["protein_seq_sha256"],
            "assay_key":task["assay_key"],
            "n_unique_compounds":int(task["n_unique_compounds"]),
            "raw_source_records":s["count"],
            "source_category":sorted(s["source"]),
            "doi_candidates":sorted(s["doi"]),
            "pmid_candidates":sorted(s["pmid"]),
            "uniprot_accessions":sorted(s["uniprot"]),
            "target_names":sorted(s["target"]),
            "assay_name":a.get("name",""),
            "assay_description":a.get("description",""),
            "previous_30_task_worklist":task["task_id"] in old_review_task,

            "status":"UNREVIEWED_METADATA_PRIORITY_ONLY_NOT_APPROVED",
            **checks,
        })
    shortlist=choose_review_leads(items)
    assert len(shortlist)==40
    assert len({r["protein_component_id"] for r in shortlist})==40
    bysource=Counter(";".join(x["source_category"]) for x in shortlist)
    result={
        "status":"LABEL_BLIND_COMPONENT_BALANCED_SOURCE_TRIAGE_NOT_ASSAY_APPROVED",
        "selection_rules":{
            "metadata_priority_only":True,
            "max_one_task_per_preliminary_mmseqs_component":True,
            "source_metadata_priority":"article DOI, PMID, exact source sequence hash, one accession and target annotation, nonpatent literature, assay method clues and condition completeness",
            "tie_breakers":"negative number of unique compounds (identity count), then lexicographic SHA256 task ID",
            "never_used_for_priority":["pKd","Kd numeric value","median affinity","model score","development outcome","prospective test label","replicate disagreement magnitude"],
            "review_slots":SLOTS,
        },
        "full_candidate_counts":dict(counts),
        "metadata_counts_across_eligible_tasks":dict(indicators),
        "shortlist_distinct_components":len({r["protein_component_id"] for r in shortlist}),
        "shortlist_source_categories":dict(bysource),
        "shortlist_with_prior_worklist_overlap":sum(r["previous_30_task_worklist"] for r in shortlist),
        "shortlist_with_doi":sum(r["has_one_article_doi"] for r in shortlist),
        "shortlist_with_pmid":sum(r["has_one_pmid"] for r in shortlist),
        "shortlist_with_spr_itc_bli_keyword":sum(bool(r["explicit_physical_binding_method_keywords"]) for r in shortlist),
        "shortlist_with_risk_keyword":sum(bool(r["potential_indirect_or_competition_keywords"]) for r in shortlist),
        "shortlist_missing_ph":sum(not r["ph_values_reported"] for r in shortlist),
        "shortlist_missing_temperature":sum(not r["temp_values_reported"] for r in shortlist),
        "shortlist":shortlist,
        "scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "approval_count":0,
        "reserved_balm_prospective_test_touched":False,
        "warning":[
            "Metadata presence/keywords do not verify original paper measurements or assay semantics.",
            "Source construct hash equality to candidate sequence does not confirm original wet-lab protein construct.",
            "Source DOI or PMID points to a publication, not an approved affinity result.",
            "Original preliminary MMseqs homology components are not a scientifically certified split.",
            "No affinity labels or model results were consulted to rank the worklist.",
            "A reviewer must inspect source tables, compound identities, assay conditions, construct and exact Kd endpoint; all remain unreviewed.",
            "No final scientific cohort or split was generated.",
        ],
    }
    REPORT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    csvfields=[
        "rank","task_id","protein_component_id","protein_sha256",
        "assay_key","n_unique_compounds","raw_source_records",
        "metadata_priority_score","source_category","doi_candidates",
        "pmid_candidates","uniprot_accessions","target_names",
        "assay_name","assay_description",
        "explicit_physical_binding_method_keywords",
        "potential_indirect_or_competition_keywords",
        "metadata_flags","previous_30_task_worklist","reviewer_decision"
    ]
    with WORKLIST.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=csvfields)
        w.writeheader()
        for i,row in enumerate(shortlist,1):
            d={k:row.get(k,"") for k in csvfields}
            d["rank"]=i
            for key in (
                "source_category","doi_candidates","pmid_candidates",
                "uniprot_accessions","target_names",
                "explicit_physical_binding_method_keywords",
                "potential_indirect_or_competition_keywords","metadata_flags"
            ):
                d[key]="; ".join(row[key])
            d["reviewer_decision"]="UNREVIEWED"
            w.writerow(d)
    md=[
        "# Component-balanced BindingDB source-review priorities",
        "",
        "**Unreviewed metadata screen — NOT an approved affinity cohort, not experimental validation.**",
        "",
        f"- Source: full November 2026 BindingDB *pre-split*, technically screened exact-looking-Kd candidate pool; {len(tasks)} eligible tasks from {counts['distinct_eligible_protein_components']} preliminary MMseqs protein components.",
        f"- Complete source ID lineage: {counts['reactant_ids_with_exact_source_row']:,} eligible reactant IDs joined to original bibliographic and construct metadata; none missing.",
        f"- Candidate tasks with single DOI: {indicators['has_one_article_doi']}; with single PMID: {indicators['has_one_pmid']}; with SPR/ITC/BLI keywords: {indicators['tasks_with_any_SPR_ITC_BLI_keyword']}.",
        f"- Candidate tasks with any reported pH: {indicators['tasks_with_reported_ph']}; temperature: {indicators['tasks_with_reported_temperature']}. Missing conditions are a warning rather than automatic proof of incompatibility.",
        f"- Selected {len(shortlist)} review leads from {len({x['protein_component_id'] for x in shortlist})} distinct protein components; overlap with previous unreviewed 30-task queue: {result['shortlist_with_prior_worklist_overlap']}.",
        "- Ranking uses only source documentation, sequence metadata and unique molecular identity counts. It DOES NOT use numeric Kd/pKd labels, prediction scores, or holdout performance.",
        "",
        "## Top review leads (metadata evidence only)",
        "",
        "| Rank | Task ID prefix | Component prefix | Unique compounds | Source DOI | Explicit SPR/ITC/BLI clue | Flags |",
        "|---:|---|---|---:|---|---|---|"
    ]
    for i,r in enumerate(shortlist[:20],1):
        doi=r["doi_candidates"][0] if len(r["doi_candidates"])==1 else "(none/multiple)"
        method=",".join(r["explicit_physical_binding_method_keywords"]) or "not stated"
        flags=",".join(r["metadata_flags"][:2]) or "no metadata flags"
        md.append(f"| {i} | {r['task_id'][:12]} | {r['protein_component_id'][:12]} | {r['n_unique_compounds']} | {doi} | {method} | {flags} |")
    md+=["",
        "## Manual verification required before a source can be approved",
        "",
        "1. Open the original publication or patent and identify the exact table, supplementary cell or deposited result for the BindingDB reactant IDs.",
        "2. Confirm experimental Kd as a directly measured equilibrium dissociation constant, not IC50, Ki, EC50, competition-derived estimate, mixture, censored bound or imputed value.",
        "3. Establish compound structure, salt/stereochemistry/tautomer equivalence between publication and BindingDB source for all task records.",
        "4. Confirm experimentally assayed protein construct, truncations, variants, post-translational modifications and sequence relative to UniProt/candidate source.",
        "5. Assess assay-method comparability, experimental conditions and within-task replicate discrepancies. Require explicit evidence for acceptance.",
        "6. Have an independent reviewer adjudicate each task, then validate source evidence manifest and protein/chemical split before an approval artifact is possible.",
        "",
        "## Files and integrity",
        "",
        "- Machine-readable metadata evidence: research/bindingdb_component_balanced_source_review_priority_20261108.json.",
        "- Review queue CSV: research/bindingdb_component_balanced_source_review_priority_20261108.csv. Its reviewer_decision column is initialized UNREVIEWED; this is NOT an authorized gate artifact.",
        "- Reproduce via scripts/prioritize_bindingdb_component_source_review_dev.py.",
        "- This work does NOT write data/audit/bindingdb_source_approved_cohort.json or a frozen protein split and does not change the scientific cohort gate.",
        "",
    ]
    MARKDOWN.write_text("\n".join(md),encoding="utf8")
    print("TRIAGE_COMPLETE",json.dumps({
        "eligible":len(tasks),"components":counts["distinct_eligible_protein_components"],
        "source_lineages":len(eligible_rids) if False else counts["reactant_ids_with_exact_source_row"],
        "shortlist":len(shortlist),"indicators":dict(indicators),
        "shortlist_doi":result["shortlist_with_doi"],
        "shortlist_methods":result["shortlist_with_spr_itc_bli_keyword"],
        "shortlist_prior_queue_overlap":result["shortlist_with_prior_worklist_overlap"],
        "report":str(REPORT)
    }),flush=True)

if __name__=="__main__":main()
