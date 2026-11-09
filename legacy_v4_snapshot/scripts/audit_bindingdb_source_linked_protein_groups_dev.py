"""Conservative protein-component AND source-publication linkage preflight.

Original protein MMseqs components are joined whenever eligible tasks share a
traceable DOI, PMID, or patent number. This is a source-batch-leakage
SENSITIVITY GRAPH, not certified protein homology or biological validation.
Selection uses metadata only; NO affinity labels, development/test results.
"""
import csv
import hashlib
import json
from collections import Counter,defaultdict

from dynacpi.balm_public_data_dev import ROOT
from scripts.prioritize_bindingdb_component_source_review_dev import (
    gather,metadata_indicators,choose_review_leads,SLOTS,
)

AUDIT=ROOT/"data/audit"
OUT=ROOT/"research/bindingdb_publication_connected_protein_review_preflight_20261108.json"
WORKLIST=ROOT/"research/bindingdb_publication_disjoint_source_review_queue_20261108.csv"
REPORT=ROOT/"research/BINDINGDB_PUBLICATION_CONNECTED_COMPONENT_PREFLIGHT_20261108.md"

def stable_hash(raw):
    return hashlib.sha256(raw.encode("utf8")).hexdigest()

class DisjointSet:
    def __init__(self,nodes):
        self.parent={key:key for key in nodes}
        self.size={key:1 for key in nodes}
    def find(self,key):
        parent=self.parent[key]
        if parent!=key:
            self.parent[key]=self.find(parent)
        return self.parent[key]
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return False
        if self.size[a]<self.size[b] or (
                self.size[a]==self.size[b] and a>b):
            a,b=b,a
        self.parent[b]=a
        self.size[a]+=self.size[b]
        return True
    def groups(self):
        groups=defaultdict(set)
        for x in self.parent:
            groups[self.find(x)].add(x)
        return [tuple(sorted(v)) for v in groups.values()]

def source_ids(meta):
    """Only exact non-empty source identifiers, no direct numeric Kd values."""
    vals=set()
    for category in ("doi","pmid","patent"):
        for value in meta[category]:
            v=value.strip().lower()
            if category=="doi":
                v=v.removeprefix("https://doi.org/").removeprefix("doi:")
            if not v or v in ("na","n/a","unknown","none","-"):
                continue
            # An article publication group may share PMID and DOI; separate
            # token prefix avoids collisions between identifier namespaces.
            vals.add(category+":"+v)
    return vals

def connect_components(tasks,source):
    """Conservative union of preliminary protein components by study identity."""
    components={r["protein_component_id"] for r in tasks.values()}
    dsu=DisjointSet(components)
    publications=defaultdict(set)
    for key,task in tasks.items():
        component=task["protein_component_id"]
        for publication in source_ids(source[key]):
            publications[publication].add(component)
    edge_counts=Counter()
    shared={}
    for publication,comps in sorted(publications.items()):
        if len(comps)<2:continue
        ordered=sorted(comps)
        edge_counts[publication.split(":",1)[0]+"_cross_component_publications"]+=1
        for component in ordered[1:]:
            if dsu.union(ordered[0],component):
                edge_counts["unions"]+=1
        if len(shared)<30:
            shared[publication]={"components":len(comps),
                                 "component_prefixes":[x[:12] for x in ordered[:8]]}
    resulting=dsu.groups()
    all_orig=set()
    membership={}
    for members in resulting:
        ident=stable_hash("source_group_v1|"+"|".join(members))
        for component in members:
            assert component not in all_orig
            all_orig.add(component)
            membership[component]=ident
    assert all_orig==components
    return {
        "components":len(components),
        "source_connected_families":len(resulting),
        "family_sizes":sorted((len(x) for x in resulting),reverse=True),
        "cross_component_source_identifiers":dict(edge_counts),
        "shared_source_identifier_count":sum(len(comps)>1 for comps in publications.values()),
        "source_identifier_total":len(publications),
        "shared_identifier_examples":shared,
        "source_family_by_protein_component":membership,
        "family_members":{stable_hash("source_group_v1|"+"|".join(members)):list(members)
                          for members in resulting},
    }

def choose_source_independent_leads(items,mapping,k=SLOTS):
    """One review lead per provenance-connected family, not just protein."""
    byscore=sorted(items,key=lambda row:(
        -row["metadata_priority_score"],-row["n_unique_compounds"],
        row["task_id"]))
    chosen=[];used=set()
    for item in byscore:
        family=mapping[item["protein_component_id"]]
        if family in used:continue
        used.add(family)
        chosen.append({**item,"source_family_id":family})
        if len(chosen)>=k:break
    return chosen

def main():
    if any(path.exists() for path in (OUT,WORKLIST,REPORT)):
        raise FileExistsError("Existing source family preflight; do not overwrite")
    tasks,source,assay,counts=gather()
    linkage=connect_components(tasks,source)
    old=json.loads((ROOT/"research/bindingdb_component_balanced_source_review_priority_20261108.json").read_text())
    assert len(old["shortlist"])==40
    assert len(tasks)==1146
    old_families={
        linkage["source_family_by_protein_component"][r["protein_component_id"]]
        for r in old["shortlist"]
    }
    items=[]
    for key,task in tasks.items():
        ind=metadata_indicators(task,source[key],assay[key])
        items.append({
            "task_id":task["task_id"],
            "protein_component_id":task["protein_component_id"],
            "n_unique_compounds":task["n_unique_compounds"],
            "assay_key":task["assay_key"],
            "protein_sha256":task["protein_seq_sha256"],
            "source_category":sorted(source[key]["source"]),
            "doi_candidates":sorted(source[key]["doi"]),
            "pmid_candidates":sorted(source[key]["pmid"]),
            "patent_candidates":sorted(source[key]["patent"]),
            "uniprot_accessions":sorted(source[key]["uniprot"]),
            "target_names":sorted(source[key]["target"]),
            "assay_name":assay[key]["name"],
            "assay_description":assay[key]["description"],
            "original_source_rows":source[key]["count"],
            "review_decision":"UNREVIEWED",
            **ind,
        })
    shortlist=choose_source_independent_leads(
        items,linkage["source_family_by_protein_component"])
    assert len(shortlist)==40
    assert len({r["protein_component_id"] for r in shortlist})==40
    assert len({r["source_family_id"] for r in shortlist})==40
    # Public cross-protein DOI/PMID/patent links do not cross shortlist slots.
    used_ids={}
    for i,row in enumerate(shortlist,1):
        key=(row["protein_sha256"],row["assay_key"])
        for x in source_ids(source[key]):
            assert x not in used_ids,(x,i,used_ids[x])
            used_ids[x]=i
    initial_shortlist_shared=len(old["shortlist"])-len(old_families)
    count_families=Counter(
        linkage["source_family_by_protein_component"][r["protein_component_id"]]
        for r in tasks.values())
    counts_source={
        "technical_tasks_ge15":len(tasks),
        "preliminary_MMseqs_components":linkage["components"],
        "source_connected_components":linkage["source_connected_families"],
        "protein_components_merged_across_source_edges":
            linkage["components"]-linkage["source_connected_families"],
        "max_source_family_protein_components":max(linkage["family_sizes"]),
        "largest_source_family_task_count":max(count_families.values()),
        "old_40_component_only_shortlist_source_families":len(old_families),
        "old_40_component_only_shortlist_duplicate_families":initial_shortlist_shared,
        "new_review_lead_count":len(shortlist),
        "new_independent_source_family_count":len({r["source_family_id"] for r in shortlist}),
        "new_single_doi_leads":sum(r["has_one_article_doi"] for r in shortlist),
        "new_explicit_method_leads":sum(bool(r["explicit_physical_binding_method_keywords"]) for r in shortlist),
        "new_method_risk_leads":sum(bool(r["potential_indirect_or_competition_keywords"]) for r in shortlist),
    }
    expected_evidence_digests={}
    for name in ("bindingdb_full_kd_candidate_tasks_PRE_SPLIT.jsonl",
                 "bindingdb_full_kd_compound_median_candidates.jsonl",
                 "bindingdb_full_kd_source_provenance.jsonl",
                 "bindingdb_full_kd_candidates.jsonl",
                 "bindingdb_full_kd_assay_contexts.jsonl"):
        p=AUDIT/name
        h=hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""):
                h.update(chunk)
        expected_evidence_digests[name]=h.hexdigest()
    result={
        "status":"QUARANTINED_LABEL_BLIND_PUBLICATION_LINKED_PROTEIN_COMPONENTS_NO_SPLIT_OR_APPROVAL",
        "method":"Union of preliminary MMseqs protein components sharing original BindingDB Article DOI, PMID, or patent number for any technically eligible >=15-compound task; each study family links complete protein components transitively.",
        "data_source_integrity_sha256":expected_evidence_digests,
        "overview":counts_source,
        "technical_metadata_counts":dict(counts),
        "linkage":linkage,
        "source_independent_review_leads":shortlist,
        "ranking_provenance":"Metadata-only indicators defined in scripts/prioritize_bindingdb_component_source_review_dev.py; unique compound identity COUNT breaks ties. Never rank by Kd/pKd value, replicate discrepancy, or model outcomes.",
        "never_approved_tasks":True,
        "scientific_gate":"BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "warnings":[
            "Shared publication DOI/PMID/patent can indicate common assay batch and needs split stratification. It does NOT prove label leakage by itself.",
            "DOI/PMID often absent. Missing identifiers cannot be connected; source families may be undermerged.",
            "All source connections are limited to technically eligible tasks. An ineligible shared-source record can still link studies.",
            "Protein components and study families are not validated assay-coherent protein-homology groups.",
            "A 40-lead review shortlist is not automatically a 40-task approved training cohort.",
            "Numerical Kd/pKd labels and heldout test/development performance were not used for selection.",
            "DO NOT generate or approve a frozen scientific split from this report.",
        ],
    }
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    columns=[
        "rank","task_id","source_family_id","protein_component_id","protein_sha256",
        "assay_key","n_unique_compounds","metadata_priority_score",
        "source_category","doi_candidates","pmid_candidates","patent_candidates",
        "uniprot_accessions","target_names","assay_name","assay_description",
        "explicit_physical_binding_method_keywords",
        "potential_indirect_or_competition_keywords","metadata_flags","review_decision"
    ]
    with WORKLIST.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        for n,row in enumerate(shortlist,1):
            item={k:row.get(k,"") for k in columns}
            item["rank"]=n
            for field in ("source_category","doi_candidates","pmid_candidates",
                          "patent_candidates","uniprot_accessions","target_names",
                          "explicit_physical_binding_method_keywords",
                          "potential_indirect_or_competition_keywords","metadata_flags"):
                item[field]="; ".join(row[field])
            w.writerow(item)
    report=[
        "# Publication-linked protein-group preflight for BindingDB",
        "",
        "**Quarantined metadata-only evidence / 2026-11-08. No source or assay approved. No prospective test touched.**",
        "",
        "## Why protein-disjoint alone is insufficient",
        "",
        "Different protein components can share a BindingDB Article DOI, PubMed ID or patent: a nominally target-disjoint evaluation then reuses study-level assay, publication and library metadata across splits. This does not prove prediction leakage, but must be examined before declaring independent generalization.",
        "",
        "## Resulting candidate independence structure",
        "",
        f"- {counts_source['technical_tasks_ge15']:,} technically eligible source-linked tasks at >=15 distinct compound identities (not scientifically reviewed).",
        f"- {counts_source['preliminary_MMseqs_components']} preliminary MMseqs protein components joined into **{counts_source['source_connected_components']} conservative study families** using matching DOI/PMID/patent identifiers.",
        f"- **{counts_source['protein_components_merged_across_source_edges']} net protein-component merges** arising from shared source identifiers; largest source family has {counts_source['max_source_family_protein_components']} original protein components and {counts_source['largest_source_family_task_count']} tasks.",
        f"- Original 40-component-only metadata shortlist had **{counts_source['old_40_component_only_shortlist_duplicate_families']} repeated source-family positions**. The revised 40-lead queue has **40 distinct protein components AND 40 distinct study-source families**.",
        f"- Revised shortlist has {counts_source['new_single_doi_leads']} single-DOI leads, {counts_source['new_explicit_method_leads']} text mentions of SPR/ITC/BLI, and {counts_source['new_method_risk_leads']} indirect, kinetic or construct-review keywords.",
        "",
        "## Source-linked review leads",
        "",
        "| Rank | Task prefix | Protein component prefix | Study family prefix | Unique compounds | Source DOI | Method hint |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for i,row in enumerate(shortlist[:20],1):
        doi=row["doi_candidates"][0] if len(row["doi_candidates"])==1 else "(no single DOI)"
        m=",".join(row["explicit_physical_binding_method_keywords"]) or "(none)"
        report.append(f"| {i} | {row['task_id'][:12]} | {row['protein_component_id'][:12]} | {row['source_family_id'][:12]} | {row['n_unique_compounds']} | {doi} | {m} |")
    report+=["",
        "## Important interpretation limits",
        "",
        "Publication links are a **conservative source independence guard**, not a proof that all records sharing a DOI are the same assay. PMID, DOI and patent links may merge genuinely distinct assays; missing IDs permit unrecognized source overlap. The groups still require wet-lab construct, assay, chemistry and protein homology certification, and a reviewer-approved split.",
        "",
        "No affinity value or DEV/test score was used to select these 40 leads. All task decisions stay **UNREVIEWED**. Do not use this queue as a training cohort, a validation set or a source approval manifest.",
        "",
        "## Reproduction",
        "",
        "- scripts/audit_bindingdb_source_linked_protein_groups_dev.py",
        "- research/bindingdb_publication_connected_protein_review_preflight_20261108.json",
        "- research/bindingdb_publication_disjoint_source_review_queue_20261108.csv",
        "- tests_v4/bindingdb_publication_group_preflight_test.py",
        "- All source JSONL input checksums recorded in the JSON output; original raw assay records and affinity values remain in local quarantined data/audit and are not committed.",
        "",
    ]
    REPORT.write_text("\n".join(report),encoding="utf8")
    print("PUBLICATION_SOURCE_PREFLIGHT",json.dumps(counts_source),flush=True)

if __name__=="__main__":main()
