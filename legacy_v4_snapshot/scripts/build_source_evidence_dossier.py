"""Build label-blind source evidence dossier; no assay approvals are generated."""
from pathlib import Path
import csv
import html
import json
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
target=root/"research/BindingDB_30_Source_Evidence_Dossier_20261105.md"
source=audit/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
with source.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
p=json.loads((audit/"bindingdb_assay_review_30_publication_screen.json").read_text(encoding="utf-8"))
u=json.loads((audit/"bindingdb_assay_review_30_uniprot_construct_screen.json").read_text(encoding="utf-8"))
c=json.loads((audit/"bindingdb_assay_review_30_construct_clues.json").read_text(encoding="utf-8"))
ps={x["task_id"]:x for x in p["tasks"]}
us={x["task_id"]:x for x in u["tasks"]}
cs={x["task_id"]:x for x in c["tasks"]}
assert len(rows)==len(ps)==len(us)==len(cs)==30
def clean(s,maxlen=180):
    return html.unescape(str(s or "")).replace("|","/").replace("\n"," ").strip()[:maxlen]
lines=[
 "# BindingDB v4 - Evidence dossier for 30 sampled assay tasks",
 "",
 "Status: PRELIMINARY / NOT APPROVED. All thirty tasks require source-level review.",
 "",
 "This dossier contains bibliographic metadata and UniProt references only. It does not contain measured affinities, query labels, predicted values, or protein-group assignments.",
 "",
 "A metadata sequence matching UniProt does not prove the experimentally assayed construct was identical. A PMID or DOI match proves bibliographic indexing, not measurement semantics.",
 "",
 "## Overview",
 "",
 "- 30 candidate tasks from 30 preliminarily distinct protein components.",
 "- "+str(u["references_fetched"])+" UniProt references retrieved, "+str(u["counts"].get("EXACT_REFERENCE_SEQUENCE",0))+" metadata reference sequences matched exactly.",
 "- "+str(p["unique_pubmed_ids_indexed"])+" indexed unique PubMed references; "+str(p["counts"].get("PATENT_ONLY",0))+" patent-only tasks.",
 "- "+str(c["counts"].get("tasks_with_textual_review_clues",0))+" tasks have textual risk clues requiring extra attention.",
 "- Zero approved experimental affinity tasks.",
 "",
 "## Review index",
 "",
 "| ID (short) | Protein accession | Assay | Source | Review flags |",
 "|---|---|---|---|---|"
]
for r in rows:
    id=r["task_id"];pub=ps[id];unip=us[id];clue=cs[id]
    titles=[x.get("title","") for x in pub["indexed_publications"] if x.get("title")]
    source_title=titles[0] if titles else (",".join(pub["patents"]) or "Unindexed source")
    flags=", ".join(clue["review_clues"]) or "None detected (still review)"
    lines.append("| "+id[:12]+" | "+clean(r["swissprot"] or r["trembl"],30)+" | "+clean(r["assay_name"],60)+" | "+clean(source_title,115)+" | "+clean(flags,105)+" |")
lines+=["","## Task-level verification notes",""]
for n,r in enumerate(rows,1):
    id=r["task_id"];pub=ps[id];unip=us[id];clue=cs[id]
    lines+=["### "+str(n)+". "+id[:15]+" - "+clean(r["assay_name"],110),"",
      "- Protein annotation: "+clean(r["target_names"],115)+"; accession "+clean(r["swissprot"] or r["trembl"],55),
      "- Protein sequence reference: "+unip["screen_status"]+" (NOT assayed construct verification).",
      "- Raw assay identity: "+r["assay_key"]+"; candidate compounds: "+r["unique_compounds"]+".",
      "- PubMed IDs: "+(clean(r["pmids"],115) or "none")+"; DOI: "+(clean(r["article_dois"],150) or "none")+"; patent: "+(clean(r["patent_numbers"],100) or "none")+".",
      "- Flags: "+(", ".join(clue["review_clues"]) or "no keyword flags; source still unverified")+".",
      "- Source links:"]
    for m in unip["sequence_reference_matches"]:
        if m.get("reference_url","").startswith("https://rest.uniprot.org/uniprotkb/"):
            lines.append("  - UniProt "+m["accession"]+": "+m["reference_url"])
    for x in pub["indexed_publications"]:
        if x.get("metadata_url","").startswith("https://europepmc.org/article/MED/"):
            lines.append("  - Indexed article: "+clean(x.get("title"),135)+"; "+x["metadata_url"]+"; DOI "+clean(x.get("doi"),100))
    if not pub["indexed_publications"]:lines.append("  - Primary patent or unindexed article must be consulted.")
    lines+=["- To verify: experimental construct sequence and variant; actual exact Kd measurement semantics and qualifiers; consistency of assay conditions; original table/supplementary evidence.","- Reviewer decision: UNREVIEWED.",""]
lines+=["## Review controls","",
  "Enter formal decisions only in data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv.",
  "Run scripts/validate_assay_reviews.py after review. This dossier is not an approved scientific dataset."]
if target.exists():raise FileExistsError("Refusing to overwrite existing dossier")
target.write_text("\n".join(lines)+"\n",encoding="utf-8")
manifest={"status":"SOURCE_DOSSIER_CREATED_NO_APPROVAL","file":str(target),"entries":len(rows),"approved_tasks":0,
          "contains_affinity_values":False}
(audit/"bindingdb_assay_review_30_dossier_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(json.dumps(manifest,indent=2))
