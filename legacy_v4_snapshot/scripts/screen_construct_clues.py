"""Text-only red flags in assay descriptions for manual construct verification.

An exact UniProt sequence in metadata DOES NOT establish that the experimental
protein preparation was identical; the text can mention truncations, tagged
domains, complexes or engineered variants. All results require adjudication.
"""
from collections import Counter
from pathlib import Path
import csv
import json
import re
root=Path(__file__).resolve().parents[1]
audit=root/"data/audit"
src=audit/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
uniprot=json.loads((audit/"bindingdb_assay_review_30_uniprot_construct_screen.json").read_text(encoding="utf-8"))
matched={r["task_id"]:r for r in uniprot["tasks"]}
patterns={
"EXPLICIT_RESIDUE_RANGE":r"\b(?:residues?|amino acids?|aa)\s*\(?\s*\d{1,4}\s*(?:-|–|to)\s*\d{1,4}",
"CONSTRUCT_DOMAIN":r"\b(?:kinase domain|binding domain|catalytic domain|protein domain|extracellular domain|cytoplasmic domain|N[- ]terminal domain|C[- ]terminal domain)\b",
"ENGINEERED_TAG":r"\b(?:His[- ]?6|6[- ]?His|His[- ]tag|FLAG[- ]tag|GST[- ]fusion|recombinant protein|tagged protein)\b",
"MUTATION_LANGUAGE":r"\b(?:mutant|mutation|mutated|variant|substitution|alanine substitution|wild[- ]type|deletion mutant|truncated)\b",
"COMPLEX_OR_COFAC":r"\b(?:complexed with|protein complex|heterodimer|heterotrimer|complex of|elongin|heteromer|cofactor|co-receptor)\b",
"INHIBITION_OR_COMPETITION":r"\b(?:IC50|inhibition assay|inhibition constant|competitive binding|competition assay|displacement assay)\b",
"NOT_FOUND_DESCRIPTION":r"^\s*(?:not found|n/?a|unknown|unavailable)\s*$",
}
findings=[];counter=Counter()
with src.open(encoding="utf-8-sig",newline="") as f:
    for row in csv.DictReader(f):
        desc=row["assay_name"]+" "+row["assay_description"]
        flags={name:bool(re.search(p,desc,flags=re.IGNORECASE)) for name,p in patterns.items()}
        hit=[key for key,is_hit in flags.items() if is_hit]
        if hit:counter["tasks_with_textual_review_clues"]+=1
        for h in hit:counter[h]+=1
        if matched[row["task_id"]]["screen_status"]!="EXACT_REFERENCE_SEQUENCE":
            raise ValueError("Unexpected input sequence audit result")
        findings.append({"task_id":row["task_id"],"assay_key":row["assay_key"],
                         "uniprot_reference_metadata_exact":True,
                         "review_clues":hit,
                         "needs_experimental_construct_review":True,
                         "auto_assay_approval":False})
if len(findings)!=30:raise ValueError("Unexpected review sample size")
report={"status":"ASSAY_TEXT_CONSTRUCT_SCREEN_ONLY","scope":"30 blinded review tasks",
        "counts":dict(counter),"tasks":findings,
        "note":"Exact metadata-to-UniProt identity is compatible with an experimentally truncated or modified construct; none is automatically approved.",
        "limitations":["Regular-expression flags may miss synonyms or trigger on unrelated wording.",
                       "Full source/patent must be examined for actual construct and endpoint semantics.",
                       "No affinity values accessed."]}
dest=audit/"bindingdb_assay_review_30_construct_clues.json"
dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ("tasks","limitations")},indent=2),flush=True)
