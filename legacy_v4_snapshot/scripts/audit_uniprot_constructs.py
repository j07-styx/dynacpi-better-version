"""Label-blind check of review queue protein sequences against UniProt accessions.

Reference equality is evidence of a sequence match, NOT evidence the assayed
construct was identical. Short domains, mutations, affinity tags and assay
conditions require inspection of the actual publications.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
from Bio.Align import PairwiseAligner
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"data/audit"
QUEUE=AUDIT/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
FASTA=AUDIT/"bindingdb_full_kd_sequences.fasta"
OUT=AUDIT/"bindingdb_assay_review_30_uniprot_construct_screen.json"
CACHE=ROOT/"data/reference/uniprot"
CACHE.mkdir(parents=True,exist_ok=True)

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def parse_fasta():
    table={}
    current=None
    with FASTA.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                current=line[1:].strip()
            elif line.strip():
                if current in table:raise ValueError("Unexpected multifasta sequence record")
                table[current]=line.strip()
    return table

def fetch(accession):
    out=CACHE/(accession+".json")
    if out.exists():
        d=json.loads(out.read_text(encoding="utf-8"))
        if d.get("primaryAccession") and d.get("sequence",{}).get("value"):
            return {"accession":accession,"payload":d,"cache_hit":True}
    url="https://rest.uniprot.org/uniprotkb/"+accession+".json"
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"DynaCPI-v4-construct-audit/0.2 (academic prototype)"})
        with urllib.request.urlopen(req,timeout=22) as r:
            d=json.load(r)
        if not d.get("sequence",{}).get("value"):
            raise ValueError("No reference sequence")
        out.write_text(json.dumps(d,ensure_ascii=False),encoding="utf-8")
        return {"accession":accession,"payload":d,"cache_hit":False}
    except Exception as ex:
        return {"accession":accession,"error":type(ex).__name__+": "+str(ex)[:180]}

def local_stats(query,ref):
    a=PairwiseAligner()
    a.mode="local"
    a.match_score=2.0
    a.mismatch_score=-1.0
    a.open_gap_score=-10.0
    a.extend_gap_score=-0.5
    aligned=a.align(query,ref)[0]
    n=0;matches=0
    for (s1,e1),(s2,e2) in zip(aligned.aligned[0],aligned.aligned[1]):
        if e1-s1!=e2-s2:raise ValueError("Alignment block length mismatch")
        n+=e1-s1
        matches+=sum(x==y for x,y in zip(query[s1:e1],ref[s2:e2]))
    return {"aligned_residues":n,"matched_residues":matches,
            "identity_on_aligned":round(matches/n,5) if n else None,
            "fraction_assay_sequence_aligned":round(n/len(query),5),
            "fraction_reference_sequence_aligned":round(n/len(ref),5)}

def main():
    fasta=parse_fasta()
    with QUEUE.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    if len(rows)!=30:raise ValueError("Unexpected review queue size")
    accession_re=re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9](-[0-9]+)?$|^[OPQ][0-9][A-Z0-9]{3}[0-9](-[0-9]+)?$|^[A-NR-Z][0-9](?:[A-Z0-9]{2}[0-9]){2}(-[0-9]+)?$")
    refs={}
    for r in rows:
        accs=[x.strip() for x in (r["swissprot"]+" | "+r["trembl"]).split("|") if x.strip()]
        r["_accessions"]=sorted(set(accs))
        for x in r["_accessions"]:
            if accession_re.fullmatch(x):refs[x]=None
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,acc):acc for acc in sorted(refs)}
        for fut in as_completed(futures):
            v=fut.result()
            refs[v["accession"]]=v
    counters=Counter()
    output=[]
    for r in rows:
        key=r["protein_sha256"]
        seq=fasta.get(key)
        if seq is None or sha(seq)!=key:
            raise ValueError("Missing/mismatched target sequence "+key)
        entry={"task_id":r["task_id"],"assay_key":r["assay_key"],
               "protein_sha256":key,"assayed_target_sequence_length":len(seq),
               "sequence_reference_matches":[]}
        if not r["_accessions"]:
            counters["no_accession"]+=1
            entry["screen_status"]="NO_UNIPROT_ACCESSION"
        else:
            statuses=[]
            for acc in r["_accessions"]:
                e=refs.get(acc)
                if e is None or "error" in e:
                    entry["sequence_reference_matches"].append({"accession":acc,"status":"REFERENCE_FETCH_FAILED",
                           "reason":e.get("error") if e else "not a parseable accession"})
                    statuses.append("FETCH_FAILED")
                    continue
                v=e["payload"]["sequence"]["value"]
                if seq==v: status="EXACT_REFERENCE_SEQUENCE"
                elif seq in v:status="ASSAY_SEQUENCE_CONTIGUOUS_SUBSEQUENCE"
                elif v in seq:status="REFERENCE_CONTIGUOUS_SUBSEQUENCE"
                else:status="NONEXACT"
                match={"accession":acc,"status":status,
                       "reference_length":len(v),"reference_sha256":sha(v),
                       "reference_primary_accession":e["payload"].get("primaryAccession"),
                       "organism":e["payload"].get("organism",{}).get("scientificName"),
                       "reference_url":"https://rest.uniprot.org/uniprotkb/"+acc+".json"}
                if status=="NONEXACT":
                    match["local_alignment"]=local_stats(seq,v)
                entry["sequence_reference_matches"].append(match)
                statuses.append(status)
            if "EXACT_REFERENCE_SEQUENCE" in statuses:entry["screen_status"]="EXACT_REFERENCE_SEQUENCE"
            elif "ASSAY_SEQUENCE_CONTIGUOUS_SUBSEQUENCE" in statuses:entry["screen_status"]="SUBSEQUENCE_MATCH"
            elif "REFERENCE_CONTIGUOUS_SUBSEQUENCE" in statuses:entry["screen_status"]="REFERENCE_IN_TARGET"
            elif "NONEXACT" in statuses:entry["screen_status"]="NONEXACT_REFERENCE_RELATION"
            else:entry["screen_status"]="NO_REFERENCE_SEQUENCE_FETCHED"
            counters[entry["screen_status"]]+=1
        output.append(entry)
    report={"status":"COMPUTATIONAL_SEQUENCE_SCREEN_ONLY_NO_TASK_APPROVED",
            "checked_at_utc":datetime.now(timezone.utc).isoformat(),
            "references_requested":len(refs),
            "references_fetched":sum(bool(v and "payload" in v) for v in refs.values()),
            "task_count":len(rows),"counts":dict(counters),
            "tasks":output,
            "limitations":[
                "Exact equality to UniProt does not prove the actual assayed construct has the sequence.",
                "Subsequence or locally similar sequence may reflect expected domains, tags or mutations.",
                "Source publication/patent and experimental construct require manual verification.",
                "No affinity labels accessed; do not use statuses as automatic experimental approval.",
                "UniProt sequences may change over time; response payloads cached locally."]}
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("tasks","limitations")},indent=2),flush=True)
if __name__=="__main__":main()
