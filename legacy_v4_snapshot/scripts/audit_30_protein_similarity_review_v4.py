"""Independent all-pairs protein sequence homology challenge of fixed 30 review targets.

Protein sequences ONLY. No chemical identity, individual affinity value,
source Reactant_set_id, model weights, approved cohort, or split assignment.
The existing MMseqs2 components may miss valid domain-local homologs.
"""
from __future__ import annotations
import csv,hashlib,json,math,statistics,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
from Bio.Align import PairwiseAligner,substitution_matrices
ROOT=Path(__file__).resolve().parents[1]
Q=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
F=ROOT/"data/audit/bindingdb_full_kd_sequences.fasta"
OUT=ROOT/"research/source_access_v4"
THRESHOLDS=[
 {"name":"original_like_30pct_shorter80_min80","identity":0.30,"shorter_coverage":0.80,"min_aligned":80},
 {"name":"medium_40pct_shorter70_min80","identity":0.40,"shorter_coverage":0.70,"min_aligned":80},
 {"name":"strict_50pct_shorter80_min100","identity":0.50,"shorter_coverage":0.80,"min_aligned":100},
 {"name":"high_80pct_shorter90_min80","identity":0.80,"shorter_coverage":0.90,"min_aligned":80},
 {"name":"reciprocal_30pct_shorter80_longer50_min80","identity":0.30,"shorter_coverage":0.80,"longer_coverage":0.50,"min_aligned":80},
 {"name":"reciprocal_40pct_shorter70_longer50_min80","identity":0.40,"shorter_coverage":0.70,"longer_coverage":0.50,"min_aligned":80},
]
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_targets():
 with Q.open(encoding="utf-8-sig",newline="") as f:
  rows=list(csv.DictReader(f))
 assert len(rows)==30 and len({r["task_id"] for r in rows})==30
 assert all(r["curation_decision"]=="UNREVIEWED" for r in rows)
 seqs={}
 ident=None;parts=[]
 with F.open(encoding="utf-8") as f:
  for line in f:
   s=line.strip()
   if not s:continue
   if s.startswith(">"):
    if ident is not None:seqs[ident]="".join(parts)
    ident=s[1:]
    parts=[]
   else:parts.append(s)
 if ident is not None:seqs[ident]="".join(parts)
 assert len(seqs)==4157
 assert all(hashlib.sha256(seq.encode()).hexdigest()==key for key,seq in seqs.items())
 assert all(r["protein_sha256"] in seqs for r in rows)
 assert len({r["protein_sha256"] for r in rows})==30
 return sorted(rows,key=lambda r:r["task_id"]),seqs
def align_metrics(aligner,seq_a,seq_b):
 aln=aligner.align(seq_a,seq_b)[0]
 # aln.aligned gives same-length ungapped matching coordinate blocks;
 # count identity in aligned paired residues only, with gaps excluded.
 pairs=0;matches=0;covered_a=0;covered_b=0
 for (s0,e0),(s1,e1) in zip(aln.aligned[0],aln.aligned[1]):
  assert e0-s0==e1-s1
  size=int(e0-s0)
  pairs+=size
  covered_a+=size;covered_b+=size
  matches+=sum(a==b for a,b in zip(seq_a[s0:e0],seq_b[s1:e1]))
 return {"aligned_residue_pairs":pairs,"matches":matches,
         "identity":matches/pairs if pairs else 0,
         "coverage_shorter":pairs/min(len(seq_a),len(seq_b)),
         "coverage_longer":pairs/max(len(seq_a),len(seq_b)),
         "score":float(aln.score)}
def components(rows,edges):
 parent={r["task_id"]:r["task_id"] for r in rows}
 def find(x):
  while parent[x]!=x:
   parent[x]=parent[parent[x]]
   x=parent[x]
  return x
 for edge in edges:
  a,b=find(edge["task_a"]),find(edge["task_b"])
  if a!=b:parent[max(a,b)]=min(a,b)
 groups=defaultdict(list)
 for r in rows:groups[find(r["task_id"])].append(r["target_names"])
 return sorted([sorted(v) for v in groups.values()],key=lambda x:(-len(x),x[0]))
def run():
 t0=time.monotonic()
 rows,seqs=load_targets()
 aligner=PairwiseAligner()
 aligner.mode="local"
 aligner.substitution_matrix=substitution_matrices.load("BLOSUM62")
 aligner.open_gap_score=-10
 aligner.extend_gap_score=-0.5
 edges=[];possible=[]
 for index,(a,b) in enumerate(combinations(rows,2),1):
  m=align_metrics(aligner,seqs[a["protein_sha256"]],seqs[b["protein_sha256"]])
  edge={"task_a":a["task_id"],"task_b":b["task_id"],
        "name_a":a["target_names"],"name_b":b["target_names"],
        "same_existing_mmseqs_component":a["protein_component_id"]==b["protein_component_id"],
        "len_a":len(seqs[a["protein_sha256"]]),"len_b":len(seqs[b["protein_sha256"]]),**m}
  passed=[t["name"] for t in THRESHOLDS if m["identity"]>=t["identity"] and
          m["coverage_shorter"]>=t["shorter_coverage"] and
          m["coverage_longer"]>=t.get("longer_coverage",0) and
          m["aligned_residue_pairs"]>=t["min_aligned"]]
  edge["thresholds_passed"]=passed
  if passed:edges.append(edge)
  if m["aligned_residue_pairs"]>=60 and m["identity"]>=.25:
   possible.append(edge)
  if index%100==0:print(json.dumps({"stage":"protein_pair_scan","checked":index,"of":435}),flush=True)
 assert index==435
 summaries=[]
 for t in THRESHOLDS:
  valid=[e for e in edges if t["name"] in e["thresholds_passed"]]
  clusters=components(rows,valid)
  summaries.append({"name":t["name"],"number_of_direct_pairs":len(valid),
                    "cross_existing_components":sum(not e["same_existing_mmseqs_component"] for e in valid),
                    "components_in_review_30":len(clusters),
                    "clusters_over_one_target":[c for c in clusters if len(c)>1]})
 # Unverified local matches can be long enough for a domain while failing
 # operational existing full-candidate clustering; log challenge summaries only.
 possible.sort(key=lambda x:(-x["identity"]*x["coverage_shorter"],
                             -x["aligned_residue_pairs"],x["name_a"]))
 notable=[{k:v for k,v in x.items() if k in (
   "name_a","name_b","len_a","len_b","aligned_residue_pairs",
   "matches","identity","coverage_shorter","coverage_longer",
   "same_existing_mmseqs_component","thresholds_passed")} for x in possible[:15]]
 sha_before=filehash(Q)
 report={
  "status":"UNAPPROVED_30_TARGET_PROTEIN_ONLY_ALIGNMENT_CHALLENGE",
  "completed_utc":datetime.now(timezone.utc).isoformat(),
  "method":"Biopython PairwiseAligner, Smith-Waterman local, BLOSUM62, gap open -10, extension -0.5; identical residues/aligned ungapped positions",
  "scope":"Fixed 30 unreviewed human-assay candidate protein sequences only, no compounds or raw affinity labels",
  "review_queue_sha256":sha_before,"protein_fasta_sha256":filehash(F),
  "task_count":len(rows),"distinct_sequence_count":len({r["protein_sha256"] for r in rows}),
  "distinct_prior_mmseqs_components":len({r["protein_component_id"] for r in rows}),
  "all_pair_alignments":435,"thresholds":THRESHOLDS,
  "summary":summaries,"notable_protein_only_local_alignments":notable,
  "source_record_review_decisions_changed":False,"scientifically_approved_tasks":0,
  "frozen_scientific_split_created":False,
  "elapsed_seconds":round(time.monotonic()-t0,2),
  "limits":[
   "Only the 30 selected reference proteins are compared, not all 4157 input proteins.",
   "Local sequence similarity is not proof of shared physical assay constructs or function.",
   "The existing MMseqs connected components were built via full 4157-sequence transitive graph; 30-target direct edges are not equivalent.",
   "Different thresholds are sensitivity analysis, not a validated choice for future independent splits.",
   "Low identities over one small protein segment may be biologically unrelated; an apparent pair is only a lead for independent profile/domain review, not evidence of true family homology.",
   "No main scientific training, test assignments, or BindingDB data review approvals are authorized."
  ]}
 if filehash(Q)!=sha_before:raise PermissionError("Original review decisions changed during protein screen")
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/"protein_only_review30_pairwise_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 lines=["# Independent protein-only homology challenge: 30 fixed source-review targets","",
        "**No drug/compound inputs, no affinity values, and no approved scientific cohorts.**","",
        "BLOSUM62 Smith-Waterman local alignments independently check whether a high-coverage sequence relation exists across targets assigned to different original MMseqs components.",
        "",
        "| Protein-only identity/coverage policy | Direct matching pairs | Cross original MMseqs groups | Review-set connected components |",
        "| --- | ---: | ---: | ---: |"]
 for s in summaries:
  lines.append(f"| {s['name']} | {s['number_of_direct_pairs']} | {s['cross_existing_components']} | {s['components_in_review_30']} |")
 lines+=["","## Notable local protein-only similarities",""]
 for n in notable:
  lines.append(f"- {n['name_a']} / {n['name_b']}: {n['identity']:.1%} identical aligned residues over {n['aligned_residue_pairs']} paired positions, covering {n['coverage_shorter']:.1%} of shorter sequence")
 lines+=["","Original task review status remains 30 UNREVIEWED, no approved science. This analysis is a split-leakage challenge, not an approved evaluation split or biological assay validation."]
 (OUT/"PROTEIN_ONLY_HOMOLOGY_CHALLENGE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print(json.dumps({"status":report["status"],"compared_protein_pairs":435,
                  "prior_components":report["distinct_prior_mmseqs_components"],
                  "threshold_summaries":summaries,"notable_top":notable[:4],
                  "elapsed_seconds":report["elapsed_seconds"]},indent=2),flush=True)
if __name__=="__main__":run()
