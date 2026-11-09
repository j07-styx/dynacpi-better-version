"""Protein-only review30 challenge: no assay labels or split assignments."""
from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
Q=ROOT/"data/audit/bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
F=ROOT/"data/audit/bindingdb_full_kd_sequences.fasta"
D=ROOT/"research/source_access_v4/protein_only_review30_pairwise_audit.json"
r=json.loads(D.read_text(encoding="utf-8"))
assert r["status"]=="UNAPPROVED_30_TARGET_PROTEIN_ONLY_ALIGNMENT_CHALLENGE"
assert r["task_count"]==r["distinct_sequence_count"]==r["distinct_prior_mmseqs_components"]==30
assert r["all_pair_alignments"]==435
assert r["scientifically_approved_tasks"]==0
assert r["frozen_scientific_split_created"] is False
assert r["source_record_review_decisions_changed"] is False
assert r["review_queue_sha256"]==hashlib.sha256(Q.read_bytes()).hexdigest()
assert r["protein_fasta_sha256"]==hashlib.sha256(F.read_bytes()).hexdigest()
rows=list(csv.DictReader(Q.open(encoding="utf-8-sig",newline="")))
assert len(rows)==30 and all(x["curation_decision"]=="UNREVIEWED" for x in rows)
by={x["name"]:x for x in r["summary"]}
assert len(by)==6
weak=by["original_like_30pct_shorter80_min80"]
assert (weak["number_of_direct_pairs"],weak["cross_existing_components"],
        weak["components_in_review_30"])==(2,2,28)
for key in ("medium_40pct_shorter70_min80",
            "strict_50pct_shorter80_min100",
            "high_80pct_shorter90_min80",
            "reciprocal_30pct_shorter80_longer50_min80",
            "reciprocal_40pct_shorter70_longer50_min80"):
 s=by[key]
 assert s["number_of_direct_pairs"]==s["cross_existing_components"]==0
 assert s["components_in_review_30"]==30
assert all(t["name"] in by for t in r["thresholds"])
notables=r["notable_protein_only_local_alignments"]
cross_weak=[v for v in notables if "original_like_30pct_shorter80_min80" in v["thresholds_passed"]]
assert len(cross_weak)==2
assert all(not v["same_existing_mmseqs_component"] for v in cross_weak)
assert all(v["coverage_longer"]<0.20 for v in cross_weak)
assert all(v["identity"]<0.40 for v in cross_weak)
for n in notables:
 assert n["matches"]<=n["aligned_residue_pairs"]
 assert 0<=n["identity"]<=1 and 0<=n["coverage_shorter"]<=1 and 0<=n["coverage_longer"]<=1
assert not (ROOT/"data/audit/bindingdb_frozen_protein_group_split.json").exists()
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"
print(json.dumps({"status":"PASS_PROTEIN_ONLY_REVIEW30_HOMOLOGY_CHALLENGE",
 "pairs":435,"old_components":30,"permissive_cross_group_pairs":2,
 "stricter_reciprocal_pairs":0,"approved_assays":0,
 "scientific_cohort_gate":"BLOCKED"},indent=2))
