"""Cross-check two source-method construct flags against immutable source data.

Engineering metadata check only: source publication methods have been inspected
by the reviewer, but this automated check does not itself authenticate the
physical construct or approve ANY experimental Kd record.
"""
from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"data/audit"
R=ROOT/"research/source_access_v4"
review=A/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
source=review.read_bytes()
queue={r["task_id"]:r for r in csv.DictReader(review.open(encoding="utf-8-sig",newline=""))}
refs={r["task_id"]:r for r in json.loads((A/"bindingdb_assay_review_30_uniprot_construct_screen.json").read_text(encoding="utf-8"))["tasks"]}
index=json.loads((R/"source_access_index.json").read_text(encoding="utf-8"))
pubs={p["pmid"]:p for p in index["publications"]}
specs=[
 ("69f76b68cfe450d9847cbcf66f2aacfaf9059b740d9266ab6f68052affd644a8","P17342","35333039","PMC9014859",27,541,"sec4.4"),
 ("83775558aead8993bb3100cbac91d08f9d2879c15bad712388ba49dd380c0a59","O14786","29648813","PMC5957473",273,427,"sec4.2.1")
]
if len(queue)!=30 or index["review_file_sha256"]!=hashlib.sha256(source).hexdigest():
    raise PermissionError("Source review queue identity or hash changed")
if any(row["curation_decision"]!="UNREVIEWED" for row in queue.values()):
    raise PermissionError("Manual source review queue has been adjudicated")
out=[]
for task,acc,pmid,pmcid,start,end,sec in specs:
    row=queue[task]
    assert pmid in row["pmids"]
    assert pubs[pmid]["pmcid"]==pmcid and pubs[pmid]["metadata_access"]=="EXACT_PMID_MATCH"
    assert pubs[pmid]["fulltext_access"]=="OPEN_ACCESS_SECTION_INDEXED"
    ref=refs[task]
    matched=[x for x in ref["sequence_reference_matches"] if x["accession"]==acc and x["status"]=="EXACT_REFERENCE_SEQUENCE"]
    assert len(matched)==1
    full=int(ref["assayed_target_sequence_length"])
    assert full==int(matched[0]["reference_length"])
    assert 1<=start<=end<=full and end-start+1<full
    out.append({"task_id":task,"protein_accession":acc,"pmid":pmid,"pmcid":pmcid,
                "source_methods_section":sec,
                "candidate_full_reference_length":full,
                "paper_reported_construct_region":[start,end],
                "nominal_fragment_length":end-start+1,
                "reference_vs_reported_region_differs":True,
                "verified_physical_construct_sequence":False,
                "individual_affinity_values_reviewed":False,
                "source_assay_compatibility_verified":False,
                "human_review_status":"UNREVIEWED",
                "scientific_approval":False})
assert len(out)==2
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status
assert current_status(ROOT)["status"]=="BLOCKED_UNAPPROVED_NOT_TRAINING_READY"
if review.read_bytes()!=source:raise PermissionError("Human review sheet changed")
report={"status":"PASS_SOURCE_LOCATOR_CONSTRUCT_RISK_ONLY_NOT_APPROVED",
        "canonical_review_sha256":hashlib.sha256(source).hexdigest(),
        "reconciled_task_count":len(out),"full_30_task_review_queue_unchanged":True,
        "scientifically_approved_task_count":0,"cohort_training_gate":"BLOCKED",
        "source_section_locator_checks_not_full_text_authentication":True,
        "records":out}
(R/"construct_alignment_risk_screen.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({"status":report["status"],"n":len(out),"targets":[(x["protein_accession"],x["candidate_full_reference_length"],x["nominal_fragment_length"]) for x in out],"approved":0},indent=2))
