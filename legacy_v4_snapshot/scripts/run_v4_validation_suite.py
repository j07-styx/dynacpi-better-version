"""Run all implemented DynaCPI v4 engineering/unit checks sequentially.

IMPORTANT: Passing these tests does NOT establish model accuracy or an
approved dataset. No training is initiated by this suite.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
checks=[
 ("adapter_optimization","tests_v4/check_adapter.py"),
 ("episode_data_boundaries","tests_v4/episode_data_real_manifest_test.py"),
 ("assay_review_gate","tests_v4/assay_review_gate_test.py"),
 ("source_access_index_integrity","tests_v4/source_access_index_integrity_test.py"),
 ("construct_alignment_risk_integrity","tests_v4/construct_alignment_risk_integrity_test.py"),
 ("metadata_risk_triage_integrity","tests_v4/metadata_risk_triage_integrity_test.py"),
 ("p0_source_crosswalk_integrity","tests_v4/p0_source_crosswalk_integrity_test.py"),
 ("record_lineage_synthetic","tests_v4/record_lineage_synthetic_test.py"),
 ("record_lineage_real_integrity","tests_v4/record_lineage_real_integrity_test.py"),
 ("source_record_evidence_synthetic","tests_v4/record_source_evidence_synthetic_test.py"),
 ("source_record_evidence_real_integrity","tests_v4/record_source_evidence_real_integrity_test.py"),
 ("geometry_factorial_dev_integrity","tests_v4/geometry_factorial_dev_integrity_test.py"),
 ("six_publication_source_evidence_integrity","tests_v4/six_publication_source_evidence_integrity_test.py"),
 ("open_access_method_evidence_integrity","tests_v4/open_access_method_evidence_integrity_test.py"),
 ("nrp1_primary_table3_numeric_integrity","tests_v4/nrp1_primary_table3_numeric_integrity_test.py"),
 ("protein_homology_challenge_integrity","tests_v4/protein_homology_challenge_integrity_test.py"),
 ("scientific_cohort_gate","tests_v4/scientific_cohort_gate_test.py"),
 ("paired_cluster_statistics_synthetic","tests_v4/paired_evaluation_test.py"),
 ("frozen_adapter_real_chemistry","tests_v4/frozen_adapter_real_chemistry_test.py"),
 ("support_K_sweep_result_integrity","tests_v4/k_sweep_result_integrity_test.py"),
 ("geometry_seed_variability_integrity","tests_v4/geometry_seed_variability_integrity_test.py"),
 ("trained_sequence_checkpoint_interface","tests_v4/trained_sequence_checkpoint_inference_test.py"),
 ("trained_geometry_checkpoint_parity","tests_v4/trained_geometry_checkpoint_parity.py"),
 ("full_real_structure_to_adapter","tests_v4/full_inference_real_3d_synthetic_labels_test.py"),
 ("group_split_dryrun","tests_v4/group_split_dryrun_test.py"),
 ("sequence_synthetic","tests_v4/sequence_smoke.py"),
 ("geometry_real_structure_synthetic_labels","tests_v4/geometry_smoke.py"),
 ("sequence_real_esm_synthetic_labels","tests_v4/cached_sequence_pair_smoke.py"),
 ("sequence_geometry_real_esm_synthetic_labels","tests_v4/cached_esm_gvp_pair_smoke.py"),
 ("archived_davis_checkpoint_parity","tests_v4/legacy_checkpoint_parity.py"),
]
logs=root/"research/validation_suite_logs"
logs.mkdir(parents=True,exist_ok=True)
out=root/"data/audit/v4_validation_suite.json"
report={"scope":"Engineering and provenance checks only: NO DynaCPI training, approved data, OOD scientific evaluation, or research novelty result",
        "started_utc":datetime.now(timezone.utc).isoformat(),
        "tests":[],"status":"IN_PROGRESS"}
def save():
    tmp=out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report,indent=2),encoding="utf-8")
    os.replace(tmp,out)
save()
environment=os.environ.copy()
environment["PYTHONHASHSEED"]="0"
for label,relative in checks:
    target=root/relative
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    started=time.monotonic()
    print(json.dumps({"stage":"start","test":label}),flush=True)
    try:
        proc=subprocess.run([sys.executable,str(target)],cwd=root,env=environment,
                            capture_output=True,text=True,timeout=180,
                            errors="replace")
        log=(proc.stdout or "")+"\n--- STDERR ---\n"+(proc.stderr or "")
        (logs/(label+".log")).write_text(log,encoding="utf-8")
        passed=proc.returncode==0
        result={"test":label,"path":relative,"script_sha256":digest,
                "status":"PASS" if passed else "FAIL",
                "exit_code":proc.returncode,"elapsed_seconds":round(time.monotonic()-started,2),
                "log":str((logs/(label+".log")).relative_to(root)),
                "output_tail":(proc.stdout or "")[-2500:] if passed else log[-4000:]}
    except subprocess.TimeoutExpired as ex:
        result={"test":label,"path":relative,"script_sha256":digest,
                "status":"TIMEOUT","elapsed_seconds":round(time.monotonic()-started,2)}
    report["tests"].append(result)
    save()
    print(json.dumps({"stage":"done","test":label,"status":result["status"],
                      "elapsed_seconds":result["elapsed_seconds"]}),flush=True)
report["finished_utc"]=datetime.now(timezone.utc).isoformat()
report["passed"]=sum(t["status"]=="PASS" for t in report["tests"])
report["failed"]=len(report["tests"])-report["passed"]
report["status"]="PASS_ENGINEERING_ONLY" if report["failed"]==0 else "FAIL_SOME_ENGINEERING_CHECKS"
save()
print(json.dumps({"status":report["status"],"passed":report["passed"],"failed":report["failed"],
                  "report":str(out)}),flush=True)
if report["failed"]:
    raise SystemExit(1)
