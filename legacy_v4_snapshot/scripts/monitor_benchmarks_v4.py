"""DynaCPI v4 local metric monitor — read-only observer of benchmark artifacts.

This command NEVER trains models, tunes hyperparameters, changes curated source
data, approves assays or freezes experimental splits.
It writes only research/monitoring/ snapshots and records changes in existing
result files. Intended to be run from the VS Code task or hourly automation.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.scientific_cohort_gate import current_status

MON=ROOT/"research/monitoring"
MON.mkdir(parents=True,exist_ok=True)
OUT=MON/"latest.json"
MD=MON/"LATEST.md"
EVENTS=MON/"events.jsonl"

def fetch(relative):
    path=ROOT/relative
    if not path.exists():return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError,UnicodeDecodeError):
        return {"status":"BROKEN_JSON","path":relative}

def sha(path):
    if not path.exists():return None
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def gpu():
    try:
        p=subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=8)
        if p.returncode:
            return {"status":"GPU_STATUS_UNAVAILABLE","reason":p.stderr[-150:]}
        return {"status":"OK","snapshot":p.stdout.strip()}
    except (subprocess.SubprocessError,OSError) as ex:
        return {"status":"GPU_STATUS_UNAVAILABLE","reason":type(ex).__name__}

def read_status():
    suite=fetch("data/audit/v4_validation_suite.json")
    sweep=fetch("research/support_k_sweep_dev/results.json")
    seq=fetch("research/pilot_sequence/history.json")
    geom=fetch("research/pilot_geometry_9train_4val/history.json")
    geomcheck=fetch("research/pilot_geometry_9train_4val/checkpoint_parity.json")
    source_index=fetch("research/source_access_v4/source_access_index.json")
    p0_evidence=fetch("research/source_access_v4/p0_source_crosswalk.json")
    lineage=fetch("research/source_access_v4/record_level_provenance_30.json")
    evidence_readiness=fetch("research/source_access_v4/record_source_evidence_readiness_30.json")
    six_source_leads=fetch("research/source_access_v4/six_task_publication_evidence.json")
    protein_homology=fetch("research/source_access_v4/protein_only_review30_pairwise_audit.json")
    triage=fetch("research/source_access_v4/assay_metadata_risk_triage_30.json")
    construct_risk=fetch("research/source_access_v4/construct_alignment_risk_screen.json")
    manual_review=fetch("data/audit/bindingdb_assay_manual_review_status.json")
    seed_variability=fetch("research/pilot_geometry_seed_sweep_dev/seed_variability.json")
    seed_comparison=fetch("research/pilot_geometry_seed_sweep_dev/adapter_seed_comparison.json")
    geom_factorial=fetch("research/geometry_factorial_dev/factorial.json")
    source_gate=current_status(ROOT)
    report={
     "scope":"ENGINEERING_REPRODUCIBILITY_MONITOR; NO SCIENTIFIC BINDINGDB BENCHMARK",
     "observed_utc":datetime.now(timezone.utc).isoformat(),
     "project":str(ROOT),
     "gpu":gpu(),
     "cohort_gate":source_gate,
     "validation_suite":{"status":suite.get("status"),"passed":suite.get("passed"),"failed":suite.get("failed"),
                         "finished_utc":suite.get("finished_utc")} if isinstance(suite,dict) else {"status":"MISSING"},
     "sequence_engineering_pilot":{
         "epoch_count":len(seq),
         "best_validation_target_macro_rmse":min((r["target_macro_rmse"] for r in seq),default=None),
         "epoch_of_best":min(seq,key=lambda r:r["target_macro_rmse"])["epoch"] if seq else None,
         "saved_checkpoint_sha256":sha(ROOT/"research/pilot_sequence/checkpoint.pt")
     } if isinstance(seq,list) else {"status":"MISSING"},
     "geometry_engineering_pilot":{
        "epoch_count":len(geom),
        "best_geometry_validation_target_macro_rmse":min((r["validation_geometry_target_macro_rmse"] for r in geom),default=None),
        "frozen_sequence_validation_target_macro_rmse":geom[0]["validation_sequence_target_macro_rmse"] if geom else None,
        "saved_checkpoint_sha256":sha(ROOT/"research/pilot_geometry_9train_4val/best_checkpoint.pt"),
        "checkpoint_reload_status":geomcheck.get("status") if isinstance(geomcheck,dict) else "MISSING",
     } if isinstance(geom,list) else {"status":"MISSING"},
     "support_K_development":{
        "status":sweep.get("status"),
        "results":[{"K":r["K"],"B_macro_RMSE":r["B_macro_RMSE"],"D_macro_RMSE":r["D_macro_RMSE"],
                    "D_minus_B_macro_RMSE":r["D_minus_B_macro_RMSE"],"mean_W":r["mean_W"],
                    "mean_V":r["mean_V"]}
                   for r in sweep.get("macro_by_K",[])],
        "artifact_sha256":sha(ROOT/"research/support_k_sweep_dev/results.json")
     } if isinstance(sweep,dict) else {"status":"NOT_RUN"},
     "geometry_seed_variability_dev":{
         "status":seed_variability.get("status"),
         "n_seeds":seed_variability.get("n_seeds"),
         "n_better_than_sequence":seed_variability.get("n_better_than_sequence"),
         "n_worse_than_sequence":seed_variability.get("n_worse_than_sequence"),
         "seed_results":[{"seed":r["seed"],"best_epoch":r["best_epoch"],
                         "geometry_minus_sequence_rmse":r["geometry_minus_sequence_rmse"]}
                        for r in seed_variability.get("seeds",[])]
     } if isinstance(seed_variability,dict) else {"status":"NOT_RUN"},
     "geometry_factorial_dev":{
       "status":geom_factorial.get("status"),
       "n_targets":geom_factorial.get("independent_dev_protein_count"),
       "n_correlated_target_draw_K_trials":geom_factorial.get("total_target_draw_K_trials"),
       "prior_benchmark_parity_max_rmse":geom_factorial.get("previous_K_sweep_agreement_max_abs_RMSE"),
       "K5_by_seed":[{"seed":r["seed"],
           "sequence_calibrated_RMSE":r["S_B_macro_RMSE"],
           "geometry_calibrated_RMSE":r["G_B_macro_RMSE"],
           "geometry_full_RMSE":r["G_D_macro_RMSE"],
           "geometry_main_effect_RMSE":r["G_B_minus_S_B_RMSE_macro"],
           "differential_effect_RMSE":r["G_D_minus_G_B_RMSE_macro"]}
        for r in geom_factorial.get("macro_by_seed_K",[]) if r["K"]==5]
     } if isinstance(geom_factorial,dict) else {"status":"NOT_RUN"},
     "geometry_adapter_seed_comparison_dev":{
         "status":seed_comparison.get("status"),
         "maximum_abs_D_minus_B":seed_comparison.get("D_minus_B_max_abs_selected"),
         "K20_by_seed":[{"seed":r["seed"],"B_RMSE":r["B_RMSE"],"D_RMSE":r["D_RMSE"],
                         "D_minus_B_RMSE":r["D_minus_B_RMSE"]}
                        for r in seed_comparison.get("table_selected_K",[]) if r["K"]==20]
     } if isinstance(seed_comparison,dict) else {"status":"NOT_RUN"},
     "protein_only_review30_homology_challenge":{
       "status":protein_homology.get("status"),
       "protein_count":protein_homology.get("task_count"),
       "independent_local_pair_checks":protein_homology.get("all_pair_alignments"),
       "original_mmseqs_components":protein_homology.get("distinct_prior_mmseqs_components"),
       "sensitivity":[{"policy":s["name"],"direct_pairs":s["number_of_direct_pairs"],"cross_mmseqs_components":s["cross_existing_components"]} for s in protein_homology.get("summary",[])],
       "scientifically_approved_tasks":protein_homology.get("scientifically_approved_tasks")
     } if isinstance(protein_homology,dict) else {"status":"NOT_RUN"},
     "six_publication_source_leads":{
       "status":six_source_leads.get("status"),
       "task_count":six_source_leads.get("target_task_count"),
       "candidate_record_count":six_source_leads.get("candidate_record_count"),
       "primary_spr_lead_count":six_source_leads.get("priority_spr_assay",{}).get("records"),
       "review_citation_not_original_count":six_source_leads.get("known_review_citation_not_original_experiment"),
       "scientifically_approved":six_source_leads.get("source_approved_task_count")
     } if isinstance(six_source_leads,dict) else {"status":"NOT_RUN"},
     "source_record_evidence_readiness":{
       "status":evidence_readiness.get("status"),
       "total_reactant_records":evidence_readiness.get("source_record_count"),
       "self_reported_review_states":evidence_readiness.get("self_reported_review_state_counts"),
       "task_count":evidence_readiness.get("task_count"),
       "scientifically_approved":evidence_readiness.get("source_scientifically_approved_task_count"),
       "training_authorized":evidence_readiness.get("main_scientific_training_authorized")
     } if isinstance(evidence_readiness,dict) else {"status":"NOT_RUN"},
     "raw_record_lineage_audit":{
        "status":lineage.get("status"),
        "task_count":lineage.get("tasks"),
        "compound_group_count":lineage.get("compound_groups"),
        "reactant_count":lineage.get("source_reactant_references"),
        "source_records_found":lineage.get("original_reactant_ids_found"),
        "candidate_records_found":lineage.get("unaggregated_reactant_ids_found"),
        "critical_lineage_flags":lineage.get("critical_lineage_flags"),
        "scientifically_approved_tasks":lineage.get("approved_tasks"),
        "source_assay_methods_verified":lineage.get("source_biological_review_completed")
     } if isinstance(lineage,dict) else {"status":"NOT_RUN"},
     "p0_source_review_evidence":{
        "status":p0_evidence.get("status"),
        "source_identified_tasks":p0_evidence.get("task_count"),
        "approved_count":p0_evidence.get("approved_count"),
        "record_level_affinity_verified":sum(int(bool(x.get("each_numeric_Kd_measurement_verified"))) for x in p0_evidence.get("records",[])),
        "source_tiers":[x.get("observation_source_tier") for x in p0_evidence.get("records",[])]
     } if isinstance(p0_evidence,dict) else {"status":"NOT_RUN"},
     "metadata_review_triage":{
       "status":triage.get("status"),
       "n_candidate_tasks":triage.get("task_count"),
       "n_scientifically_approved":triage.get("approved_tasks"),
       "review_priority_counts":triage.get("priority_counts"),
       "metadata_flag_counts":triage.get("risk_flag_counts")
     } if isinstance(triage,dict) else {"status":"NOT_RUN"},
     "source_curation_readiness":{
       "status":source_index.get("status"),
       "candidate_tasks":source_index.get("task_count"),
       "scientifically_approved_tasks":source_index.get("scientifically_approved_tasks"),
       "source_status_counts":source_index.get("source_status_counts"),
       "publication_access_counts":source_index.get("publication_access_counts"),
       "human_review_unreviewed":manual_review.get("counts",{}).get("UNREVIEWED") if isinstance(manual_review,dict) else None,
       "human_review_approved":manual_review.get("approved_tasks") if isinstance(manual_review,dict) else None,
       "construct_region_alignment_flags":construct_risk.get("reconciled_task_count") if isinstance(construct_risk,dict) else None,
       "construct_risk_reviewed_approvals":construct_risk.get("scientifically_approved_task_count") if isinstance(construct_risk,dict) else None
     } if isinstance(source_index,dict) else {"status":"NOT_RUN"},
     "watch_artifacts":{
         str(rel):sha(ROOT/rel) for rel in [
             "research/pilot_sequence/history.json",
             "research/pilot_geometry_9train_4val/history.json",
             "data/audit/v4_validation_suite.json",
             "research/support_k_sweep_dev/results.json",
             "research/pilot_geometry_seed_sweep_dev/seed_variability.json",
             "research/pilot_geometry_seed_sweep_dev/adapter_seed_comparison.json",
             "research/geometry_factorial_dev/factorial.json",
             "research/support_k_sweep_seed_dev/seed_1106/results.json",
             "research/support_k_sweep_seed_dev/seed_1107/results.json",
             "data/audit/bindingdb_assay_manual_review_status.json",
             "research/source_access_v4/source_access_index.json",
             "research/source_access_v4/construct_alignment_risk_screen.json",
             "research/source_access_v4/assay_metadata_risk_triage_30.json",
             "research/source_access_v4/p0_source_crosswalk.json",
             "research/source_access_v4/record_level_provenance_30.json",
             "research/source_access_v4/record_source_evidence_readiness_30.json",
             "research/source_access_v4/six_task_publication_evidence.json",
             "research/source_access_v4/protein_only_review30_pairwise_audit.json",
             "research/source_access_v4/P0_SOURCE_CROSSWALK.md",
             "research/source_access_v4/PRELIMINARY_CONSTRUCT_ALIGNMENT_REVIEW.md",
             "research/source_access_v4/worklist_manifest.json",
             "data/audit/bindingdb_full_kd_candidate_episode_manifest_validation.json"
         ]},
     "limitations":[
         "Four-target DAVIS support sweep reuses checkpoint-selected development targets and reporting-boundary labels.",
         "All true BindingDB exact-Kd research results remain blocked on approved constructs and frozen protein-group split.",
         "This is a read-only local artifact monitor; a missing process output is not evidence of successful training.",
         "GPU usage is a snapshot; it does not establish an active DynaCPI compute task."
     ]}
    return report

def main():
    old=None
    if OUT.exists():
        try:old=json.loads(OUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:pass
    report=read_status()
    changed=[]
    prior=old.get("watch_artifacts",{}) if isinstance(old,dict) else {}
    for k,h in report["watch_artifacts"].items():
        if prior.get(k)!=h:changed.append(k)
    if isinstance(old,dict) and old.get("cohort_gate")!=report["cohort_gate"]:
        changed.append("cohort_gate_state")
    report["change_since_previous_monitor"]=changed
    OUT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    with MD.open("w",encoding="utf-8") as f:
        f.write("# DynaCPI v4 — local benchmark monitor\n\n")
        f.write("Updated (UTC): "+report["observed_utc"]+"\n\n")
        f.write("**Status:** ENGINEERING ONLY. No scientifically approved BindingDB cohort or frozen protein-group split.\n\n")
        f.write("**GPU:** "+report["gpu"].get("snapshot",report["gpu"]["status"])+"\n\n")
        vs=report["validation_suite"]
        f.write("**Last consolidated regression check:** "+str(vs.get("passed"))+"/"+str((vs.get("passed") or 0)+(vs.get("failed") or 0))+"; status "+str(vs.get("status"))+"\n\n")
        f.write("**Scientific gate:** "+str(report["cohort_gate"].get("status"))+"\n\n")
        cur=report["source_curation_readiness"]
        if cur.get("candidate_tasks") is not None:
            f.write("**Scientific source review:** "+str(cur.get("human_review_unreviewed"))+" tasks unreviewed; "+str(cur.get("human_review_approved"))+" approved; "+str(cur.get("candidate_tasks"))+" access records indexed (metadata only).\n\n")
            f.write("**Construct-to-reference alignment flags:** "+str(cur.get("construct_region_alignment_flags"))+" source-located tasks needing physical construct verification; "+str(cur.get("construct_risk_reviewed_approvals"))+" approved by this check.\n\n")
            paper=report["six_publication_source_leads"]
            if paper.get("task_count") is not None:
                f.write("**Publication source review leads:** "+str(paper.get("task_count"))+" tasks / "+str(paper.get("candidate_record_count"))+" preliminary records, "+str(paper.get("review_citation_not_original_count"))+" citation provenance discrepancy, "+str(paper.get("scientifically_approved"))+" independently approved. Bibliographic evidence only.\n\n")
            ev=report["source_record_evidence_readiness"]
            if ev.get("total_reactant_records") is not None:
                f.write("**Local source-evidence review worksheet (private):** "+str(ev.get("total_reactant_records"))+" source records; states "+str(ev.get("self_reported_review_states"))+"; "+str(ev.get("scientifically_approved"))+" scientifically approved. Worksheet is NOT committed.\n\n")
            lin=report["raw_record_lineage_audit"]
            if lin.get("reactant_count") is not None:
                f.write("**Raw source lineage (technical only):** "+str(lin.get("source_records_found"))+"/"+str(lin.get("reactant_count"))+" source reactants and "+str(lin.get("candidate_records_found"))+" preliminary candidate records reconciled; "+str(len(lin.get("critical_lineage_flags") or {}))+" critical issue types; zero assay source approvals.\n\n")
            p0=report["p0_source_review_evidence"]
            if p0.get("source_identified_tasks") is not None:
                f.write("**P0 external source leads:** "+str(p0.get("source_identified_tasks"))+" identified; "+str(p0.get("approved_count"))+" approved; "+str(p0.get("record_level_affinity_verified"))+" fully verified record-level Kd task(s).\n\n")
            tr=report["metadata_review_triage"]
            if tr.get("review_priority_counts"):
                f.write("**30-task metadata triage (not approval):** "+", ".join(str(k)+"="+str(v) for k,v in sorted(tr["review_priority_counts"].items()))+".\n\n")
        sk=report["support_K_development"].get("results",[])
        if sk:
            f.write("| Support K | Calibration B RMSE | Full D RMSE | D−B RMSE | Mean kernel W | Mean disagreement V |\n")
            f.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for r in sk:
                f.write("| "+str(r["K"])+" | "+" | ".join(f"{r[v]:.8g}" for v in ("B_macro_RMSE","D_macro_RMSE","D_minus_B_macro_RMSE","mean_W","mean_V"))+" |\n")
        f.write("\nThis is a same-four-target development diagnostic and does not show independent cold-family improvement.\n")
        gp=report["geometry_seed_variability_dev"]
        if gp.get("seed_results"):
            f.write("\n## Geometry training seed variability (DEV only)\n\n")
            f.write("| Seed | Best epoch | Geometry minus sequence RMSE |\n| ---: | ---: | ---: |\n")
            for x in gp["seed_results"]:
                f.write(f"| {x['seed']} | {x['best_epoch']} | {x['geometry_minus_sequence_rmse']:+.6f} |\n")
        sc=report["geometry_adapter_seed_comparison_dev"]
        if sc.get("K20_by_seed"):
            f.write("\n## K20 B-versus-D by geometry seed (DEV only)\n\n")
            f.write("| Seed | B RMSE | D RMSE | D−B |\n| ---: | ---: | ---: | ---: |\n")
            for x in sc["K20_by_seed"]:
                f.write(f"| {x['seed']} | {x['B_RMSE']:.7f} | {x['D_RMSE']:.7f} | {x['D_minus_B_RMSE']:.3g} |\n")
        fac=report["geometry_factorial_dev"]
        if fac.get("K5_by_seed"):
            f.write("\n## Geometry main effect vs differential adapter (K=5, DEV only)\n\n")
            f.write("| Seed | Sequence calibrated | Geometry calibrated | Full geometry | Geometry minus sequence | Differential minus geometry |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for x in fac["K5_by_seed"]:
                f.write("| "+str(x["seed"])+" | "+f"{x['sequence_calibrated_RMSE']:.6f}"+" | "+f"{x['geometry_calibrated_RMSE']:.6f}"+" | "+f"{x['geometry_full_RMSE']:.6f}"+" | "+f"{x['geometry_main_effect_RMSE']:+.6f}"+" | "+f"{x['differential_effect_RMSE']:+.3g}"+" |\n")
            f.write("\nSame four checkpoint-selected DAVIS proteins; three seeds are not three independent benchmarks.\n")
        f.write("\nChanged watched artifacts since prior run: "+(", ".join(changed) if changed else "none")+"\n")
    if changed:
        with EVENTS.open("a",encoding="utf-8") as f:
            f.write(json.dumps({"observed_utc":report["observed_utc"],"changed":changed,
                                "suite":report["validation_suite"],
                                "scientific_gate":report["cohort_gate"].get("status")})+"\n")
    print(json.dumps({"timestamp_utc":report["observed_utc"],
      "suite":report["validation_suite"],"gate":report["cohort_gate"].get("status"),
      "support_K":report["support_K_development"].get("results",[]),
      "changed_files":changed,"report_path":str(OUT),"gpu":report["gpu"]},indent=2),flush=True)

if __name__=="__main__":main()
