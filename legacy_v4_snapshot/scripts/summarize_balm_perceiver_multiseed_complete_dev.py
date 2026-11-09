"""Freeze a reproducible matched-dev audit of all finished BALM seed runs.

No neural retraining and no reserved test access. Results already inspected,
hence inferential intervals are descriptive and NOT independent evidence.
"""
import hashlib,json,statistics
from pathlib import Path
import numpy as np

from dynacpi.balm_public_data_dev import ROOT,get_train_dev
from scripts.compare_balm_perceiver_vs_ensemble_dev import row_hash

BASE=ROOT/"research"
SOURCE=BASE/"balm_perceiver_multiseed_dev_20261108"
OUT=BASE/"balm_matched_multiseed_complete_audit_20261108.json"
REPORT=BASE/"BALM_COMPLETE_MULTI_SEED_PERCEIVER_VS_MEMORY_20261108.md"
LOCKED=BASE/"balm_ensemble_dev_predictions_20261108.json"
MEMORY=BASE/"balm_memory_vs_perceiver_matched_innerdev_20261108.json"
EXPECTED={10:[2103,2601,2602,2603,2604,2605],30:[2601,2602]}

def run_metadata(epochs):
    reports=[json.loads(path.read_text(encoding="utf8"))
             for path in sorted(SOURCE.glob(f"perceiver_seed*_ep{epochs}.json"))]
    reports.sort(key=lambda r:r["seed"])
    assert [r["seed"] for r in reports]==EXPECTED[epochs]
    return reports

def parse():
    if OUT.exists() or REPORT.exists():raise FileExistsError("Report already exists")
    dev=get_train_dev()["inner_dev"]
    assert len(dev)==3891 and len(set(r["group"] for r in dev))==173
    sha=row_hash(dev)
    ens=json.loads(LOCKED.read_text(encoding="utf8"))
    assert ens["dev_row_sha256"]==sha
    mem=json.loads(MEMORY.read_text(encoding="utf8"))
    assert mem["dev_row_sha256"]==sha and mem["no_reserved_test_data_access"]
    deep=json.loads((BASE/"balm_bindingdb_deepdta_10epoch_dev.json").read_text())
    assert deep["validation_rows"]==3891

    summary={}
    all_runs={}
    for epochs in (10,30):
        rset=run_metadata(epochs)
        all_runs[epochs]=rset
        detail=[]
        for r in rset:
            assert r["dev_rows"]==3891 and r["dev_row_sha256"]==sha
            assert len(r["history"])==epochs and len(r["dev_predictions_by_epoch"])==epochs
            assert all(len(row)==3891 for row in r["dev_predictions_by_epoch"])
            best=min(r["history"],key=lambda x:x["pooled_dev_rmse"])
            final=r["history"][-1]
            detail.append({
                "seed":r["seed"],
                "epochs":epochs,
                "final_pooled_rmse":final["pooled_dev_rmse"],
                "final_macro_rmse":final["macro_protein_group_dev_rmse"],
                "final_train_mse_normalized":final["train_mse_normalized"],
                "best_dev_pooled_epoch":best["epoch"],
                "best_dev_pooled_rmse":best["pooled_dev_rmse"],
                "total_epoch_elapsed_seconds":round(sum(x["seconds"] for x in r["history"]),2),
                "final_minus_best_pooled":final["pooled_dev_rmse"]-best["pooled_dev_rmse"],
            })
        pooled=[d["final_pooled_rmse"] for d in detail]
        macro=[d["final_macro_rmse"] for d in detail]
        matched_name=("balm_perceiver_vs_ensemble_paired_dev_20261108.json"
                      if epochs==10 else
                      "balm_perceiver_vs_ensemble_paired_dev_20261108_ep30.json")
        matched=json.loads((BASE/matched_name).read_text())
        assert matched["seeds"]==EXPECTED[epochs] and matched["dev_rows"]==3891
        assert matched["dev_groups"]==173 and matched["bootstrap_draws"]==5000
        selected="ridge_interaction_blend_0.25"
        comparison=matched["comparisons"][f"{selected}_vs_perceiver_FINAL_EPOCH_PRIMARY"]
        assert abs(comparison["perceiver_mean_over_seeds"]["pooled"]-statistics.mean(pooled))<1e-5
        assert abs(comparison["perceiver_mean_over_seeds"]["macro"]-statistics.mean(macro))<1e-5
        assert comparison["ci95_pooled"][1]<0 and comparison["ci95_macro"][1]<0
        summary[str(epochs)]={
            "num_runs":len(rset),
            "seeds":EXPECTED[epochs],
            "per_seed":detail,
            "final_epoch_seedmean_pooled":float(statistics.mean(pooled)),
            "final_epoch_seed_sd_pooled":float(statistics.stdev(pooled)),
            "final_epoch_seedmean_macro":float(statistics.mean(macro)),
            "final_epoch_seed_sd_macro":float(statistics.stdev(macro)),
            "locked_ensemble_vs_perceiver_primary":comparison,
            "locked_ensemble_vs_optimistic_dev_best":matched["comparisons"][
                f"{selected}_vs_perceiver_BEST_DEV_EPOCH_OPTIMISTIC"],
            "locked_ensemble_vs_seed_averaged_predictions":matched["comparisons"][
                f"{selected}_vs_perceiver_SEED_AVERAGED_PREDICTIONS_FINAL"],
            "development_winning_rule_satisfied":True,
        }
    repeated_training_consistency={}
    for seed in (2601,2602):
        r10=next(r for r in all_runs[10] if r["seed"]==seed)
        r30=next(r for r in all_runs[30] if r["seed"]==seed)
        dif=float(np.max(np.abs(np.asarray(r10["dev_predictions_by_epoch"])-
                                 np.asarray(r30["dev_predictions_by_epoch"][:10]))))
        assert dif==0.
        repeated_training_consistency[str(seed)]={"identical_first_ten_epoch_predictions":True,
                                                  "max_abs_difference":dif}
    deep_final=deep["history"][-1]
    deep_best=min(deep["history"],key=lambda r:r["pooled_dev_rmse"])
    memory_metrics=mem["model_metrics"]
    locked_metrics=memory_metrics["locked_explicit_interaction_ensemble"]
    p=mem["paired_group_bootstrap"]
    for epoch in (10,30):
        x=p[f"memory_joint_esm8_k64_minus_perceiver_{epoch}ep_final_seedmean"]
        assert x["ci95_pooled"][1]<0 and x["ci95_macro"][1]<0
    m=p["memory_joint_esm8_k64_minus_locked_explicit_interaction_ensemble"]
    assert m["ci95_pooled"][0]<0<m["ci95_pooled"][1]
    assert m["ci95_macro"][0]<0<m["ci95_macro"][1]
    output={
        "status":"BALM_MATCHED_INNER_DEV_COMPLETE_SEED_AUDIT_NO_PROSPECTIVE_TEST",
        "cohort":"3,891 repeatedly inspected BALM original inner_dev rows, 173 original 3-mer protein graph groups",
        "row_sha256":sha,
        "locked_classic_ensemble":"ridge_interaction_blend_0.25",
        "locked_classic_metrics":locked_metrics,
        "perceiver_model":"Upstream PerceiverCPI InteractionModel unchanged, fixed AdamW 0.0003, 500 protein tokens, batch 32, supervised fit on identical 12,315 unique TRAIN protein+SMILES inputs",
        "seed_budgets":summary,
        "first_ten_epoch_reproducibility":repeated_training_consistency,
        "deepdta_inspired_control":{
            "source":"balm_bindingdb_deepdta_10epoch_dev.json",
            "seed":2103,
            "final_epoch":deep_final["epoch"],
            "final_epoch_pooled":deep_final["pooled_dev_rmse"],
            "final_epoch_macro":deep_final["macro_group_dev_rmse"],
            "optimistic_best_dev_epoch":deep_best["epoch"],
            "optimistic_best_dev_pooled":deep_best["pooled_dev_rmse"],
            "note":"DeepDTA best DEV epoch selected with DEV labels, so not a fair locked training-only selection; lacks paired predictions for significance.",
        },
        "memory_model_dev_metrics":memory_metrics,
        "memory_model_vs_classic":p["memory_joint_esm8_k64_minus_locked_explicit_interaction_ensemble"],
        "memory_protein_aware_vs_chemical_only":p["memory_joint_esm8_k64_minus_memory_chemical_only_k64"],
        "memory_model_vs_perceiver":{
            str(epoch):p[f"memory_joint_esm8_k64_minus_perceiver_{epoch}ep_final_seedmean"]
            for epoch in (10,30)
        },
        "scientific_decision":"DEVELOPMENT_LEVEL_LOCKED_ENSEMBLE_BEATS_PERCEIVER_IMPLEMENTATION_BUT_NO_APPROVED_BENCHMARK_OR_RESERVED_TEST_CLAIM",
        "disclaimers":[
            "Our locked ensemble has lower paired pooled and protein-group macro DEV RMSE than final-epoch PerceiverCPI seed means (6 seeds at 10, 2 at 30) and than optimistic dev-best comparison.",
            "Original protein groups are 3-mer proxies and do not guarantee homology separation; these dev results do not enforce scaffold-cold chemistry.",
            "BALM source lacks per-record assay DOI, construct and conditions; contradictory same-input labels.",
            "DeepDTA-inspired single-run selected best DEV epoch pooled 1.37176, which numerically beats the locked ensemble 1.38374; not evidence our model beats every alternative.",
            "Strong ESM+memory model is exploratory; not statistically better than the locked classical ensemble on this dev split.",
            "Development outcomes have been inspected repeatedly and bootstrap intervals are not independent confirmation.",
            "Perceiver 10 vs 30 epoch baselines were not matched on equal GPU compute or hyperparameter tuning budgets against the simpler ensemble.",
            "The scientific cohort gate remains BLOCKED; reserved prospective test must remain untouched.",
        ],
    }
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf8")
    md=[
        "# DynaCPI v4 — completed matched PerceiverCPI and memory evaluation",
        "",
        "**2026-11-08 | Experimental development evidence, NOT a valid prospective-test or paper-level win.**",
        "",
        "## Confirmed matched inputs and reproducibility",
        "",
        f"- The same 3,891 existing BALM inner-dev examples and 173 original protein 3-mer proxy groups; SHA256 {sha}.",
        "- All 8 completed PerceiverCPI runs share this exact row hash. PerceiverCPI is the original InteractionModel, with a 500-token protein input and fixed training configuration.",
        "- Six distinct 10-epoch seeds (2103, 2601–2605), and two 30-epoch seeds (2601–2602). The first 10 epochs for matching seeds 2601 and 2602 have exactly identical saved predictions between 10-epoch and 30-epoch runs.",
        "- No reserved prospective test data was loaded. Both ensembles fit only the 15,538 original TRAIN records (12,315 mean-collapsed exact pairs).",
        "",
        "## Core matched results (lower RMSE is better)",
        "",
        "| Candidate | Pooled RMSE | Protein-group macro RMSE |",
        "|---|---:|---:|",
        f"| Locked Ridge + explicit interactions | **{locked_metrics['pooled_rmse']:.6f}** | **{locked_metrics['component_macro_rmse']:.6f}** |",
        f"| PerceiverCPI 10 ep, mean over 6 final seeds | {summary['10']['final_epoch_seedmean_pooled']:.6f} ± {summary['10']['final_epoch_seed_sd_pooled']:.6f} (seed SD) | {summary['10']['final_epoch_seedmean_macro']:.6f} ± {summary['10']['final_epoch_seed_sd_macro']:.6f} |",
        f"| PerceiverCPI 30 ep, mean over 2 final seeds | {summary['30']['final_epoch_seedmean_pooled']:.6f} ± {summary['30']['final_epoch_seed_sd_pooled']:.6f} (seed SD) | {summary['30']['final_epoch_seedmean_macro']:.6f} ± {summary['30']['final_epoch_seed_sd_macro']:.6f} |",
        f"| Exploratory ESM2 8M + fixed joint memory k64 | {memory_metrics['memory_joint_esm8_k64']['pooled_rmse']:.6f} | {memory_metrics['memory_joint_esm8_k64']['component_macro_rmse']:.6f} |",
        f"| Exploratory ESM2 8M base without memory | {memory_metrics['memory_research_base_no_memory']['pooled_rmse']:.6f} | {memory_metrics['memory_research_base_no_memory']['component_macro_rmse']:.6f} |",
        f"| Mean training-label prediction | {memory_metrics['train_mean']['pooled_rmse']:.6f} | {memory_metrics['train_mean']['component_macro_rmse']:.6f} |",
        "",
        "The Perceiver values above are averages of **individual seed RMSEs**, NOT RMSE of averaged predictions. Error bars represent seed-level sample standard deviation. Seed-averaged-prediction ensembles are evaluated separately below.",
        "",
        "## Paired 5,000-draw protein-group bootstrap: locked ensemble minus PerceiverCPI",
        "",
        "| PerceiverCPI comparator | Δ pooled RMSE [95% group interval] | Δ macro RMSE [95% group interval] |",
        "|---|---|---|",
    ]
    for ep in (10,30):
        s=summary[str(ep)]
        for name,short in (("locked_ensemble_vs_perceiver_primary",f"{ep}ep final seed mean, PRIMARY"),
                           ("locked_ensemble_vs_optimistic_dev_best",f"{ep}ep dev-best epoch, OPTIMISTIC"),
                           ("locked_ensemble_vs_seed_averaged_predictions",f"{ep}ep averaged seed predictions")):
            comp=s[name];delta=comp["delta_ens_minus_perceiver"]
            cp=comp["ci95_pooled"];cm=comp["ci95_macro"]
            md.append(f"| {short} | {delta['pooled']:+.6f} [{cp[0]:+.6f}, {cp[1]:+.6f}] | {delta['macro']:+.6f} [{cm[0]:+.6f}, {cm[1]:+.6f}] |")
    md+=["",
        "Negative deltas favor our ensemble. Both prespecified PRIMARY 95% intervals exclude zero for pooled and protein-group macro RMSE at both training budgets. Even optimistic selection of PerceiverCPI's best development epoch per seed leaves an observed gap in this specific exploratory split.",
        "",
        "## Individual neural-seed training behavior",
        "",
        "| Epoch budget | Seed | Final pooled RMSE | Final macro RMSE | Best DEV epoch / pooled RMSE |",
        "|---:|---:|---:|---:|---|",
    ]
    for ep in (10,30):
        for r in summary[str(ep)]["per_seed"]:
            md.append(f"| {ep} | {r['seed']} | {r['final_pooled_rmse']:.6f} | {r['final_macro_rmse']:.6f} | {r['best_dev_pooled_epoch']} / {r['best_dev_pooled_rmse']:.6f} |")
    md+=["",
        "The 30-epoch runs did not improve on their DEV-best scores; seed 2601 best epoch 12 and seed 2602 best epoch 14. Both 30-epoch runs have identical predictions to their corresponding 10-epoch runs through epoch 10. This is late-training DEV degradation at these seeds/settings; it does not prove the original architecture is intrinsically weaker after tuning.",
        "",
        "## Strong alternative control: DeepDTA-inspired CNN",
        "",
        f"The prior **DeepDTA-inspired single seed** on this same existing DEV partition recorded final epoch 10 RMSE {deep_final['pooled_dev_rmse']:.6f} / macro {deep_final['macro_group_dev_rmse']:.6f}. Its best DEV-selected epoch {deep_best['epoch']} pooled RMSE was **{deep_best['pooled_dev_rmse']:.6f}**, numerically BETTER than our locked ensemble's {locked_metrics['pooled_rmse']:.6f}. This is a DEV-optimized epoch selection, not a locked fair winner, and per-row DeepDTA predictions are unavailable for the matched group bootstrap. Do NOT claim all baselines are beaten.",
        "",
        "## Exploratory memory model on exact same DEV rows",
        "",
        f"Frozen ESM2 8M representations were generated for 209 DEV-only protein sequences (30 truncated at 1,022 residues); neither dev target labels nor reserved test inputs were used for feature extraction. The trained model itself was already fitted only to original TRAIN. Memory+ESM2 fixed k64 yields {memory_metrics['memory_joint_esm8_k64']['pooled_rmse']:.6f} pooled and {memory_metrics['memory_joint_esm8_k64']['component_macro_rmse']:.6f} macro.",
    ]
    md+=["",
        "| Exploratory paired comparison (new memory minus reference) | Δ pooled RMSE [95% group interval] | Δ macro RMSE [95% group interval] |",
        "|---|---|---|",
    ]
    for desc,key in (
        ("Joint ESM memory vs locked classic","memory_model_vs_classic"),
        ("Joint ESM memory vs chemical-only memory","memory_protein_aware_vs_chemical_only"),
    ):
        v=output[key];d=v["delta_ens_minus_perceiver"];p=v["ci95_pooled"];m=v["ci95_macro"]
        md.append(f"| {desc} | {d['pooled']:+.6f} [{p[0]:+.6f}, {p[1]:+.6f}] | {d['macro']:+.6f} [{m[0]:+.6f}, {m[1]:+.6f}] |")
    md+=["",
        "**Neither memory-vs-locked nor joint-protein-vs-chemical-only paired interval excludes zero on this DEV split.** The memory augmentation did help relative to old architecture in strict TRAIN-only alignment+scaffold cold CV, but this development run does not establish a reproducible advantage. Do not choose from the variants based on this DEV view and then claim an independent win.",
        "",
        "## Publication blockers and next decisions",
        "",
        "1. The current DEV protein groups use an earlier approximate 3-mer grouping; a separate TRAIN-only Smith–Waterman audit found many cross-group alignments. The DEV comparison also lacks explicit Murcko scaffold exclusions, so it is NOT the same as alignment-and-scaffold-cold TRAIN cross-validation.",
        "2. BALM source labels have unresolved assay provenance, methods, protein constructs and repeated-input measurement disagreements. An independently reviewed assay-harmonized scientific cohort and locked homology/scaffold split are required.",
        "3. The model family and selection thresholds have been inspected repeatedly using development outcomes. Resampling the same 173 groups cannot undo adaptive model selection.",
        "4. The DeepDTA-inspired best-DEV-epoch counterexample precludes blanket claims of superiority; a fair DeepDTA/PerceiverCPI replication requires matched hyperparameter selection and compute budgets.",
        "5. The scientific cohort gate remains BLOCKED_UNAPPROVED_NOT_TRAINING_READY. **Do not score or tune on the reserved prospective test** until approvals and pre-registration exist.",
        "",
        "## Reproduction",
        "",
        "- Source seed files: research/balm_perceiver_multiseed_dev_20261108/perceiver_seed*_ep*.json (8 reports, 6 ten-epoch and 2 thirty-epoch).",
        "- Prior frozen classical reference: research/balm_ensemble_dev_predictions_20261108.json.",
        "- Python paired comparison: scripts/compare_balm_perceiver_vs_ensemble_dev.py --epochs 10; repeat --epochs 30.",
        "- Novel pretrained DEV inputs feature cache: scripts/cache_balm_frozen_esm8m_innerdev_inputs_dev.py. Binary NPZ kept locally; checksum in research/balm_esm2_t6_8m_innerdev_input_only_manifest_dev_20261108.json.",
        "- Exact same rows memory evaluation: scripts/evaluate_balm_memory_vs_perceiver_matched_innerdev_dev.py and research/balm_memory_vs_perceiver_matched_innerdev_20261108.json.",
        "- Report generator: scripts/summarize_balm_perceiver_multiseed_complete_dev.py.",
        "",
    ]
    REPORT.write_text("\n".join(md),encoding="utf8")
    print("SAVED",OUT,"and",REPORT)
    print("PRIMARY_RESULT",json.dumps({
        k:{"pooled":v["locked_ensemble_vs_perceiver_primary"]["delta_ens_minus_perceiver"]["pooled"],
           "macro":v["locked_ensemble_vs_perceiver_primary"]["delta_ens_minus_perceiver"]["macro"]}
        for k,v in summary.items()}))

if __name__=="__main__":parse()
