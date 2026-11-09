"""Produce a public aggregate development report without exporting raw observations."""
import hashlib, importlib.metadata, json, platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
study=ROOT/"research/bounded_fusion_study_20261107"
summary=json.loads((study/"summary.json").read_text())
analysis=json.loads((study/"analysis.json").read_text())
replay=json.loads((study/"checkpoint_replay.json").read_text())
audit=json.loads((ROOT/"research/roadmap_foundations_20261107.json").read_text())
probe=json.loads((ROOT/"research/gvp_gradient_probe_20261107.json").read_text())
baseline=json.loads((study/"baseline.json").read_text())
release=ROOT/"research/roadmap_release_20261107"
release.mkdir(exist_ok=True)  # Regenerates derived reports only; never model/data artifacts.
states=[
(1,"COMPLETE","Existing registry; new study has an isolated protocol and 18 run records."),
(2,"COMPLETE (development audit)","Exact sequence/pair/canonical compound/scaffold overlap plus 192 global alignments. Domain-family leakage remains unresolved."),
(3,"BLOCKED (experimental verification)","13 graph hashes verified during loading; mapping manifest reviewed; assayed constructs remain unverified."),
(4,"BLOCKED","No source-reviewed independent cohort or frozen family-disjoint split. Existing validation has been repeatedly used."),
(5,"IN PROGRESS","Floor counts audited; original units/censoring and record-level experimental meanings not independently verified."),
(6,"COMPLETE (development checkpoint parity only)","Recomputed sequence baseline 0.9256850828. Full matched scientific baseline remains blocked by step 4."),
(7,"COMPLETE (development diagnostics)","Old GVP path gradient probe plus per-layer gradients for all new trainable variants."),
(8,"COMPLETE (development sensitivity)","Prediction changes, shuffled features, disabled geometry, rotation and permutation tests."),
(9,"COMPLETE (development controls)","Three matched seeds for real, shuffled and sequence-only residuals, plus disabled geometry via head-only ablation."),
(10,"COMPLETE (development analysis)","Train/validation trajectories and per-target harms recorded; two of four targets harmed by real fusion."),
(11,"COMPLETE (development implementation)","Layer-normalized residue fusion with scalar correction bounded to 0.25 standardized label units; opt-in."),
(12,"COMPLETE (development implementation)","Differentiable compound-query softmax residue weights; not a verified binding-site predictor."),
(13,"COMPLETE (development implementation)","64-dimensional compound/sequence/local-structure attention with direct output residual."),
(14,"COMPLETE (limited development ablation)","Dropout, weight decay, anchoring and confidence value weighting tested. no_confidence removes weighting, not confidence input features."),
(15,"COMPLETE (negative development experiment)","Last prediction-layer tuning tested with and without fusion; epoch-zero model wins all six runs."),
(16,"COMPLETE (development ablations)","Six variants times three seeds; no claim of exhaustive architecture or parameter-matched capacity comparison."),
(17,"COMPLETE (development multi-seed study)","Means, sample SD, paired seed effect and descriptive target bootstrap. Four targets, not independent families."),
(18,"BLOCKED","No untouched verified holdout; no test-set evaluation performed."),
(19,"IN PROGRESS","Archived local Perceiver predictions scored on same 216 pairs; full matched paper-baseline reproduction and independent comparison blocked."),
(20,"IN PROGRESS","Development source/report/manuscript/reproduction package prepared; final scientific release blocked by steps 3-5,18-19.")]
table="| Step | Status | Evidence / limitation |\n|---|---|---|\n"+"\n".join(f"| {i} | {s} | {e} |" for i,s,e in states)
metrics_table="| Variant | Mean macro RMSE | Sample SD | Relative reduction vs frozen |\n|---|---:|---:|---:|\n"
for name,r in summary.items():
    metrics_table+=f"| {name} | {r['mean']:.9f} | {r['sd']:.9f} | {100*r['relative_reduction']:.3f}% |\n"
report=fr"""# DynaCPI development research release — 7 November 2026
## Decision
Do not promote the geometry extension. It improves the frozen sequence pilot by 0.50%, but the sequence-only residual is better on every matched seed. A training-only compound mean is substantially stronger than both. No independent scientific superiority is established.
## Results
Frozen sequence baseline: {baseline['metrics']['macro_rmse']:.9f}.
{metrics_table}
Training-only global mean RMSE: {analysis['training_only_baselines']['global_mean']['macro_rmse']:.9f}.
Training-only compound mean RMSE: {analysis['training_only_baselines']['compound_mean']['macro_rmse']:.9f}.
Archived local Perceiver CPI common-cohort RMSE: {analysis['archived_perceiver_common_cohort']['macro_rmse']:.9f}. This is a three-epoch engineering comparator, not a paper benchmark or a fresh retraining.
Real minus sequence residual paired-seed mean: {analysis['real_minus_sequence']['mean']:.9f} (positive is worse).
Real-versus-frozen descriptive target-bootstrap interval: {analysis['variants']['real']['descriptive_95_interval']}. It includes zero. Checkpoint selection used these same validation targets, so this is not an independent inference.
## Data
486 mapped training pairs (nine proteins), 216 validation pairs (four proteins), 54 shared compounds. Exact protein and pair overlap is zero. 368 training and 138 validation labels equal 5.
Global pairwise alignment screening found no pair satisfying both 30% paired identity and 80% bilateral coverage; this does not exclude conserved domain or family leakage. The full pilot contains 24/8 proteins, 1296/432 pairs, and was used for the frozen foundation.
All graph hashes matched. Manifest coverage is 100% for 13 mapped proteins; experimental construct identity is unresolved.
## Architecture and training
The optional new branch projects frozen 256-dimensional protein and compound features to 64 dimensions. Local structural inputs comprise 14 scalar graph-node features and three vector magnitudes. It does not use graph edges and is not a replacement GVP topology model. Softmax compound queries attend over normalized sequence-plus-structure residue features. An MLP produces a tanh-bounded correction of at most 0.25 standardized label units.
All other backbone weights are frozen. Two ablations selectively train only the copied final prediction layer. AdamW uses learning rate 0.001 and weight decay 0.01, eight epochs, batch size eight, anchor coefficient 0.25. Seeds are 1201–1203. Epoch zero is eligible, preventing a worse trained model from replacing the baseline.
The shuffled control permutes node scalars and vectors together within each protein. The sequence-only control omits structural values but retains the same covered residues; active parameter counts are not matched. The confidence ablation removes explicit value weighting but retains confidence as an input feature. No binding-pocket localization claim is made.
## Verification
18 saved checkpoints replayed on all 216 validation pairs; maximum absolute prediction discrepancy {max(r['max_absolute_error'] for r in replay['runs']):.12g}. Numerical differences arise from equivalent reduction paths.
Tests passed for zero-init identity, absent-geometry fallback, bounded corrections, batch independence, residue permutation, rotation invariance and query/structure gradients.
The old GVP gradient probe found zero early-layer gradients before the first update, then nonzero but tiny norms after updates. This is attenuation, not a permanently disconnected path.
## Roadmap
{table}
## Next scientific decision
Stop treating the frozen three-epoch neural checkpoint as the only meaningful baseline. Retain training-only compound means and sequence-only capacity controls. Repair data viability and freeze an independent domain/family protocol before further model selection. Do not create a test split by relabeling these repeatedly observed validation proteins.
## Reproduction
Use the existing Windows environment and local caches; Python environment inventory is in environment.json. Data and checkpoints are intentionally not in Git.
From the repository root:
```powershell
.\.venv\Scripts\python.exe tests_v4\bounded_structural_fusion_test.py
.\.venv\Scripts\python.exe scripts\audit_roadmap_foundations_dev.py --output research\foundations_new.json
.\.venv\Scripts\python.exe scripts\run_bounded_fusion_study_dev.py --output research\bounded_fusion_study_new
```
Study creation refuses an existing output directory. Analysis/replay scripts currently target the dated study path; change the explicit study path for a new run. Existing caches, source records and old checkpoints are preserved. GVP diagnostic additionally requires the existing installed gvp dependency.
## References and comparison boundaries
Perceiver CPI uses nested cross-attention and was evaluated on Davis, KIBA and Metz; its published benchmark settings differ from this engineering study [1]. Davis et al. is the original kinase-selectivity source [2]. These references do not independently certify our record mappings.
1. Nguyen et al. Perceiver CPI. Bioinformatics, btac731. https://pmc.ncbi.nlm.nih.gov/articles/PMC9848062/
2. Davis et al. Comprehensive analysis of kinase inhibitor selectivity. 2011. https://pubmed.ncbi.nlm.nih.gov/22037378/
"""
(release/"REPORT.md").write_text(report,encoding="utf-8")
(ROOT/"research/dynacpi_20_step_implementation_plan_dev.md").write_text("# DynaCPI 20-step roadmap\n\nUpdated 7 November 2026. Development execution is not scientific completion.\n\n"+table+"\n\nEvidence: research/roadmap_release_20261107/REPORT.md. New development study: research/bounded_fusion_study_20261107 (local).\n",encoding="utf-8")
paper="""# Bounded residue fusion fails to establish geometry-specific gains in a small DynaCPI development cohort
Research manuscript draft — engineering evidence only, not ready for submission.

## Abstract
We investigated whether a bounded compound-conditioned residue readout could overcome attenuated gradients in an existing structure-augmented affinity predictor. Eighteen controlled runs on a nine-protein training and four-protein validation cohort compared six configurations across three seeds. Real structural features reduced target-macro RMSE from 0.925685 to 0.921045 on average, but a sequence-only residual reached 0.920612 and a training-only compound mean reached 0.839873. Results do not establish geometry-specific value or independent generalization.

## Introduction
Perceiver CPI integrates compound and protein representations using nested cross-attention [1]. Our extension explores structural information within a frozen sequence foundation. We ask whether gains survive sequence-only and disrupted-structure controls, rather than assuming that architectural complexity implies useful biology.

## Methods
The engineering cohort, transformations, full architecture, matched controls, seeds, optimizer settings, checkpoint selection and exact metrics are specified in REPORT.md. Local scalar structure features and rotation-invariant vector norms are fused with projected frozen ESM-derived sequence representations; no graph message passing is used in the new readout. Output corrections are bounded. We retain the frozen baseline as an epoch-zero candidate. Model selection and evaluation here use the same development validation set.

## Results
Real-structure mean RMSE was 0.921045 (sample SD 0.001718); sequence-only residual mean was 0.920612 (SD 0.001778). Shuffled structures achieved 0.921663 (SD 0.003075). Real structure was worse than sequence-only on each matched seed. Two validation proteins improved and two worsened. A descriptive target bootstrap interval for real-minus-frozen RMSE spans zero. Selective final-layer tuning never improved over epoch zero. All 18 saved checkpoints replayed within 0.000002 affinity units. The original GVP probe showed zero early-layer gradients at initialization followed by extremely small nonzero gradients.

## Discussion
The direct residual removes an indirect readout bottleneck but does not establish that geometry adds useful information. Compound means substantially outperform these neural variants, exposing the weakness of using only the frozen neural model as comparator. The geometry subset is small, its chemistry overlaps across splits, and its affinity floor is common. No causal conclusion about the usefulness of all protein structure follows from this negative result.

## Limitations
Experimental constructs and censoring semantics remain incompletely verified. No frozen family-disjoint independent holdout is available. Seeds reuse the same four validation targets and are not independent biological replicates. Bootstrap intervals are descriptive after model selection. The sequence control is not active-parameter matched. Confidence weighting removal is not complete removal of confidence metadata. The archived Perceiver comparison is a local three-epoch implementation, not a reproduction of published benchmark scores.

## Conclusion
This study supports retaining simpler baselines and repairing the evaluation foundation before promoting a geometry extension. It provides working software and falsifiable negative results, not a claim of a state-of-the-art model.

## Data and code availability
Source and aggregate reports are versioned on the dynacpi-v4 branch. Raw local records, caches and checkpoints are excluded from publication. Reproduction requires the recorded input hashes and existing local environment.

## References
[1] Nguyen et al. Perceiver CPI. https://pmc.ncbi.nlm.nih.gov/articles/PMC9848062/
[2] Davis et al. Comprehensive analysis of kinase inhibitor selectivity. https://pubmed.ncbi.nlm.nih.gov/22037378/
"""
(release/"MANUSCRIPT_DRAFT.md").write_text(paper,encoding="utf-8")
env={"python":platform.python_version(),"packages":{}}
for name in ["torch","numpy","scipy","rdkit","biopython","torch-geometric","torch-scatter","torch-cluster","fair-esm"]:
    try:env["packages"][name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:env["packages"][name]="not registered"
(release/"environment.json").write_text(json.dumps(env,indent=2))
(release/"aggregate_results.json").write_text(json.dumps({"summary":summary,"real_minus_sequence":analysis["real_minus_sequence"],
    "baseline_rmse":baseline["metrics"]["macro_rmse"],"compound_mean_rmse":analysis["training_only_baselines"]["compound_mean"]["macro_rmse"],
    "replay_max_error":max(r["max_absolute_error"] for r in replay["runs"]),"gvp_probe":probe},indent=2))
files=["dynacpi/bounded_structural_fusion.py","scripts/run_bounded_fusion_study_dev.py","scripts/audit_roadmap_foundations_dev.py",
       "scripts/analyze_bounded_fusion_study_dev.py","scripts/verify_bounded_checkpoints_dev.py","scripts/diagnose_gvp_gradient_path_dev.py",
       "tests_v4/bounded_structural_fusion_test.py"]
hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
hashes.update(audit["inputs"])
hashes["research/pilot_sequence/checkpoint.pt"]=baseline["checkpoint_sha256"]
(release/"hashes.json").write_text(json.dumps(hashes,indent=2))
print("Release written:",release)
