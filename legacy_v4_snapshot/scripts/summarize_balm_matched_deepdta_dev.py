"""Summarize the completed matched 10- and 30-epoch DeepDTA baseline runs.

Source is already committed classical and Perceiver comparisons plus
new DeepDTA seed reports. No original prospective test data.
"""
import json
from dynacpi.balm_public_data_dev import ROOT

BASE=ROOT/"research"
OUT=BASE/"BALM_COMPLETE_MATCHED_DEEPDTA_COMPARISON_20261108.md"

def numeric(comp):
    if "delta_ens_minus_perceiver" in comp:
        d=comp["delta_ens_minus_perceiver"]
        return d["pooled"],comp["ci95_pooled"],d["macro"],comp["ci95_macro"]
    return (comp["delta_a_minus_b_pooled"],comp["ci95_pooled"],
            comp["delta_a_minus_b_macro"],comp["ci95_macro"])

def main():
    if OUT.exists():raise FileExistsError(OUT)
    refs={}
    for e in (10,30):
        refs[e]=json.loads((BASE/f"balm_deepdta_vs_perceiver_and_cpi_paired_dev_20261108_ep{e}.json").read_text())
    locked=json.loads((BASE/"balm_matched_multiseed_complete_audit_20261108.json").read_text())["locked_classic_metrics"]
    md=[
        "# Completed DeepDTA + PerceiverCPI matched BALM development comparison",
        "",
        "**2026-11-08 | Repeatedly inspected BALM inner DEV; NOT a prospective test result.**",
        "",
        "All models share frozen original BALM TRAIN (15,538 raw pairs / 12,315 duplicate-mean collapsed) and original 3,891 DEV rows, 173 approximate 3-mer protein graph groups. There are six paired 10-epoch seeds (2103, 2601–2605) and two paired 30-epoch seeds (2601–2602). DeepDTA-inspired 1.94M parameter CNN and upstream PerceiverCPI ~4.67M parameter InteractionModel differ in input lengths and compute requirements. Epoch counts are not equivalent GPU-time budgets.",
        "",
        "## Final epoch comparison (lower RMSE is better)",
        "",
        "| Model | 10-epoch pooled | 10-epoch macro | 30-epoch pooled | 30-epoch macro |",
        "|---|---:|---:|---:|---:|",
        f"| Locked Ridge + interaction booster | {locked['pooled_rmse']:.6f} | {locked['component_macro_rmse']:.6f} | {locked['pooled_rmse']:.6f} | {locked['component_macro_rmse']:.6f} |",
    ]
    for family in ("deepdta","perceivercpi"):
        a=refs[10]["seed_family_metrics"][family]
        b=refs[30]["seed_family_metrics"][family]
        md.append(f"| {family}: mean over individual seed scores | {a['mean_of_seed_pooled_rmse']:.6f} | {a['mean_of_seed_macro_rmse']:.6f} | {b['mean_of_seed_pooled_rmse']:.6f} | {b['mean_of_seed_macro_rmse']:.6f} |")
    md+=["",
         "## Paired 5,000-draw protein-group bootstrap deltas",
         "",
         "| Comparison (A minus B) | Pooled RMSE delta [95% CI] | Macro RMSE delta [95% CI] |",
         "|---|---|---|"]
    checks=(
        ("Locked vs DeepDTA final","locked_ridge_interactions_minus_deepdta_seedmean_FINAL"),
        ("Locked vs DeepDTA DEV-best optimistic","locked_ridge_interactions_minus_deepdta_BEST_DEV_EPOCH_OPTIMISTIC"),
        ("ESM+memory vs DeepDTA final","esm_memory_joint8_k64_minus_deepdta_seedmean_FINAL"),
        ("DeepDTA vs Perceiver final","deepdta_seedmean_minus_perceiver_seedmean_FINAL")
    )
    for e in (10,30):
        for title,key in checks:
            d,p,m,ci=numeric(refs[e]["comparisons"][key])
            md.append(f"| {e} ep: {title} | {d:+.5f} [{p[0]:+.5f}, {p[1]:+.5f}] | {m:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] |")
    md+=["",
         "**Negative deltas favor A.** A confidence interval crossing zero is inconclusive under this group bootstrap. Even an interval excluding zero does not correct repeated development-data inspection or guarantee new-cohort generalization.",
         "",
         "## DeepDTA individual seed behavior",
         "",
         "| Budget | Seed | Final pooled | Final macro | Best-dev epoch / pooled |",
         "|---:|---:|---:|---:|---|"]
    for e in (10,30):
        for row in refs[e]["individual_seed_metrics"]:
            md.append(f"| {e} | {row['seed']} | {row['deepdta_final_pooled']:.6f} | {row['deepdta_final_macro']:.6f} | {row['deepdta_best_dev_epoch']} / {row['deepdta_best_dev_pooled']:.6f} |")
    md+=["",
         "## Scientific qualifications and next gates",
         "",
         "- Previous separate DeepDTA single-seed run achieved pooled DEV-best RMSE 1.371757 (epoch 8), numerically better than locked ensemble 1.383743, though DEV-selected epochs are optimistic. Preserve this counterexample.",
         "- Current model family and hyperparameters were explored repeatedly on DEV. These comparisons are not independent significance tests. Best-DEV-epoch selection is explicitly labeled optimistic.",
         "- Protein homology grouping uses an approximate 3-mer graph; subsequent exhaustive TRAIN Smith-Waterman alignment found cross-group related proteins. The original DEV split does not enforce Murcko scaffold exclusions.",
         "- BALM affinity labels have missing assay DOI, method, and protein construct provenance, as well as contradictory same-input observations. Scientific cohort approval gate remains blocked.",
         "- Same-seed separate 10- and 30-epoch DeepDTA jobs match only at epoch 1 and diverge starting epoch 2; this limits interpreting epoch-count changes as a pure causal effect.",
         "- Full original paper training schedules and tuned computation-matched configurations of DeepDTA and PerceiverCPI were NOT reproduced. Do not claim paper-level architecture superiority.",
         "- Frozen prospective test remained untouched. Obtain reviewed source/assay provenance, validated split, and registered analysis protocol before attempting test.",
         "",
         "## Reproduction",
         "",
         "Scripts: scripts/train_balm_deepdta_multiseed_matched_dev.py; scripts/compare_balm_deepdta_multiseed_matched_dev.py --epochs 10 or --epochs 30. Inputs: research/balm_deepdta_multiseed_dev_20261108/deepdta_seed*_ep*.json and equivalent PerceiverCPI seed files. Outputs: research/balm_deepdta_vs_perceiver_and_cpi_paired_dev_20261108_ep10.json and ep30.json. Integrity checks: tests_v4/balm_deepdta_multiseed_matched_test.py.",
         ""]
    OUT.write_text("\n".join(md),encoding="utf8")
    print("SAVED",OUT)
    for e in (10,30):
        for key in ("locked_ridge_interactions_minus_deepdta_seedmean_FINAL",
                    "deepdta_seedmean_minus_perceiver_seedmean_FINAL"):
            print("SUMMARY",e,key,refs[e]["comparisons"][key])

if __name__=="__main__":main()

