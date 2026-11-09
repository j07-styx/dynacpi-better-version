# Double-cold protein-conditioned residual experiment — 2026-11-09

**Status: negative exploratory result. Not an independent or PerceiverCPI benchmark.**

## Question

Can a molecular chemical prior plus a learned, protein-dependent
compound-specific correction beat the previously observed
**0.6080476 target-macro double-cold RMSE** of the original chemical-only
kernel comparator?

## Design

Script: \`C:\Users\J0124\DynaCPI-Backend\test_conditional_residual_20261109.py\`

- Reuses only original DAVIS engineering **TRAIN** 24 protein × 54 compound measurements.
- Reuses three earlier target-identity-disjoint and fingerprint-/scaffold-disjoint outer folds.
- Outer fit: 16 proteins × 36 compounds; double-cold evaluation on the complement.
- Morgan/Tanimoto chemical prior learned on the outer fit rectangle.
- Kernel ridge residual is centered by **fit-set compound means** and uses both compound similarity and protein embeddings. Compared raw pooled ESM-2, train-centered ESM-2, and 3-mer protein-sequence kernels; regularization and correction strength were chosen by paired *inner* double-cold validation.
- Alpha=0 (no protein correction) was allowed as a candidate.
- No labels from the eight previously inspected DAVIS DEV targets, BindingDB candidates, or a reserved test set were loaded.
- All choices are inner-fold trained; the three outer scores are for reporting only.
- However, **the same outer splits and chemical baseline had already been inspected by earlier work** and motivated this experiment. The evaluation is exploratory, not confirmatory or independent.

## Results — target-macro RMSE, lower is better

| Comparison | RMSE |
|---|---:|
| Prior chemical-only comparator (previous nested selection) | **0.608048** |
| New chemical-only prior (independently reselected inner ridge) | 0.609942 |
| New protein-conditional kernel | **0.616156** |
| Protein identity swapped in the same trained model | 0.626179 |

The protein correction improves fold 0 (0.565707 to **0.553487**),
approximately ties fold 1 (0.756796 to 0.757858), and
hurts fold 2 (0.507324 to **0.537125**). The mean protein effect
magnitude is about **0.05174 pKd**.

Protein identity matters to the new predictor (swapping its protein features
worsens the result), but the correction does **not** reliably add predictive
value beyond a chemical-only comparator. The model does not meet the 0.6080
threshold and does not beat published PerceiverCPI.

## Limitations and scientific interpretation

- Only 24 historical training targets and 54 compounds. All prior experiment
  iterations and earlier outer-fold outcomes have been seen already.
- Protein identities were disjoint, but protein families / homologs were not
  grouped into independently validated protein components.
- Chemical novelty was defined by same Murcko scaffold OR Morgan Tanimoto
  at least 0.50, with all train-vs-heldout similarities below 0.50.
- Absolute pKd labels include historical DAVIS floor conventions; these
  are not independently verified construct-specific laboratory affinity measurements.
- The method is **simple kernel ridge correction**, not a novel validated 3D CPI
  architecture. It is not directly matched to the published PerceiverCPI splits.
- A more complicated predictor doing worse than a simple prior is a valid
  negative outcome. Further parameter tuning on the same outer folds would
  compound selection bias.

## Next defensible decision

**Do not promote this correction** to the live backend. Keep
\`profile_chemical\` as an explicit control and the anchored neural v4.3 models
as engineering candidates. Before reporting a scientific gain:

1. Use a fresh, independently source-reviewed, compound/scaffold and protein-family-disjoint cohort that has not been tuned against.
2. Require matched PerceiverCPI evaluation on the identical split, label transformations and metrics.
3. Demonstrate both absolute RMSE gain and a protein-label/protein-feature counterfactual gain beyond a reliable chemical prior.

Local artifacts: \`PROTOCOL.json\`, \`PROGRESS.json\`, \`RESULTS.json\`.
