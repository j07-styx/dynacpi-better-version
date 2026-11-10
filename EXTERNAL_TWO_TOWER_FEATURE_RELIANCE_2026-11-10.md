# External validation-only two-tower reliance diagnostics — 10 Nov 2026

## Question and design

Does the implemented LowRankDynaCPI v5 residual head use BOTH the frozen ESM-2 protein embedding and learned Morgan chemical representation beyond its training-derived chemical prior? We ran fixed-output VALIDATION-ONLY feature-permutation negative controls; no additional training and no new TEST evaluation.

For KIBA and Metz, on the three previously frozen protein-cold, compound-cold and double-cold validation splits, the original two-seed checkpoints (2931 + 2932) were reconstructed and their frozen validation metrics EXACTLY VERIFIED. Then independently permuted one modality's feature rows with five fixed seeds, holding observed validation pair IDs, original affinity scores, the other modality's features and training-only chemical anchor unchanged.

- Protein feature shuffle: randomly reassign normalized protein embedding rows among distinct validation proteins, five seeds 31001–31005.
- Chemical feature shuffle: randomly reassign Morgan 2048-bit feature rows supplied to the learned chemical tower among distinct validation ligands, five seeds 32001–32005. KEEP the correctly derived TRAINING-ONLY chemical prior for each true ligand ID. This is an ablation of the learned chemical interaction residual, not the fixed prior.
- Metric is change in VAL target-macro RMSE relative to original two-seed selected checkpoints. Positive means shuffled model worse, indicating that the particular representation matters. Input shuffling also moves predictions outside training-distribution, so this is an important sanity check, NOT causal biological proof.

## Actual results: validation target-macro RMSE and mean five-permutation change

| External dataset / split | DynaCPI VAL macro RMSE | Protein embedding shuffle Δ | Chemical tower fingerprint shuffle Δ |
|---|---:|---:|---:|
| KIBA protein-cold | 0.657419 | +0.088075 | +0.086368 |
| KIBA compound-cold | 0.642766 | +0.158978 | +0.117243 |
| KIBA double-cold | 0.781430 | +0.022975 | +0.023281 |
| Metz protein-cold | 0.707470 | +0.058666 | +0.037675 |
| Metz compound-cold | 0.758222 | +0.238550 | +0.133873 |
| Metz double-cold | 0.777782 | +0.019021 | +0.026828 |

All twelve mean deltas are positive. This supports the limited conclusion that both neural representation branches carry useful predictive signal instead of all output being a simple mean-lookup chemical prior. Feature permutation does NOT demonstrate the model is causal, calibrated or superior to trained PerceiverCPI.

Weak generalization is especially notable under DOUBLE COLD: the protein and chemical shuffle penalties are small (+0.0190 to +0.0268 Metz; ~+0.023 KIBA). Previously locked KIBA double-cold DynaCPI pooled VALIDATION RMSE 0.758663 was actually WORSE than the training-only chemical-prior pooled validation RMSE 0.755423. Preserve this negative result rather than hiding it.

All numbers are VALIDATION diagnostics, NOT independent sealed TEST evidence. KIBA is a composite score, NOT pKd; KIBA and Metz RMSE values are not directly comparable in absolute magnitude.

## Reproducibility and data boundaries

Public source code in the repository:
- audit_external_protein_feature_shuffle_val_20261110.py
- audit_external_chemical_feature_shuffle_val_20261110.py
- train_external_sparse_dynacpi_v5_20261110.py
- evaluate_external_kiba_metz_all_six_once_20261110.py
- KIBA_METZ_EXTERNAL_VALIDATION_STATUS_2026-11-10.md

Full per-seed diagnostic JSON in ignored LOCAL external_benchmarks_20261110/locked_external_protocol_v1/{dataset}/{scenario}/DYNACPI_PROTEIN_SHUFFLE_VALIDATION_DIAGNOSTIC.json and DYNACPI_CHEMICAL_FEATURE_SHUFFLE_VALIDATION_DIAGNOSTIC.json. Source files, sealed test indexes and affinity labels, original PerceiverCPI checkpoints, DynaCPI model states and local logs remain gitignored.

Permutation OOD effects and repeated validation evaluation do NOT constitute a true newly held-out confirmatory experiment. Original PerceiverCPI two-seed external matched checkpoints are being trained; the sealed six-scenario evaluation gate remains locked. External success requires the same held-out test rows, honest metric aggregation, and reporting every negative split. GVP 3D structure and K-shot support adaptation are UNIMPLEMENTED and cannot be credited for these results.
