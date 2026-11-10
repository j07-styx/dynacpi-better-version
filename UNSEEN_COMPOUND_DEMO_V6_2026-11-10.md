# DynaCPI v6: Unseen compound prediction demonstration — 10 November 2026

**Status:** 12 real checkpoint-backed predictions on six molecules excluded from the 68 original DAVIS training compounds, against two original DAVIS protein sequences (ABL1 and EGFR).

## Protocol

- Backend: fully refitted LowRankDynaCPI v6, all DAVIS 442 protein entries × 68 compounds, two seeds (2931, 2932), frozen ESM-2 650M embeddings, 2048-bit Morgan compounds, heuristic left-censored labels, CPU inference for prediction head.
- POST through **existing live React Vite frontend proxy** at http://127.0.0.1:5173/api/v2/predict to v6 model on port 8783, passing complete public DAVIS ABL1 and EGFR protein sequences and the new compound SMILES. No labels used in this test.
- The backend itself returned \`compound_in_original_davis = false\` for **every tested ligand**, \`protein_in_original_davis = true\` for both targets, and \`chemical_prior_source = Morgan Tanimoto kernel estimate\`.
- Follow-up script [analyze_unseen_compounds_v6_20261110.py](analyze_unseen_compounds_v6_20261110.py) recomputed each candidate's maximum Morgan Tanimoto similarity to all 68 original DAVIS compound fingerprints. It reads local private prediction log + prepared features; it does **not** fit or update model weights.
- These are **predicted pKd values** (higher = model predicts higher affinity), NOT experimental observations.

## Actual unseen-ligand inference results

| Ligand | SMILES | ABL1 pKd (pred.) | EGFR pKd (pred.) | Max Morgan Tanimoto vs any DAVIS training ligand |
|---|---|---:|---:|---:|
| Caffeine | Cn1c(=O)c2c(ncn2C)n(C)c1=O | 6.054554 | 5.393358 | 0.1875 |
| Aspirin | CC(=O)Oc1ccccc1C(=O)O | 5.302889 | 5.174658 | 0.1692 |
| Acetaminophen | CC(=O)Nc1ccc(O)cc1 | 5.431609 | 5.201283 | 0.2034 |
| Ibuprofen | CC(C)Cc1ccc(C(C)C(=O)O)cc1 | 5.442811 | 5.000000 | 0.1579 |
| Ethanol | CCO | 5.271558 | 5.010705 | 0.0862 |
| Naproxen | COc1ccc2cc(CC(C)C(=O)O)ccc2c1 | 5.247734 | 5.000000 | 0.2500 |

12/12 inference requests succeeded. The low Tanimoto similarities (0.086–0.250) place all six examples far from the *68 original DAVIS chemistry entries* under this radius-2 fingerprint representation; it is **not** a quantitative confidence or accuracy assessment. Novel compound priors are inferred from chemical fingerprint similarity to only 68 training compounds, then the two trained neural heads predict residuals.

### Scientific cautions

1. **There is no measured Kd/pKd ground truth here** for these newly evaluated protein–compound pairs. Consequently, **unseen-ligand RMSE is unknown**, and neither ranking nor absolute numerical values should be taken as experimentally accurate.
2. pKd=5 is a hard lower prediction clamp, so the exact 5.000000 values are clipped; if the reported two-seed standard deviation is zero at the clamp, this **does not signify high certainty**. Even nonzero seed disagreement is not a calibrated predictive interval.
3. Both proteins were seen when the full DAVIS model was refitted, so this probes **unseen ligand / known protein** inference, not double-cold generalization or independent protein prediction.
4. The nearest DAVIS inhibitor fingerprints are chemically distant from these examples; caffeine's predicted 6.05 against ABL1 in particular should not be interpreted as a credible claim of observed ABL1 inhibition.
5. The final v6 fit RMSE of 0.298741 is IN-SAMPLE on all DAVIS pairs. The earlier 0.585203 held-out v5 result is from different checkpoints on a protein-held-out split, and neither is an unseen-ligand benchmark or a matched PerceiverCPI comparison.

## How to demo

Open http://127.0.0.1:5173/ on the Windows research PC. Click **Load DAVIS example**, replace its SMILES with \`Cn1c(=O)c2c(ncn2C)n(C)c1=O\` or \`CC(=O)Oc1ccccc1C(=O)O\`, and click **Predict affinity**. The target loaded by default is not ABL1/EGFR, so the demo prediction may differ from the ABL1/EGFR rows above; for exact reproduction use those known DAVIS protein sequences. The UI will label such a pair as chemical extrapolation, not a held-out verified estimate.

**Next meaningful validation:** predeclare and run a held-out-*scaffold* DAVIS split, retrain using only retained compounds and proteins, and score the frozen held-out compounds with real Kd labels. Our fully refitted v6 has seen every original DAVIS ligand, so cannot serve as that cold-chemical test without retraining.
