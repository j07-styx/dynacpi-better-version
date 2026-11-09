# Centered geometry: per-target diagnostics (DAVIS development only)

Seed 1126; selected cohort: 486 training and 216 validation pairs, four validation proteins with 54 pairs each. Not independent scientific evaluation.

Frozen sequence-only target-macro RMSE: 0.9256850780053539. Best geometry target-macro RMSE: 0.9256875145943682 at epoch 1; epoch 5: 0.9258728107071799.

| Validation protein sequence SHA-256 prefix | Pairs | Baseline RMSE | Geometry epoch 1 RMSE | Geometry epoch 5 RMSE | Delta epoch 5 |
|---|---:|---:|---:|---:|---:|
| `266e943682a0` | 54 | 1.285434927 | 1.285434848 | 1.285424578 | -0.000010348 |
| `673349c5c82f` | 54 | 1.378873133 | 1.378873055 | 1.378858117 | -0.000015016 |
| `75ba1d9bc769` | 54 | 0.617670648 | 0.617680494 | 0.618444918 | +0.000774270 |
| `841f0c04e7ea` | 54 | 0.420761604 | 0.420761661 | 0.420763629 | +0.000002025 |

**Interpretation:** Two proteins improve slightly, one deteriorates substantially, and one deteriorates slightly by epoch 5. This is target-heterogeneous behavior on only four validation proteins. The strongest deterioration dominates the mean. Do not choose hyperparameters on these four proteins and then treat them as an unbiased holdout.

Next: use a separate verified development cohort and structure-negative controls to establish whether geometry provides genuine signal, then evaluate on a genuinely independent protein-disjoint holdout.

Local artifacts (not committed): `research/pilot_compound_conditioned_geometry_centered_per_target_dev/seed_1126/`.
