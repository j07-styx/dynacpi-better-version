# Centered compound-conditioned geometry: DAVIS engineering pilot

Development-only analysis; **not** a held-out scientific benchmark or evidence of improvement.

Frozen sequence-only validation target-macro RMSE: **0.9256850780053539** (lower is better).

| Seed | Best epoch | Best validation target-macro RMSE | Delta vs baseline |
|---|---:|---:|---:|
| 1122 | 1 | 0.9256842701696273 | -0.0000008078357266 |
| 1123 | 1 | 0.9256853319664828 | +0.0000002539611289 |
| 1124 | 1 | 0.9256864253200320 | +0.0000013473146781 |

Mean of the three per-seed best validation scores: **0.925685342485381**; mean delta: **+0.000000264480027** (slightly worse).

All three runs trained on 486 pairs and validated on 216 pairs (9 training proteins and 4 validation proteins), stopped after five epochs with four non-improving epochs. These are small, selected development cohorts; per-seed best checkpoint selection uses the same validation set.

Seed 1121 has only one recorded epoch and no completed manifest; it is **excluded** from the three-run mean. Its existing files were preserved.

## Interpretation

Geometry performance is effectively indistinguishable from the frozen sequence-only baseline on this selected pilot. Small improvements in one seed are not replicated across the others. Longer training worsened validation performance. Do not claim a generalization gain, compare to PerceiverCPI paper metrics, or publish these development results as an independent evaluation.

## Next experiments

Investigate whether the geometry gate/residual has sufficient effect (gate distribution, correction magnitude, per-target error deltas), then test on a separately validated protein-disjoint cohort. Preserve provenance, immutable checkpoints and the development/holdout boundary.

Source artifacts (not committed): `research/pilot_compound_conditioned_geometry_centered_multiseed_dev/seed_{1122,1123,1124}/manifest.json` and `history.json`.
