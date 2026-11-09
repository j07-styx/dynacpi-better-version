# Centered geometry: deterministic node-feature shuffle control (development only)

**Design:** Same seed 1124 and selected DAVIS engineering cohort (486 training pairs, 216 validation pairs). Each protein graph receives a fixed deterministic permutation of node scalar/vector feature rows; graph edges, edge features, residue indices and frozen sequence foundation remain unchanged. This disrupts node feature/topology alignment but **does not** destroy all graph information. Outputs are isolated.

| Configuration | Best validation target-macro RMSE | Best epoch |
|---|---:|---:|
| Frozen sequence-only baseline | 0.9256850780053539 | n/a |
| Correctly aligned geometry, seed 1124 | 0.9256864253200320 | 1 |
| Shuffled node features, seed 1124 | 0.9256851028316732 | 1 |

Both geometry variants stopped at epoch 5 after four non-improving epochs. Shuffled control minus aligned geometry: **-0.0000013224883588** (lower RMSE, but negligible).

**Interpretation:** Correct node-feature alignment did not improve predictive performance in this tiny engineering pilot. This is not evidence that protein structure is uninformative in general: the control retains topology, the foundation is frozen, the cohort is selected and small, and model selection uses validation. Do not report as a scientific negative conclusion or an independent benchmark.

**Next:** Verify construct/graph mapping and establish a protein-disjoint holdout, then compare aligned and shuffled geometry across multiple seeds on a larger approved cohort.

Local artifacts (not committed): `research/pilot_compound_conditioned_geometry_centered_node_shuffle_dev/seed_1124/`.
