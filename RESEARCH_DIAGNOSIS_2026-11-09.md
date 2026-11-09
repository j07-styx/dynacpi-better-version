# DynaCPI v4 — Why geometry training underperforms
**9 November 2026 | Technical diagnosis | Source: local Demo-smth/dynacpi-v4 checkout**

## Bottom line

Our DynaCPI v4 geometry branch does not yet deliver an identifiable chemistry-specific 3D contribution. The principal measured failure begins upstream: **the frozen sequence foundation is effectively compound-insensitive** for 54 chemically diverse DAVIS pilot compounds, and geometry training creates almost a protein-specific bias. This finding is stronger and more actionable than blaming the GVP architecture alone.

This is an **engineering and previously inspected development-set analysis**, not an independent biological benchmark. The scientific BindingDB source/assay/construct gate remains \`BLOCKED_UNAPPROVED_NOT_TRAINING_READY\`.

## What was audited

1. Original \`dynacpi/sequence_model.py\`, \`dynacpi/geometry_branch.py\`, \`dynacpi/frozen_pair_predictor.py\`, and \`scripts/train_v4_geometry_trainonly_20261109.py\`.
2. Archived DynaCPI v4 experiments: \`research/bounded_fusion_study_20261107/\`, the GVP seeds, confidence gates, compound-conditioned variants, structural cross-attention, node-shuffle, and readout-path sensitivity checks.
3. Newly replayed 3 real-alignment and 3 misaligned-alignment trained checkpoints, **all with frozen original foundation**.
4. Same trained checkpoint evaluated twice—correct 3D graph/residue mapping vs deliberately permuted residue mapping.
5. Direct decomposition of the original predictor’s mean pooling vs compound-query residue-attention contributions.
6. Molecular fingerprint diversity, learned compound encoder layer activations, singleton-versus-batch inference consistency, and attention concentration.

The separate backend API and training runs were built on the existing local data and checkpoints, but **nothing from this diagnostic was committed or pushed to the old GitHub repository**.

## New train-only-selected geometry runs (all 6 epochs, three seeds)

| Seed | Correct geometry dev macro RMSE | Misaligned control dev macro RMSE | Frozen sequence dev macro RMSE |
|---|---:|---:|---:|
| 1901 | 0.936573 | 0.934605 | 0.925685 |
| 1902 | 0.939935 | 0.937851 | 0.925685 |
| 1903 | 0.937222 | 0.936788 | 0.925685 |

Checkpoint selection used a **2-protein inner subset from the original 9 protein training cohort**; the remaining 7 provided 378 GVP training pairs. Crucially, the frozen foundation was originally trained on **all nine**, including these two, so the inner split is not a clean foundation-level holdout. Four previously inspected DAVIS validation proteins (216 pairs) were only scored after checkpoint selection but were used repeatedly in earlier work. Neither split is adequate for scientific performance claims.

All six geometry runs lost against the frozen sequence checkpoint on the previously inspected development panel. Training loss was almost flat (~0.749 standardized units). This is not proof that geometry is intrinsically inferior.

## Per-protein failure is dominated by ADCK4

Seed 1901, real alignment, 54 compounds per target:

| Protein | Frozen sequence RMSE | Geometry RMSE | Delta | Mean geometry prediction shift | SD of shift across compounds |
|---|---:|---:|---:|---:|---:|
| FRK | 1.285435 | 1.283626 | -0.001809 | +0.003784 | 0.0000136 |
| MEK5 | 1.378873 | 1.377394 | -0.001479 | +0.002537 | 0.0000026 |
| ADCK4 | 0.617671 | 0.664356 | +0.046686 | +0.101879 | 0.0000581 |
| OSR1 | 0.420762 | 0.420917 | +0.000155 | +0.000668 | 0.0000005 |

The geometry correction is practically **constant within a protein**; it does not substantially distinguish which compound the model sees. The large positive shift for ADCK4 worsens its error enough to dominate the four-protein mean.

A crucial same-weights counterfactual: keep seed-1901 geometry network weights fixed, *only* permute its residue mapping. ADCK4 predictions then differ by a mean absolute **0.000619 pKd**, compared with the real network's **+0.101879 pKd** average shift. Thus most prediction change is **not sensitive to correct residue-level alignment** in this implementation.

## Foundation already fails to discriminate molecular chemistry

The 54 compounds are distinct and structurally varied: mean pairwise Morgan Tanimoto **0.136694**; **91.4%** of pairs have similarity below 0.2.

Yet the sequence-foundation predictions are almost constant within a protein:

| Target | Recorded affinity SD | Frozen model predicted SD |
|---|---:|---:|
| FRK | 1.128337 | 0.0000135 |
| MEK5 | 1.119820 | 0.0000308 |
| ADCK4 | 0.568175 | 0.0013109 |
| OSR1 | 0.409238 | 0.0000082 |

Singleton-vs-16-molecule batch inference changed predicted pKd by only approximately 1.1–1.4×10⁻⁷ on average: batch behavior **does not** explain this collapse for the tested 54 compounds.

## Exact molecular encoder bottleneck

Representation direction diversity by measured average pairwise cosine:

| Stage | Cosine |
|---|---:|
| Raw 2048-bit fingerprint input | 0.238416 |
| Learned fingerprint projection | **0.450239** |
| Raw MPN molecular graph output | **0.994760** |
| First custom channel attention output | **0.999872** |
| Final custom channel attention output | **0.999885** |

The fingerprint projection still encodes appreciable variation. The MPN is already highly similar across compounds, and the custom channel attention produces near-identical embeddings.

Architecture mechanism from \`dynacpi/sequence_model.py\`:
- \`graph = MPN(smiles)\`, rescaled by per-molecule min/max.
- \`mixed = graph + ChannelAttention(FP(fingerprint), graph, graph)\`.
- \`compound = ChannelAttention(mixed, mixed, mixed)\`.

The molecular fingerprint enters the first attention as **query**, while nearly constant graph features provide its keys and values. There is **no direct fingerprint residual skip** into the final compound representation. The final attention layer further compresses the remaining chemical variation. The learned weights/limited training could exacerbate this; this trace does not prove the architecture alone universally must collapse.

The compound-conditioned protein attention is also diffuse: **95–97% effective residue coverage**, with across-compound attention-weight variation ~**1.1–1.2%** of across-residue variation. This makes fine-grained ligand-specific structural corrections hard to learn.

## Geometry readout ablation

Same frozen seed-1901 model, ADCK4:
- Full geometry tokens: **+0.101879** mean pKd shift.
- Geometry tokens only in the **mean-pool** pathway: **+0.046962**.
- Geometry tokens only in the **compound-query attention** pathway: **+0.056144**.

Both paths contribute; it is incorrect to attribute everything to mean pooling. But neither path produces meaningful compound-specific variation, consistent with the collapsed molecular query embedding.

## Archived studies reinforce the diagnosis

- \`bounded_fusion_study_20261107/summary.json\`: mean development RMSE across 3 seeds, **real 0.921045, shuffled 0.921663, sequence-only residual 0.920612**, vs frozen foundation 0.925685. The sequence-only control was slightly stronger than real geometry: no isolated geometry advantage.
- \`centered_geometry_multiseed_dev_summary.md\`: real centered geometry mean **0.925685342** vs sequence 0.925685078; essentially indistinguishable.
- \`centered_geometry_node_shuffle_control_dev.md\`: shuffled structural node features are indistinguishable/slightly better than correctly aligned nodes.
- \`pilot_compound_conditioned_geometry_centered_diagnostics_dev/seed_1125/manifest.json\`: the geometry gate was ~2.9% at epoch 1; mean output shift around **9.4×10⁻⁷ pKd**, indicating that the conservative gate essentially silenced the branch.
- \`pilot_gated_geometry_extended_dev/seed_1110/manifest.json\`: validation error increases with successive epochs.
- \`pilot_structural_cross_attention_warmstart_dev/seed_1128/manifest.json\`: validation error also increases with successive epochs.

These repeated previously inspected development comparisons should not be used to select yet another model and then treated as a fresh test.

## External research and implications

1. **PerceiverCPI** (Bioinformatics 2023): nested cross-attention for protein/compound interaction. Our v4 "SequenceFoundation" is *not* a reproduction of the published PerceiverCPI architecture. Claims of beating the paper are not supported.
   https://academic.oup.com/bioinformatics/article/39/1/btac731/6842322
2. **GVP-GNN** (ICLR 2021): geometric vector perceptrons model 3D protein geometry, originally demonstrated for protein model quality assessment and protein design. A GVP branch alone does not guarantee ligand-binding discrimination.
   https://mlanthology.org/iclr/2021/jing2021iclr-learning/
3. **GenSPARC**: structure-aware protein language representations with Foldseek/SaProt and compound multimodal representations plus interaction attention. That is more than simply adding geometric residual tokens to a weak frozen predictor.
   https://www.nature.com/articles/s42004-025-01844-0
4. **EquiCPI** (JCIM 2025): geometry via ESMFold/DiffDock-L, atomic points and equivariant interaction modeling. Unlike our whole-protein backbone GVP, it introduces ligand-level 3D context.
   https://pubmed.ncbi.nlm.nih.gov/40600339/
5. **PLABench** (September 2026 bioRxiv preprint): target-dependent structural-model performance, sequence models sometimes winning, and importance of leakage-controlled, family-level evaluation. This matches the *kind* of heterogeneity seen here but does not validate our particular result.
   https://sciety.org/articles/activity/10.64898/2026.09.17.752433
6. **CleanSplit/GEMS** (Nature Machine Intelligence 2025): severe structure-overlap leakage can inflate affinity benchmark results, strengthening the rationale for human assay/source validation and disjoint splits.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12552120/

## Diagnostic conclusion (ranked)

1. **Verified major bottleneck**: severe chemistry representation / prediction collapse in the frozen sequence foundation. Diverse fingerprints become near-identical learned compound embeddings and near-constant pKd predictions.
2. **Verified secondary failure**: geometry corrections are almost protein-wide offsets, insensitive to correct residue-to-graph alignment, and miscalibrate ADCK4.
3. **Highly plausible contributing cause**: diffuse protein attention and no ligand-pose/structural contact supervision; the 3D branch has no credible way to learn which spatial neighbors are relevant to a particular molecular structure.
4. **Experimental limitations**: only 7 current GVP fit proteins / 378 pairs; frozen foundation trained for 3 epochs on the original pilot; scientifically unverified assay constructs, numeric floor/censoring ambiguities, and repeatedly consulted four-protein DEV.
5. **Not observed as the cause**: CUDA/CPU device mismatch in the historical training run, invalid graph checksums in the selected 13 structures, or singleton-vs-batched molecular encoder results for these 54 compounds. The graph identity integrity checks had already passed; experimental construct provenance remains unresolved.

## Stop/repair plan — not yet a performance claim

1. **Do not keep scaling GVP epochs yet.** First create **unit assertions of chemical sensitivity**: chemically diverse inputs should not receive effectively identical embeddings and predictions for the same protein.
2. Replace the compound architecture's fingerprint-as-query-only branch with a **direct residual/concatenated 2048-bit fingerprint projection** and an MPN route that preserves molecular discrimination. Compare a capacity-matched **sequence-only** repaired baseline before reintroducing geometry.
3. Replace or redesign the diffuse protein readout to preserve **compound-conditioned local residue information**, then verify that changing ligand identity changes residue attention and predicted affinity.
4. Only then test genuine geometry benefit with **matched real versus scrambled graph mapping** and direct compound/structure interaction controls, using separate training-only validation.
5. Obtain independently source-reviewed, physical-construct-verified and protein/source-family-disjoint data before any confirmatory comparison. Do **not** unfreeze or score the reserved prospective test, and do not auto-approve BindingDB candidate tasks.
6. Finally compare to a strong **true PerceiverCPI** backbone with identical support-label budget and same evaluation units, not only to our weak SequenceFoundation baseline.

**Status:** Root-cause investigation complete to the level of traceable frozen-model activations and paired counterfactuals. No remedy has yet been implemented or validated; no improved accuracy or paper-ready claim is asserted.

### Local raw diagnostic artifacts (not pushed)
- \`C:\Users\J0124\dynacpi_v4\research\dynacpi_v4_trainonly_geometry_20261109\DIAGNOSIS_20261109.json\`
- \`...\READOUT_PATH_DIAGNOSIS_20261109.json\`
- \`...\ATTENTION_DISCRIMINATION_DIAGNOSIS_20261109.json\`
- \`...\CHEMISTRY_PIPELINE_DIAGNOSIS_20261109.json\`
- \`...\COMPOUND_ENCODER_LAYER_DIAGNOSIS_20261109.json\`
