# TransferCPI architecture and repository review

## Reviewed state

Started from commit 0827a0f; subsequent concurrent work reached 83d7051. Existing work now uses frozen 82 training / 20 inner-development / 33 prospective-test protein components. Train and dev counts are 27,118 and 7,387 pairs. Source/construct QC remains provisional. Existing PerceiverCPI comparisons used an older split and cannot serve as a matched baseline for this new model. The recent DeepDTA 5.26% gain compared best observed epochs, not final epochs, and still lost to the mean comparator. We preserve those findings and all running work.

Feature audit found ESM coverage for only 214/410 training sequences and 3/41 dev sequences. Missingness would confound a pretrained-model comparison. The new preparation step fills all 451 train/dev sequences, reusing 217 raw caches and extracting 234 missing sequences. No prospective-test features are generated. Checkpoint SHA256 is checked against the existing ESM cache metadata.

## Literature basis

GenSPARC combines structure-aware pretrained protein representations, molecular graph and property-aware language representations, and interaction attention. Its published Karimi and sequence-hard Davis/KIBA/Metz comparisons motivate pretrained local representations, but its scores cannot be compared directly with this provisional BindingDB split. Official sources: https://www.nature.com/articles/s42004-025-01844-0 and https://github.com/pfnet-research/GenSPARC .

PSC-CPI studies multi-scale protein sequence/structure contrasting and missing-modality inference: https://arxiv.org/abs/2402.08198 and https://github.com/LirongWu/PSC-CPI . This supports retaining local information and testing modality contributions. We do not claim to implement its contrastive pretraining here.

## Implemented architecture (original code)

1. Frozen ESM2-650M representations over the complete input sequence, using existing 1000-residue windows with 128 overlap. Compress to at most 64 contiguous, nonoverlapping mean tokens. All residues contribute; local resolution is reduced, and these tokens are not precise binding-site annotations.
2. Molecular atom embeddings plus ten atom attributes and three residual bond-type message-passing blocks. All atoms retained; largest molecule has 252 atoms. No SMILES string truncation.
3. Chiral 1024-bit Morgan fingerprint and twelve physicochemical descriptors, scaled using unique training molecules only.
4. Width-96 protein projection and four-head attention in both drug-to-protein and protein-to-drug directions. Masked pooling handles variable lengths without batch/padding dependence.
5. Chemical prediction plus a small-initialized interaction residual. Residual L2 penalty 0.01 in standardized-label units discourages unconstrained target-specific corrections. Nonzero initialization preserves gradient flow to the attention/protein branches.

Total trainable parameters: 572,226, excluding the frozen pretrained ESM model. No SaProt/Foldseek, pretrained chemical language model, or protein 3D structure is used in this version. It is GenSPARC-inspired, not a reproduction or a structure-aware model. No guarantee of superiority is made.

## Evaluation fixed before training

Fresh seed 2601, five epochs, batch32, identical deterministic row permutations, AdamW lr3e-4/weight_decay1e-4, train-only label mean/std, gradient clip1. Three arms: TransferCPI; identical architecture with chemical-only forward path; upstream PerceiverCPI InteractionModel with its 500-token protein input and 2048-bit fingerprint. The comparison controls split, sample exposure, seed and optimizer, not pretraining or encoder compute. This is a short development experiment, not full paper reproduction.

Select by minimum inner-dev component-macro RMSE; report fixed final epoch too. Also report pooled RMSE, per-component errors, train-only mean baseline and paired component-bootstrap intervals. No prospective test evaluation. Require gains over both the PerceiverCPI arm and simple comparators, then additional seeds before promotion. Pooled results can be dominated by one large dev component.

## Verification completed

Batch composition/order, padding values and atom permutation invariance; chemical-only independence from protein inputs; finite nonzero gradients through atom/protein/attention/interaction branches; state-save/load parity. All passed before training.

## Dual-pretrained variant (separate experiment)

The first candidate did not beat the first observed PerceiverCPI epoch. Rather than promote it, a second, separately logged candidate adds frozen DeepChem/ChemBERTa-77M-MTR molecular features (384-dimensional mean non-special token representation). Official checkpoint: https://huggingface.co/DeepChem/ChemBERTa-77M-MTR ; pretraining study: https://arxiv.org/abs/2209.01712 . This is molecular-property pretraining, not CPI-label fine-tuning or the SPMM encoder used in GenSPARC. The pretrained corpus may overlap downstream molecules; representation pretraining is disclosed.

Checkpoint revision is pinned to 66b895cab8adebea0cb59a8effa66b2020f204ca. All 14,455 train/dev molecules were encoded on CPU, with no missing encoder weights and no molecule exceeding the 510-token window. The model adds a chemical-language projection to the chemical head and conditions atom interaction tokens on that representation. New trainable size: 619,170. Paired arms are transfer_clm and chemical_clm, with the existing fresh PerceiverCPI run as reference under the same five-epoch schedule.

The isolated feature-extraction dependencies live under data/tooldeps_chemberta: transformers4.44.2, tokenizers0.19.1, huggingface-hub0.24.7, safetensors0.4.5, PyYAML6.0.2, regex2024.7.24. The primary venv packages were not upgraded. A queued second-seed repeat of the weaker first candidate was canceled before it acquired the GPU; replace it with replication only if the dual-pretrained candidate is promising. Keep all completed first-version results.
