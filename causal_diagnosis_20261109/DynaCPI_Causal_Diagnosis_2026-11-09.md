# DynaCPI: causal diagnosis and tested chemistry repair

9 November 2026. This continues the earlier geometry diagnosis with new checkpoint probes, a training-only capacity test, and a fresh-initialization internal target-holdout experiment.

## Main finding

The current foundation is a poorly trained, strongly averaging predictor whose easiest path is to adjust protein-level affinity. Its compound pathway is severely attenuated at two places: molecular channel attention and the final protein-valued channel attention. Freezing this foundation makes the geometry branch inherit that weakness. The evidence supports a combined architecture and optimization failure, rather than a conclusion that GVPs or protein geometry cannot help affinity prediction.

The isolated repair preserves molecular features directly and exposes them directly to the prediction head. It passes the small training-set capacity test. Held-target results are reported separately below; memorization and restored chemical sensitivity are not claims of generalization.

## 1. Correct the experimental bookkeeping first

The original `SequenceFoundation` trained on **24 proteins × 54 compounds = 1,296 pairs**, not nine proteins. Its original validation panel contains eight proteins / 432 pairs. The geometry-ready subset contains nine train and four validation proteins. Geometry fitting used seven of those train proteins, with two for internal selection.

The original foundation ran three epochs at effective batch size 64: `3 × ceil(1296 / 64) = 63 optimizer updates`. It was then frozen. The nine-protein count in the earlier diagnosis described the structure-mapped subset, not the original foundation's complete training set.

The original eight-target macro RMSE is **0.817272**. The four geometry-ready development targets have a different macro RMSE, approximately **0.925685**. These are different evaluation panels and must not be mixed.

## 2. The channel-attention layer is nearly an averaging operator

The local implementation constructs channel logits as:

`logit[i,j] = q[i] * k[j] / sqrt(256)`

Each channel is a scalar in this operation. There is no 256-term dot product. The usual scaled dot-product rationale divides by the square root of the dimension being summed over; here that dimension is one. The scaling is an architectural temperature choice, inherited from the local upstream CAB implementation, rather than a newly introduced tensor-shape error. It nevertheless contributes to very small logits in this checkpoint.

Measured effective channel coverage, `exp(entropy) / 256`:

| Operation | Coverage | Relative output difference from exactly uniform weighting |
|---|---:|---:|
| Fingerprint query over molecular graph channels | 99.99996% | 0.0330% |
| Molecular self-channel attention | 99.99907% | 0.1506% |
| Final compound query over protein channels, ADCK4 | 99.99889% | 0.8056% |

At exactly uniform weights, every channel receives the same average of the value vector. After the output projection, most molecular variation is restricted to one direction plus a common bias. The observed final compound representation has **centered effective rank 1.00016**, computed from the entropy of squared singular values across the 54 compounds. This is stronger evidence than cosine similarity alone, which can be misleading when embeddings share a large common component.

A fresh random initialization also showed an effective rank near one after the final channel layer. This is one initialization probe, not a proof that every initialization behaves identically or that the architecture cannot learn its way out.

The query gradient has the form `d output_i / d q_i = Cov_attention(k, v) / scale` before the output projection. Small covariance and flat weighting attenuate the molecular query. In a fixed training-only gradient probe, the fingerprint-projection gradient norm was **1.83e-6**, versus **2.85e-2** for the MPN: approximately **15,575× smaller**. This is a gradient-signal comparison, not an Adam parameter-update ratio.

Crucially, changing the scale from 16 to 1 on the same trained weights did **not** restore ADCK4 chemical discrimination: prediction SD changed from 0.001311 to 0.001232 pKd. A temperature-only patch is insufficient for these weights.

## 3. There are two compound bottlenecks

The molecular encoder uses the fingerprint only as a query into graph-derived values, then applies another channel-attention layer without preserving a direct fingerprint path. The final prediction also uses the compound only as a query into protein-derived values. Therefore, repairing only the encoder leaves another weak compound path at the output.

The experimental repair implemented in `causal_diagnosis.py` is:

```
c = LayerNorm(FingerprintMLP(fp) + MPN(smiles))
r = CrossAttention(query=c, keys=protein_tokens, values=protein_tokens)
p = LayerNorm(masked_mean(protein_tokens) + r)
y_hat = PairMLP(concat(c, p, c * p, abs(c - p)))
```

It removes per-molecule min/max scaling and both molecular channel-attention calls, then replaces the last channel-attention prediction route with direct molecular, protein, and interaction features. The existing ESM projection and residue attention remain. These changes are bundled in the encoder arm, so the experiment does not isolate min/max normalization from residual preservation. The pair head adds parameters; this is not a capacity-matched architecture benchmark or a novel algorithm claim.

## 4. Training-only capacity test

One original training protein was chosen by its training-label spread, and 16 compounds spanning its recorded label distribution were selected. Every arm started from the same historical foundation weights for shared components; new components were initialized with the same fixed seed. All parameters were unfrozen. Dropout was disabled for all arms, learning rate was 0.001, and every arm received 200 updates on the same 16 pairs. No held-out labels were used.

| Arm | Initial RMSE | RMSE after 200 updates |
|---|---:|---:|
| Original architecture | 1.6607 | 0.2602 |
| Molecular encoder repair only | 1.6746 | 0.0979 |
| Prediction-head repair only | 2.1146 | 0.1008 |
| Both repairs | 2.1109 | **0.0013** |

The original architecture can recover chemical variation with continued end-to-end optimization. Its failure is not mathematically inevitable. Both repairs together make this small problem much easier to fit. Single-seed training-set fitting does not establish cold-target or cold-compound performance, and intermediate losses were not monotonic.

A separate dropout probe found SD 0.06347 pKd for repeated copies of the same pair in training mode, versus only 0.0000611 across 54 distinct compounds in evaluation mode for that training target. This illustrates how stochastic training outputs can look more variable than genuine chemical predictions; it does not prove dropout caused the collapse.

## 5. Internal held-target experiment

The 24 historical training proteins were repartitioned deterministically, without labels, into 18 fit proteins and six internal validation proteins. All 54 compounds are shared. Both architectures started from matched fresh random shared weights; the supervised historical foundation checkpoint was not reused. Only pretrained ESM features were reused.

Three seeds, 20 epochs, 360 updates per arm, learning rate 3e-4, weight decay 1e-4. A batch contains the 54 compounds for one protein. Checkpoint selection uses six-target macro RMSE at epochs 0, 5, 10, 15 and 20. The original eight development proteins and prospective test are not used in this experiment. Labels are standardized from the 18 fit proteins only.

| Model | Seed 2831 | Seed 2832 | Seed 2833 | Mean macro RMSE |
|---|---:|---:|---:|---:|
| Original architecture, trained from scratch | 0.852656 | 0.885467 | 0.854968 | **0.864363** |
| Both chemical bypasses | 0.805819 | 0.830650 | 0.786210 | **0.807560** |
| Compound mean across 18 fit proteins | 0.816618 | 0.816618 | 0.816618 | **0.816618** |
| Global mean across 18 fit proteins | 0.886023 | 0.886023 | 0.886023 | **0.886023** |

The repair improves over the original in all three seeds, with a **6.57% lower mean RMSE**. It beats the compound-mean baseline in two of three seeds; its mean advantage is only **1.11%**. The sample SD across optimizer seeds is 0.01831 for the original and 0.02227 for the repair. These are not confidence intervals, and the small baseline advantage is not established as robust.

Permuting compounds against their labels, holding repaired weights fixed, increases mean inner RMSE from **0.807560 to 1.041096**. This supports useful compound sensitivity on this internal panel. The original also develops chemical sensitivity in two seeds, but seed 2832 selects a model with effectively constant predictions: prediction SD about 7e-7 and no change under the compound shuffle. The repair avoids that collapsed outcome in the three tested seeds.

The selected epochs were 20, 10, 15 for the original and 10, 5, 5 for the repair. Later epochs did not consistently improve validation error; simply extending training indefinitely is not the solution. Total training time for all six arms was approximately five minutes on the connected device.

This remains internal engineering validation on historically used data. Seed variability measures optimizer variability, not independent biological evidence. Six targets in one split cannot establish superiority over PerceiverCPI or a geometry benefit. The comparison is not parameter matched, and the new pair head does not use dropout, while inherited modules retain their existing dropout.

## 6. Why the ADCK4 correction is harmful

ADCK4's recorded mean is **5.193585 pKd**, while the frozen foundation predicts **5.436333** on average: it is already high by **0.242748**. **47 of 54 labels (87.0%)** are at the benchmark floor, pKd 5.

For an approximately constant geometry correction `delta`, the change in MSE is:

`new_MSE - old_MSE = 2 * delta * mean(prediction - label) + delta**2`

Inserting `delta = +0.101879` predicts RMSE **0.664348**, essentially the observed **0.664356**. Thus the large deterioration is explained almost completely by moving an already overestimated target mean farther upward. This arithmetic explains the harm; it does not identify which molecular structural feature produced the correction.

The new fixed-weights geometry probe separates residual magnitude from readout sensitivity:

| Target | Geometry residual RMS | Gradient norm for a uniform residue-feature shift | Actual mean geometry shift | Shift after replacing all residue residuals by their mean |
|---|---:|---:|---:|---:|
| FRK | 0.08728 | 0.00290 | +0.003784 | +0.003775 |
| MEK5 | 0.08573 | 0.00418 | +0.002537 | +0.002544 |
| ADCK4 | **0.08271** | **0.16953** | **+0.101879** | **+0.102375** |
| OSR1 | 0.08636 | 0.00221 | +0.000668 | +0.000666 |

ADCK4 receives a slightly smaller residual in RMS terms, yet its frozen readout is **41-77 times more sensitive** to uniform feature shifts. The gradient is measured at the mean compound embedding; it is a local sensitivity, not a global bound.

Replacing ADCK4's residue-specific residuals with one broadcast mean vector preserves essentially its whole prediction shift. Subtracting that mean from every residue reduces its mean shift to **-0.000497 pKd**. The same latent perturbation, formed from the seven fit proteins and applied identically to every target, produces **+0.111660** on ADCK4 but only +0.003646, +0.002519, and +0.000732 on FRK, MEK5, and OSR1. These are artificial feature-space interventions, not physical structural mutations.

This supports a specific mechanism: the frozen readout is unusually sensitive for ADCK4, and a global geometry-feature offset drives most of the effect. It does not require ADCK4 to have unusually large structural residuals. This is different from blaming only the explicit mean-pooling path: the broadcast vector changes both mean pooling and attention-derived protein values.

A frozen, nearly compound-insensitive predictor naturally encourages geometry to learn target-level residual means. Permuting residue mapping preserves pooled geometric content, while weak compound-conditioned attention offers little alignment sensitivity. Similar real/shuffled scores therefore undermine a residue-level alignment claim, but do not prove that every possible global structural descriptor is useless.

## 7. A strong simple baseline is still necessary

Using each compound's mean affinity across the 24 original training proteins gives **0.701664 macro RMSE** on all eight original development proteins, compared with **0.817272** for the frozen sequence checkpoint: a **14.15% relative RMSE reduction**. No validation labels are used to fit those means.

This is a known-compound lookup baseline. It ignores the target sequence and cannot predict a completely unseen compound. Its purpose is to demonstrate that beating the collapsed model alone is a weak success criterion. This eight-target result must not be compared numerically with the separate six-target internal experiment.

## 8. Data and objective pressures

The original training panel has **74.6%** labels at pKd 5. The benchmark distribution provides a strong incentive for an early model to predict a low-variance average. This does not make MSE intrinsically wrong, nor does it establish that all floor entries are identical physical measurements.

DeepDTA's description of DAVIS reports that the pKd-5 mass includes weak or unobserved screening pairs. Treating every such value as an exact measured 10,000 nM affinity can misstate the underlying assay. Keep the benchmark evaluation convention explicit. A censor-aware or binding/nonbinding-plus-affinity loss should be introduced only when source metadata establishes the appropriate semantics, not inferred automatically from the number 5.

## 9. Prioritized repair path

1. **Use the two-bypass repair as a candidate sequence baseline.** Keep a sufficiently trained original architecture as a control. Do not freeze the supervised chemical and interaction pathways before they demonstrate training-set discrimination and internal validation performance. Frozen pretrained ESM features are a separate, reasonable compute decision.
2. **Optimize and measure within-target discrimination.** Track per-target RMSE, centered MSE, target mean bias, within-target ranking excluding tied floor labels, and molecule-permutation sensitivity. Prediction variation alone is not success; multiple compounds can truly have similar affinity.
3. **Test a target-centered auxiliary loss on training data.** For example, add a modest, predeclared weight to the squared error between centered predictions and centered labels within each target. A protein-only offset cannot solve this term. Compare against ordinary MSE; do not assume it improves generalization, and retain the baseline target-bias term.
4. **Require a win over strong simple controls.** Include global mean, compound mean for known compounds, fingerprint-only learning for new compounds, and simple fingerprint/protein-kernel or bilinear predictors. Report both known-compound and unseen-compound settings explicitly.
5. **Stabilize the sequence baseline before bringing geometry back.** Retrain the supervised readout with the repaired compound representation; placing a new encoder in front of a frozen old readout is not a complete repair. Increase meaningful training coverage, not just GVP epochs on seven proteins.
6. **Evaluate geometric information separately from target calibration.** Compare real geometry, residue-mapping corruption, feature/edge corruption that changes geometry itself, and a parameter-matched sequence residual. A mapping shuffle alone preserves global structure summaries. Report within-target effects as well as mean shifts.
7. **Use ligand-conditioned structural interactions if pursuing structural novelty.** Conditioning geometry or atom/residue interactions on retained ligand features is more credible than an unconditional whole-protein residual. Protein backbone geometry without ligand pose is a structural prior, not an explicit bound-complex model. Sharp attention alone is not evidence of a physical binding site.
8. **Keep scientific claims separate from the demo.** No new independent evaluation was performed, no reserved test was opened, and no claim of surpassing the published PerceiverCPI benchmark is supported by these experiments.

## Reproducibility and artifacts

Scripts and aggregate results are in `C:\Users\J0124\DynaCPI-Backend\causal_diagnosis_20261109`. The original repository's model files, historical checkpoints, frontend, and live API configuration were not edited. The experimental candidates were not promoted to the application. No commits or pushes were made.

Run using the existing environment, in that directory, with CUDA available:

```powershell
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' causal_diagnosis.py
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' repair_holdout.py
& 'C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe' geometry_sensitivity.py
```

Each script refuses to overwrite its completed result. Use a new experiment directory to repeat. GPU access uses the existing lease; sources are read from the existing checkout. Aggregated results contain the source hashes, training partitions, epoch histories, per-target outcomes, and limitations. These scripts are reproducible within the existing environment rather than a standalone training package.

## Primary references

- Vaswani et al., *Attention Is All You Need*, section 3.2.1, for the actual dot-product scaling rationale: https://arxiv.org/html/1706.03762v7
- Ozturk et al., *DeepDTA: deep drug-target binding affinity prediction*, dataset section, for DAVIS pKd transformation and the floor-value distribution: https://pmc.ncbi.nlm.nih.gov/articles/PMC6129291/

The measured attention, gradients, training behavior, and prediction changes above come from this user's local code and experiments; the papers do not establish the cause of this particular checkpoint's failure.
