# KIBA & Metz double-cold validation novelty diagnostic (11 November 2026)

**Post-hoc validation-only experiment.** Preselected two-seed models and immutable
train/validation indices were replayed exactly. No held-out double-cold test
labels were indexed or scored. These findings must not be used to retrain,
tune, select, or modify any pending frozen experiments.

## Frozen overall validation comparison

| Dataset | Pairs | Dyna macro | Prior macro | Dyna pooled | Prior pooled |
|---|---:|---:|---:|---:|---:|
| KIBA | 3,166 | 0.781430 | 0.789896 | 0.758663 | 0.755425 |
| METZ | 658 | 0.777782 | 0.789936 | 0.778389 | 0.803484 |

## Error by nearest-neighbor similarity (training set only)

Scores computed per unique held-out-validation protein or molecule,
then binned by thirds of *unique-entity* nearest-training similarity.
Each diagnostic table reports pair-level pooled RMSE; negative Δ favors DynaCPI.
The chemical-prior anchor was fitted only on training labels.
Protein similarities are cosine of cached frozen ESM-2 vectors;
chemical similarities are Tanimoto of binary 2048-bit Morgan fingerprints.


### KIBA
**Protein nearest-training ESM-2 cosine**; boundaries: 0.9779, 0.9872; 34 validation entities.

| Validation similarity bin | Pairs | Dyna pooled RMSE | Prior pooled RMSE | Δ pooled | Dyna better pairs |
|---|---:|---:|---:|---:|---:|
| low similarity | 980 | 0.7716 | 0.7701 | +0.0015 | 51.5% |
| mid similarity | 1,005 | 0.6808 | 0.6843 | -0.0035 | 55.6% |
| high similarity | 1,181 | 0.8090 | 0.7995 | +0.0095 | 43.9% |

**Molecule nearest-training Morgan Tanimoto**; boundaries: 0.4103, 0.6399; 261 validation entities.

| Validation similarity bin | Pairs | Dyna pooled RMSE | Prior pooled RMSE | Δ pooled | Dyna better pairs |
|---|---:|---:|---:|---:|---:|
| low similarity | 947 | 0.7126 | 0.7032 | +0.0094 | 46.8% |
| mid similarity | 1,217 | 0.8145 | 0.8097 | +0.0048 | 50.9% |
| high similarity | 1,002 | 0.7300 | 0.7345 | -0.0045 | 52.0% |

### METZ
**Protein nearest-training ESM-2 cosine**; boundaries: 0.9803, 0.9884; 27 validation entities.

| Validation similarity bin | Pairs | Dyna pooled RMSE | Prior pooled RMSE | Δ pooled | Dyna better pairs |
|---|---:|---:|---:|---:|---:|
| low similarity | 183 | 0.7927 | 0.8027 | -0.0100 | 51.9% |
| mid similarity | 214 | 0.7633 | 0.7527 | +0.0106 | 50.0% |
| high similarity | 261 | 0.7805 | 0.8433 | -0.0628 | 61.3% |

**Molecule nearest-training Morgan Tanimoto**; boundaries: 0.3730, 0.5377; 146 validation entities.

| Validation similarity bin | Pairs | Dyna pooled RMSE | Prior pooled RMSE | Δ pooled | Dyna better pairs |
|---|---:|---:|---:|---:|---:|
| low similarity | 173 | 0.6817 | 0.6833 | -0.0016 | 52.0% |
| mid similarity | 242 | 0.8133 | 0.8361 | -0.0227 | 59.1% |
| high similarity | 243 | 0.8068 | 0.8483 | -0.0415 | 53.1% |

## Caveats

- Correlated protein–compound pair counts; subgroup sizes differ. No subgroup p-values, causal explanations, or universal out-of-domain claims.
- The bins are diagnostics, not calibrated uncertainty estimates. Validation is used for model selection, so all findings are exploratory.
- This is a DynaCPI-versus-chemical-prior *validation* audit; there is no held-out double-cold comparison to original PerceiverCPI here.
- The ESM-2 nearest-neighbor measure does not prove protein family independence; similarly, Morgan distances do not guarantee identical chemical scaffold boundaries.
- The public report intentionally excludes raw labels, predictions and compound/protein identifiers. Full aggregate diagnostics and SHA hashes are in the gitignored local output directory.
