"""Descriptive paired component bootstrap for the protein-prior exploratory dev ablation."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"research"
r=json.loads((R/"bindingdb_cold_protein_sequence_prior_inner_dev.json").read_text())
components=r["component_error_diagnostic_anonymous"]
assert len(components)==r["dev_components"] and sum(c["pair_count"] for c in components)==r["dev_pairs"]
n=np.asarray([c["pair_count"] for c in components], dtype=np.float64)
mean=np.asarray([c["train_mean_rmse"] for c in components])
safe=np.asarray([c["pooled_safe_rmse"] for c in components])
macro=np.asarray([c["macro_selected_rmse"] for c in components])
assert np.isclose(np.sqrt((n*(mean**2)).sum()/n.sum()),r["dev_train_mean"]["pooled_rmse"])
assert np.isclose(np.sqrt((n*(safe**2)).sum()/n.sum()),r["dev_safe_pooled_model"]["pooled_rmse"])
rng=np.random.default_rng(20261108)
draws=rng.integers(0,len(components),size=(10000,len(components)))
n_boot=n[draws]
def pooled(errors):
    return np.sqrt(np.sum(n_boot * errors[draws]**2,axis=1) / np.sum(n_boot,axis=1))
base_pooled=pooled(mean);base_macro=mean[draws].mean(axis=1)
report={
  "status":"EXPLORATORY_COMPONENT_BOOTSTRAP_NO_INDEPENDENT_TEST",
  "samples":int(len(draws)),
  "resampling_unit":"protein component, sampled with replacement",
  "seed":20261108,
  "component_count":len(components),
  "estimates":{},
  "caveats":["Exploratory internal development has been inspected repeatedly; these bootstrap intervals are descriptive, not confirmatory.",
             "Only 20 independent components, heavily imbalanced pair counts.",
             "Assays and constructs are not yet fully verified.",
             "No reserved-test labels or predictions were used."]
}
for name, errors in [("cv_selected_macro_prior",macro),("cv_selected_conservative_prior",safe)]:
    diff_pooled=base_pooled-pooled(errors)
    diff_macro=base_macro-errors[draws].mean(axis=1)
    report["estimates"][name]={
       "pooled_rmse_improvement_observed":float(r["dev_train_mean"]["pooled_rmse"] - r["dev_selected_model" if name=="cv_selected_macro_prior" else "dev_safe_pooled_model"]["pooled_rmse"]),
       "pooled_improvement_bootstrap_95_interval":np.quantile(diff_pooled,[.025,.975]).tolist(),
       "pooled_bootstrap_fraction_improvement":float(np.mean(diff_pooled>0)),
       "component_macro_improvement_observed":float(r["dev_train_mean"]["component_macro_rmse"] - r["dev_selected_model" if name=="cv_selected_macro_prior" else "dev_safe_pooled_model"]["component_macro_rmse"]),
       "macro_improvement_bootstrap_95_interval":np.quantile(diff_macro,[.025,.975]).tolist(),
       "macro_bootstrap_fraction_improvement":float(np.mean(diff_macro>0))
    }
(R/"bindingdb_cold_protein_sequence_prior_component_bootstrap_dev.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
