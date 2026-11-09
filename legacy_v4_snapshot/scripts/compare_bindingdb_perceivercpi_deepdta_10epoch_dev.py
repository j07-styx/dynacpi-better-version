"""Paired model comparison on the fixed provisional protein-component split."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
def load(name):return json.loads((R/name).read_text())
def main():
 deep=load('bindingdb_deepdta_reference_three_seed_dev.json');perc=load('bindingdb_perceivercpi_reference_10epoch_dev.json');audit=load('bindingdb_provisional_component_weighting_audit_dev.json')
 assert perc['train_pairs']==37441 and perc['validation_pairs']==4764 and perc['protein_cluster_disjoint']
 assert len(perc['history'])==10 and perc['history'][-1]['epoch']==10
 assert len(deep['validation_rmse_by_seed'])==3
 p=perc['history'][-1]['val_rmse'];d=deep['mean_rmse'];baseline=audit['pooled_mean_baseline_rmse']
 out={'status':'PROVISIONAL_DEVELOPMENT_SPLIT_NOT_INDEPENDENT_TEST','training_pairs':37441,'validation_pairs':4764,'validation_components':33,'fixed_epoch':10,'perceivercpi_seed_2103_epoch10_rmse':p,'deepdta_three_seed_epoch10_mean_rmse':d,'deepdta_three_seed_epoch10_sample_std':deep['sample_std_rmse'],'train_mean_pooled_rmse':baseline,'perceivercpi_minus_deepdta_mean_rmse':p-d,'perceivercpi_minus_mean_baseline_rmse':p-baseline,'perceivercpi_epoch_rmse':[x['val_rmse'] for x in perc['history']],'warnings':['Different learning rates, batch sizes, protein input lengths, and seed counts; same epoch count is not an equal compute budget.','PerceiverCPI results are one seed, DeepDTA results are three seeds.','Macro component RMSE is unavailable without per-pair predictions.','Validation set has been inspected repeatedly; no independent test.','Assay and protein construct provenance not fully verified.']}
 (R/'bindingdb_perceivercpi_deepdta_10epoch_comparison_dev.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
