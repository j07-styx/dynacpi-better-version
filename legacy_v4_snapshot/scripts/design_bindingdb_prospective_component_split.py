"""Construct a new untouched protein-component test split and training-only development split.
Existing development component hashes are excluded from the new test split.
No labels or model predictions are used to select components.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data/audit';OUT=ROOT/'research'
def rows(path):
 with path.open(encoding='utf8') as f:
  for line in f:yield json.loads(line)
comp={}
for g in rows(A/'bindingdb_full_kd_protein_components.jsonl'):
 for s in g['members']:comp[s]=g['component_id']
tasks={(r['sequence_sha256'],r['assay_key']) for r in rows(A/'bindingdb_full_kd_task_summary.jsonl') if r['unique_canonical_compounds']>=15}
samples=[r for r in rows(A/'bindingdb_full_kd_compound_median_candidates.jsonl') if (r['protein_seq_sha256'],r['assay_key']) in tasks and r['range_pKd']<=.1]
all_groups=sorted({comp[r['protein_seq_sha256']] for r in samples})
old_val=set(sorted(all_groups,key=lambda s:hashlib.sha256(('bindingdb-dev-20261107:'+s).encode()).hexdigest())[:len(all_groups)//5]);eligible=sorted(set(all_groups)-old_val)
# Test groups selected from previously training-only components, never inspected for model selection in previous experiments.
new_test=set(sorted(eligible,key=lambda s:hashlib.sha256(('dynacpi-prospective-test-20261108:'+s).encode()).hexdigest())[:max(1,len(all_groups)//5)])
remaining=set(eligible)-new_test
# Existing repeatedly inspected development validation groups are quarantined entirely.
train_groups=set(sorted(remaining,key=lambda s:hashlib.sha256(('dynacpi-prospective-train-20261108:'+s).encode()).hexdigest()))
assert train_groups.isdisjoint(new_test) and old_val.isdisjoint(train_groups|new_test)
counts=Counter(comp[r['protein_seq_sha256']] for r in samples)
report={'status':'PROSPECTIVE_SPLIT_DESIGN_NOT_YET_FROZEN_BENCHMARK','source':'BindingDB provisional Kd candidate cohort','selection':'SHA256 component-id hashing only, independent of labels','existing_dev_validation_components_quarantined':len(old_val),'existing_dev_validation_pairs_quarantined':sum(counts[g] for g in old_val),'candidate_training_components':len(train_groups),'candidate_training_pairs':sum(counts[g] for g in train_groups),'new_test_components':len(new_test),'new_test_pairs':sum(counts[g] for g in new_test),'test_component_ids_sha256':hashlib.sha256('\n'.join(sorted(new_test)).encode()).hexdigest(),'test_component_ids':sorted(new_test),'training_component_ids':sorted(train_groups),'limitations':['Prior training may have included the new test components; therefore existing trained weights MUST NOT be evaluated on this split. Retrain all candidate models from scratch using only new training components.','Existing dev validation components are quarantined due to repeated inspection.','This is not independent of earlier dataset-wide QC decisions.','Protein construct and assay verification incomplete.','Do not inspect test labels or metrics until model selection and hyperparameters are locked.']}
(OUT/'bindingdb_prospective_component_split_design_dev.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('test_component_ids','training_component_ids')},indent=2))
