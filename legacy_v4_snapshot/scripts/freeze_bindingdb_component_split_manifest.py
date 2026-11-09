"""Lock component-level train/dev/test membership without looking at test labels."""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research';A=ROOT/'data/audit'
split=json.loads((R/'bindingdb_prospective_component_split_design_dev.json').read_text())
train_candidates=split['training_component_ids'];test=set(split['test_component_ids'])
inner_dev=set(sorted(train_candidates,key=lambda s:hashlib.sha256(('dynacpi-inner-dev-20261108:'+s).encode()).hexdigest())[:max(1,len(train_candidates)//5)])
train=set(train_candidates)-inner_dev
assert train.isdisjoint(inner_dev|test) and inner_dev.isdisjoint(test)
# Component sizes are recorded for reproducibility; no labels or prediction metrics accessed.
components={}
with (A/'bindingdb_full_kd_protein_components.jsonl').open() as f:
 for line in f:
  obj=json.loads(line)
  for seq in obj['members']:components[seq]=obj['component_id']
tasks=set()
with (A/'bindingdb_full_kd_task_summary.jsonl').open() as f:
 for line in f:
  obj=json.loads(line)
  if obj['unique_canonical_compounds']>=15:tasks.add((obj['sequence_sha256'],obj['assay_key']))
counts=Counter()
with (A/'bindingdb_full_kd_compound_median_candidates.jsonl').open() as f:
 for line in f:
  obj=json.loads(line)
  if (obj['protein_seq_sha256'],obj['assay_key']) in tasks and obj['range_pKd']<=.1:
   counts[components[obj['protein_seq_sha256']]]+=1
out={'status':'FROZEN_SPLIT_MANIFEST_DESIGN_NO_TEST_EVALUATION','hash_algorithm':'SHA256 component ID sorting','train_components':sorted(train),'inner_dev_components':sorted(inner_dev),'test_components':sorted(test),'train_component_count':len(train),'inner_dev_component_count':len(inner_dev),'test_component_count':len(test),'train_pairs':sum(counts[g] for g in train),'inner_dev_pairs':sum(counts[g] for g in inner_dev),'test_pairs':sum(counts[g] for g in test),'quarantined_old_dev_components':split['existing_dev_validation_components_quarantined'],'protocol':['All new model weights must be initialized from scratch.','Train only on train_components.','Tune hyperparameters and early stopping only on inner_dev_components.','Lock architecture and checkpoint selection before first test evaluation.','Evaluate test_components once and disclose all repeated looks if any.','Old dev split results must not be reported as independent test results.'],'warning':'Dataset and assay QC remain provisional; historical experiments previously trained on some designated test components, so only fresh retraining qualifies.'}
(R/'bindingdb_frozen_component_split_manifest_dev.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if not k.endswith('_components') or k.endswith('_count')},indent=2))
