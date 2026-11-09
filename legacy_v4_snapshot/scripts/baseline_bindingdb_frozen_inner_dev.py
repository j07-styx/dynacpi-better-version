"""Train-only baseline for frozen inner-development partition. Never reads test labels."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data/audit';R=ROOT/'research'
def rows(p):
 with p.open(encoding='utf8') as f:
  for line in f:yield json.loads(line)
m=json.loads((R/'bindingdb_frozen_component_split_manifest_dev.json').read_text());tr=set(m['train_components']);dv=set(m['inner_dev_components']);test=set(m['test_components']);assert not tr&dv and not tr&test and not dv&test
comp={}
for g in rows(A/'bindingdb_full_kd_protein_components.jsonl'):
 for s in g['members']:comp[s]=g['component_id']
tasks={(r['sequence_sha256'],r['assay_key']) for r in rows(A/'bindingdb_full_kd_task_summary.jsonl') if r['unique_canonical_compounds']>=15}
train_y=[];dev_y=[];dev_comp=[]
for r in rows(A/'bindingdb_full_kd_compound_median_candidates.jsonl'):
 if (r['protein_seq_sha256'],r['assay_key']) not in tasks or r['range_pKd']>.1:continue
 c=comp[r['protein_seq_sha256']]
 if c in tr:train_y.append(r['median_pKd'])
 elif c in dv:dev_y.append(r['median_pKd']);dev_comp.append(c)
assert len(train_y)==m['train_pairs'] and len(dev_y)==m['inner_dev_pairs']
mean=float(np.mean(train_y));y=np.asarray(dev_y);pred=np.full_like(y,mean)
per={c:float(np.sqrt(np.mean((y[np.array(dev_comp)==c]-mean)**2))) for c in dv}
out={'status':'FROZEN_INNER_DEV_BASELINE_NO_TEST_ACCESS','train_pairs':len(train_y),'inner_dev_pairs':len(dev_y),'train_label_mean':mean,'train_label_std':float(np.std(train_y)),'inner_dev_label_mean':float(np.mean(dev_y)),'inner_dev_label_std':float(np.std(dev_y)),'train_mean_pooled_rmse':float(np.sqrt(np.mean((pred-y)**2))),'train_mean_component_macro_rmse':float(np.mean(list(per.values()))),'inner_dev_components':len(dv),'note':'No reserved-test records or labels were used.'}
(R/'bindingdb_frozen_inner_dev_mean_baseline.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
