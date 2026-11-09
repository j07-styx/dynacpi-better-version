"""Protein-component-level baseline audit for provisional BindingDB development split."""
import hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data/audit'
def rows(path):
 with path.open(encoding='utf8') as f:
  for line in f:yield json.loads(line)
components={}
for g in rows(A/'bindingdb_full_kd_protein_components.jsonl'):
 for s in g['members']:components[s]=g['component_id']
tasks={(r['sequence_sha256'],r['assay_key']) for r in rows(A/'bindingdb_full_kd_task_summary.jsonl') if r['unique_canonical_compounds']>=15}
samples=[r for r in rows(A/'bindingdb_full_kd_compound_median_candidates.jsonl') if (r['protein_seq_sha256'],r['assay_key']) in tasks and r['range_pKd']<=.1]
groups=sorted({components[r['protein_seq_sha256']] for r in samples});held=set(sorted(groups,key=lambda s:hashlib.sha256(('bindingdb-dev-20261107:'+s).encode()).hexdigest())[:len(groups)//5])
train=[r for r in samples if components[r['protein_seq_sha256']] not in held];val=[r for r in samples if components[r['protein_seq_sha256']] in held]
mean=float(np.mean([r['median_pKd'] for r in train]));by_comp=defaultdict(list)
for r in val:by_comp[components[r['protein_seq_sha256']]].append(r)
per=[]
for c,rr in by_comp.items():
 y=np.array([r['median_pKd'] for r in rr]);per.append({'component_id':c,'n':len(rr),'mean_label':float(y.mean()),'mean_baseline_rmse':float(np.sqrt(np.mean((y-mean)**2)))})
per.sort(key=lambda x:x['n'],reverse=True)
all_y=np.array([r['median_pKd'] for r in val]);out={'status':'PROVISIONAL_COMPONENT_WEIGHTING_AUDIT','train_only_label_mean':mean,'validation_pairs':len(val),'validation_components':len(per),'pooled_mean_baseline_rmse':float(np.sqrt(np.mean((all_y-mean)**2))),'component_macro_mean_baseline_rmse':statistics.mean(x['mean_baseline_rmse'] for x in per),'largest_component_fraction':per[0]['n']/len(val),'top_five_component_fraction':sum(x['n'] for x in per[:5])/len(val),'component_size_min':min(x['n'] for x in per),'component_size_median':statistics.median(x['n'] for x in per),'component_size_max':max(x['n'] for x in per),'components':per,'warning':'Pooled RMSE can be dominated by large components; model evaluation should include component macro RMSE and per-component metrics.'}
(ROOT/'research/bindingdb_provisional_component_weighting_audit_dev.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='components'},indent=2))
